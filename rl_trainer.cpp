#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <deque>
#include <torch/torch.h>

#include "rl_trainer.h"
#include "ai.h"
#include "ai_rl.h"
#include "ai_smart.h"
#include "ai_random.h"
#include "ai_evolved.h"
#include "ai_call.h"
#include "game.h"
#include "player.h"
#include "host_silent.h"
#include "poker_net.h"
#include "checkpoint.h"
#include "rl_dashboard.h"

static void loadBestEvolved(AIEvolved* evolved) {
    std::ifstream file("./build/logs/ga/best_final.bin", std::ios::binary);
    if (!file.good()) return;
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(int));
    std::vector<float> genome(size);
    file.read(reinterpret_cast<char*>(genome.data()), size * sizeof(float));
    evolved->setGenome(genome);
}

void runRLTraining(const std::string& checkpoint_path)
{
    system("mkdir -p ./logs/rl");

    PokerNet global_net(26, 128);
    float lr_start = 1e-3f;
    float lr_end = 1e-4f;
    // weight_decay provides L2 regularization: prevents weights from drifting to
    // extreme values and causing the policy-collapse / NaN-gradient cycle
    auto adam_opts = torch::optim::AdamOptions(lr_start).weight_decay(1e-4);
    torch::optim::Adam optimizer(global_net->parameters(), adam_opts);
    CheckpointManager cp_manager("./logs/rl/rl_poker_model", 100);

    // Load weights if resuming
    if (!checkpoint_path.empty()) {
        if (cp_manager.load_checkpoint(global_net, checkpoint_path)) {
            std::cout << "[Resume] Loaded weights from: " << checkpoint_path << "\n";
        } else {
            std::cout << "[Resume] Load failed — starting from scratch.\n";
        }
    }
    int num_epochs = 10000; // Increased for longer distillation + self-play
    int hands_per_epoch = 500;

    RLDashboard dashboard;
    dashboard.init(num_epochs, hands_per_epoch);

    bool distillation_complete = false;
    std::deque<int> win_window; // 1 for win, 0 for loss
    const int window_size = 5;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // LR and Noise calculation for the start of the epoch
        float lr = lr_start * std::pow(lr_end / lr_start, static_cast<float>(epoch) / std::max(1, num_epochs - 1));
        
        float noise = 0.0f;
        if (epoch < 1000) {
            noise = 0.3f * std::pow(0.01f / 0.3f, static_cast<float>(epoch) / 999.0f);
        }

        float ent_coeff = 0.001f;
        if (epoch >= 1000 && epoch < 3000) {
            // Decay entropy bonus from 0.001 → 0.0001 (never fully zero)
            // A residual entropy prevents permanent policy collapse
            ent_coeff = 0.001f * std::pow(0.1f, static_cast<float>(epoch - 1000) / 2000.0f);
        } else if (epoch >= 3000) {
            ent_coeff = 0.0001f; // floor: always nudge away from certainty
        }

        dashboard.beginEpoch(epoch, lr, noise);

        AIRL* agent1 = new AIRL(global_net, optimizer, 1000.0f, noise, ent_coeff);
        AI* agent2 = nullptr;

        if (!distillation_complete) {
            // Distillation phase: rotate opponents
            int opp_type = epoch % 4;
            if (opp_type == 0) agent2 = new AISmart();
            else if (opp_type == 1) agent2 = new AIRandom();
            else if (opp_type == 2) {
                AIEvolved* ev = new AIEvolved();
                loadBestEvolved(ev);
                agent2 = ev;
            }
            else agent2 = new AICall(); // "ai_human" placeholder

            dashboard.setPhase("Distillation", agent2->getAIName());
        } else {
            // Self-play phase
            agent2 = new AIRL(global_net, optimizer, 1000.0f, noise, ent_coeff);
            dashboard.setPhase("Self-Play", "AIRL (Self)");
        }

        // --- Inner sub-game loop: keeps running games until 500 hands are played ---
        int hands_played = 0;
        float epoch_net_chips = 0.0f; // cumulative agent net chips this epoch
        int epoch_sub_wins = 0;       // how many sub-games the agent won
        int epoch_sub_games = 0;      // how many sub-games were played total
        int epoch_opp_busts = 0;      // how many times the opponent lost all chips
        float epoch_loss = 0.0f;

        while (hands_played < hands_per_epoch) {
            int remaining = hands_per_epoch - hands_played;

            Rules sub_rules;
            sub_rules.buyIn = 1000;
            sub_rules.bigBlind = 10;
            sub_rules.smallBlind = 5;
            sub_rules.allowRebuy = false;
            sub_rules.fixedNumberOfDeals = remaining;

            HostSilent sub_host;
            Game sub_game(&sub_host);
            sub_game.setRules(sub_rules);
            sub_game.setSilent(true);
            sub_game.addObserverBorrowed(&dashboard);

            sub_game.addPlayer(Player(new AIBorrowed(agent1), "RL_Agent"));
            if (!distillation_complete) {
                sub_game.addPlayer(Player(new AIBorrowed(agent2), agent2->getAIName()));
            } else {
                sub_game.addPlayer(Player(new AIBorrowed(agent2), "Opponent"));
            }

            sub_game.doGame();

            int dealt = sub_game.getNumDeals();
            hands_played += dealt;

            // Sub-game result: read final stacks
            float sg_agent = static_cast<float>(sub_game.getFinalStack("RL_Agent"));
            float sg_opp = (!distillation_complete)
                ? static_cast<float>(sub_game.getFinalStack(agent2->getAIName()))
                : static_cast<float>(sub_game.getFinalStack("Opponent"));

            // Accumulate: net chips gained/lost vs starting 1000
            epoch_net_chips += (sg_agent - 1000.0f);

            // Win/loss: agent had more chips at end of sub-game
            epoch_sub_games++;
            if (sg_agent > sg_opp) epoch_sub_wins++;
            if (sg_opp == 0) epoch_opp_busts++;
        }

        // Average net chips per sub-game, mapped back to a 1000-chip buy-in reference.
        // Dividing by sub-game count prevents unbounded growth when many sub-games run per epoch
        // (e.g. opponent busts quickly, resetting stacks multiple times).
        float avg_net   = (epoch_sub_games > 0) ? epoch_net_chips / epoch_sub_games : 0.0f;
        float agent_stack = 1000.0f + avg_net;
        float opp_stack   = 2000.0f - agent_stack;

        epoch_loss = agent1->applyEpochReward(agent_stack);
        if (distillation_complete) {
            static_cast<AIRL*>(agent2)->applyEpochReward(opp_stack);
        }

        torch::nn::utils::clip_grad_norm_(global_net->parameters(), 1.0);
        optimizer.step();
        optimizer.zero_grad();

        // Fetch saliency BEFORE updating dashboard
        auto top_features = agent1->getTopFeatures();

        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
        }

        dashboard.endEpoch(agent_stack, opp_stack, epoch_loss, lr, noise, top_features, epoch_sub_wins, epoch_sub_games, epoch_opp_busts);
        dashboard.render();

        if (!distillation_complete) {
            // Epoch won if agent won majority of sub-games  
            int epoch_won = (epoch_sub_wins * 2 > epoch_sub_games) ? 1 : 0;
            win_window.push_back(epoch_won);
            if (win_window.size() > window_size) win_window.pop_front();

            if (win_window.size() == window_size) {
                int wins = 0;
                for (int w : win_window) wins += w;
                float win_rate = static_cast<float>(wins) / window_size;
                if (win_rate >= 0.80f) {
                    std::cout << "\n>>> Distillation complete! Switching to self-play at epoch " << epoch << " <<<\n";
                    distillation_complete = true;
                }
            }
        }

        // We explicitly delete agent2 since Game only deleted the AIBorrowed wrapper
        if (agent2) {
            delete agent2;
            agent2 = nullptr;
        }

        if (epoch % 10 == 0) {
            cp_manager.save_checkpoint(global_net, epoch);
            cp_manager.run_evaluation(global_net, epoch);
        }

        if (agent1) {
            delete agent1;
            agent1 = nullptr;
        }
    }

    // Training complete: save final checkpoint and write summary
    cp_manager.save_checkpoint(global_net, num_epochs);
    dashboard.writeFinalSummary("FINAL_DASHBOARD_SUMMARY.txt");
}

