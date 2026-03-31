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

    PokerNet global_net(24, 128);
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

        Rules rules;
        rules.buyIn = 1000;
        rules.bigBlind = 10;
        rules.smallBlind = 5;
        rules.allowRebuy = false;
        rules.fixedNumberOfDeals = hands_per_epoch;

        HostSilent host;
        Game game(&host);
        game.setRules(rules);
        game.setSilent(true);

        game.addObserverBorrowed(&dashboard);

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
            game.addPlayer(Player(new AIBorrowed(agent1), "RL_Agent"));
            game.addPlayer(Player(new AIBorrowed(agent2), agent2->getAIName()));
        } else {
            // Self-play phase
            agent2 = new AIRL(global_net, optimizer, 1000.0f, noise, ent_coeff);
            dashboard.setPhase("Self-Play", "AIRL (Self)");
            game.addPlayer(Player(new AIBorrowed(agent1), "RL_Agent"));
            game.addPlayer(Player(new AIBorrowed(agent2), "Opponent"));
        }

        game.doGame();
        
        float agent_stack = static_cast<float>(game.getFinalStack("RL_Agent"));
        float opp_stack = (distillation_complete) ? static_cast<float>(game.getFinalStack("Opponent")) : static_cast<float>(game.getFinalStack(agent2->getAIName()));

        float epoch_loss = agent1->applyEpochReward(agent_stack);
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

        dashboard.endEpoch(agent_stack, opp_stack, epoch_loss, lr, noise, top_features);
        dashboard.render();

        if (!distillation_complete) {
            win_window.push_back((agent_stack > 1000) ? 1 : 0);
            if (win_window.size() > window_size) win_window.pop_front();

            if (win_window.size() == window_size) {
                int wins = 0;
                for (int w : win_window) wins += w;
                float win_rate = static_cast<float>(wins) / window_size;
                if (win_rate >= 0.70f) {
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

