#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <torch/torch.h>

#include "rl_trainer.h"
#include "ai.h"
#include "ai_rl.h"
#include "game.h"
#include "player.h"
#include "host_silent.h"
#include "poker_net.h"
#include "checkpoint.h"
#include "rl_dashboard.h"

void runRLTraining()
{
    system("mkdir -p ./logs/rl");

    PokerNet global_net(23, 128);
    float lr_start = 5e-4f;
    float lr_end = 1e-4f;
    torch::optim::Adam optimizer(global_net->parameters(), lr_start);
    CheckpointManager cp_manager("./logs/rl/rl_poker_model", 100);

    int num_epochs = 1000;
    int hands_per_epoch = 1000;

    RLDashboard dashboard;
    dashboard.init(num_epochs, hands_per_epoch);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        dashboard.beginEpoch(epoch);

        Rules rules;
        rules.buyIn = 1000;
        rules.bigBlind = 10;
        rules.smallBlind = 5;
        rules.allowRebuy = true;
        rules.fixedNumberOfDeals = hands_per_epoch;

        HostSilent host;
        Game game(&host);
        game.setRules(rules);
        game.setSilent(true);

        // Attach dashboard as observer for live updates + play style tracking
        game.addObserverBorrowed(&dashboard);

        // Exponential noise decay: 0.3 → 0.01 over first 200 epochs, then 0
        float noise = 0.0f;
        if (epoch < 200) {
            noise = 0.3f * std::pow(0.01f / 0.3f, static_cast<float>(epoch) / 199.0f);
        }

        // Entropy coeff: 0.001 for first 300 epochs, linear decay to 0 by epoch 600
        float ent_coeff = 0.001f;
        if (epoch >= 300 && epoch < 600) {
            ent_coeff = 0.001f * (1.0f - static_cast<float>(epoch - 300) / 300.0f);
        } else if (epoch >= 600) {
            ent_coeff = 0.0f;
        }

        AIRL* agent1 = new AIRL(global_net, optimizer, 1000.0f, noise, ent_coeff);
        AIRL* agent2 = new AIRL(global_net, optimizer, 1000.0f, noise, ent_coeff);

        game.addPlayer(Player(agent1, "Evolved"));
        game.addPlayer(Player(agent2, "Opponent"));

        game.doGame();

        float agent_stack = static_cast<float>(game.getFinalStack("Evolved"));
        float opp_stack = static_cast<float>(game.getFinalStack("Opponent"));

        agent1->applyEpochReward(agent_stack);
        agent2->applyEpochReward(opp_stack);

        torch::nn::utils::clip_grad_norm_(global_net->parameters(), 1.0);
        optimizer.step();
        optimizer.zero_grad();

        // Exponential LR decay: 1e-3 → 1e-5 over training
        float lr = lr_start * std::pow(lr_end / lr_start, static_cast<float>(epoch) / std::max(1, num_epochs - 1));
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
        }

        dashboard.endEpoch(agent_stack, opp_stack, 0.0f, lr, noise);
        dashboard.render();

        if (epoch % 10 == 0) {
            cp_manager.save_checkpoint(global_net, epoch);
            cp_manager.run_evaluation(global_net, epoch);
            // Record eval result in dashboard
            // (run_evaluation logs to CSV; we grab the stack from the game)
        }
    }
}
