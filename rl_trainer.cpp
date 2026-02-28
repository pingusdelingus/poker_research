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

// RL Self-Play Trainer

/*
 * old
void runRLTraining()
{
    std::cout << "Starting RL Self-Play Training (REINFORCE)..." << std::endl;
    system("mkdir -p ./logs/rl");
    
    // 1. Initialize network and optimizer
    PokerNet global_net(23, 128); // 23 static, 128 hidden
    torch::optim::Adam optimizer(global_net->parameters(), 1e-4);
    CheckpointManager cp_manager("./logs/rl/rlnoGA_poker_model", 100);

    // 2. Training Loop
    for(int epoch = 0; epoch < 1000; epoch++) {
        Rules rules;
        rules.buyIn = 1000;
        rules.bigBlind = 10;
        rules.smallBlind = 5;
        rules.allowRebuy = true;
        rules.fixedNumberOfDeals = 1000;

        HostSilent host;
        Game game(&host);
        game.setRules(rules);
        game.setSilent(true);

        // Self-play: two RL agents sharing same net
        AIRL* agent1 = new AIRL(global_net, optimizer);
        AIRL* agent2 = new AIRL(global_net, optimizer);

        game.addPlayer(Player(agent1, "RL_Agent_A"));
        game.addPlayer(Player(agent2, "RL_Agent_B"));

        game.doGame();
        
        // Finalize epoch rewards (accumulate gradients)
        agent1->applyEpochReward(0.0f);
        agent2->applyEpochReward(0.0f);

        // Perform single, combined update for both agents
        torch::nn::utils::clip_grad_norm_(global_net->parameters(), 1.0);
        optimizer.step();
        optimizer.zero_grad();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " complete." << std::endl;
            cp_manager.save_checkpoint(global_net, epoch);
            cp_manager.run_evaluation(global_net, epoch);
        }
    }
}

*/
#include "ga_dashboard.h" // Using your existing GA dashboard


#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include "ga_dashboard.h"

void runRLTraining()
{
    std::cout << "Starting RL Self-Play Training (REINFORCE)..." << std::endl;
    system("mkdir -p ./logs/rl");
    
    PokerNet global_net(23, 128); 
    torch::optim::Adam optimizer(global_net->parameters(), 1e-4);
    CheckpointManager cp_manager("./logs/rl/rlnoGA_poker_model", 100);

    GADashboard dashboard;
    int num_epochs = 1000;
    int population_size = 2;
    dashboard.init(num_epochs, population_size, 128, 1000); 

    for(int epoch = 0; epoch < num_epochs; epoch++) {
        dashboard.beginGeneration(epoch, 0.0f, 0.0f);

        Rules rules;
        rules.buyIn = 1000;
        rules.bigBlind = 10;
        rules.smallBlind = 5;
        rules.allowRebuy = true;
        rules.fixedNumberOfDeals = 1000;

        HostSilent host;
        Game game(&host);
        game.setRules(rules);
        game.setSilent(true);

        AIRL* agent1 = new AIRL(global_net, optimizer);
        AIRL* agent2 = new AIRL(global_net, optimizer);

        game.addPlayer(Player(agent1, "RL_Agent_A"));
        game.addPlayer(Player(agent2, "RL_Agent_B"));

        game.doGame();
       
        float reward1 = static_cast<float>(game.getFinalStack("RL_Agent_A"));
        float reward2 = static_cast<float>(game.getFinalStack("RL_Agent_B"));

        std::vector<Individual> rl_population;
        
        Individual ind1;
        ind1.fitness = reward1;
        rl_population.push_back(ind1);

        Individual ind2;
        ind2.fitness = reward2;
        rl_population.push_back(ind2);

        std::sort(rl_population.begin(), rl_population.end(),
                  [](const Individual& a, const Individual& b) {
                      return a.fitness > b.fitness;
                  });

        float avg_reward = (reward1 + reward2) / 2.0f;

        agent1->applyEpochReward(reward1);
        agent2->applyEpochReward(reward2);

        torch::nn::utils::clip_grad_norm_(global_net->parameters(), 1.0);
        optimizer.step();
        optimizer.zero_grad();

        dashboard.setPopulationResults(rl_population, avg_reward, 1, 2);
        dashboard.render();

        if (epoch % 10 == 0) {
            cp_manager.save_checkpoint(global_net, epoch);
            cp_manager.run_evaluation(global_net, epoch);
        }

    }
}

