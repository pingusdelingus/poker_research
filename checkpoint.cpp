#include "checkpoint.h"
#include "game.h"
#include "ai_rl.h"
#include "ai_smart.h"
#include "host_terminal.h"
#include "observer_terminal_quiet.h"
#include <fstream>
#include <iostream>
#include "player.h"


CheckpointManager::CheckpointManager(std::string path, int freq) 
    : base_path(path), frequency(freq) {}

void CheckpointManager::run_evaluation(PokerNet& net, int epoch) {
    torch::NoGradGuard no_grad; 
    net->eval(); // set to evaluation mode

    HostTerminal host;
    Game eval_game(&host);
    
    // create a dummy optimizer for the constructor
    torch::optim::Adam dummy_opt(net->parameters(), 1e-4);
    
    // test bot vs a smart baseline
    AIRL* test_bot = new AIRL(net, dummy_opt);
    AISmart* baseline = new AISmart(0.5);

    eval_game.addPlayer(Player(test_bot, "EvalBot"));
    eval_game.addPlayer(Player(baseline, "Baseline"));

    Rules rules;
    rules.buyIn = 1000;
    rules.fixedNumberOfDeals = 50; // run 50 hands for a quick check
    eval_game.setRules(rules);

    std::cout << "\n--- [CHECKPOINT EVALUATION] EPOCH " << epoch << " ---" << std::endl;
    eval_game.doGame();

    // extract results from the game
    int bot_final_stack = eval_game.getFinalStack("EvalBot");
    
    std::ofstream log("training_log.csv", std::ios::app);
    log << epoch << "," << bot_final_stack << "\n";
    log.close();

    std::cout << "Evaluation Finished. Bot Stack: " << bot_final_stack << std::endl;
    
    net->train(); // return to training mode
}

void CheckpointManager::save_checkpoint(PokerNet& net, int epoch) {
    std::string filename = base_path + "/logs/_epoch_" + std::to_string(epoch) + ".pt";
    torch::save(net, filename);
    std::cout << "Model saved to " << filename << std::endl;
}
