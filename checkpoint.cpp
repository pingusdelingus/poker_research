#include "player.h"
#include "checkpoint.h"
#include "game.h"
#include "ai_rl.h"
#include "ai_smart.h"
#include "host_silent.h"
#include "observer_dashboard.h"
#include <fstream>
#include <iostream>

CheckpointManager::CheckpointManager(std::string path, int freq)
    : base_path(path), frequency(freq) {}

void CheckpointManager::run_evaluation(PokerNet& net, int epoch, ObserverDashboard* dashboard) {
    torch::NoGradGuard no_grad;
    net->eval();

    HostSilent host;
    Game eval_game(&host);
    eval_game.setSilent(true);

    torch::optim::Adam dummy_opt(net->parameters(), 1e-4);

    AIRL* test_bot = new AIRL(net, dummy_opt);
    AISmart* baseline = new AISmart(0.5);

    eval_game.addPlayer(Player(test_bot, "EvalBot"));
    eval_game.addPlayer(Player(baseline, "Baseline"));

    Rules rules;
    rules.buyIn = 1000;
    rules.fixedNumberOfDeals = 50;
    eval_game.setRules(rules);

    eval_game.doGame();

    int bot_final_stack = eval_game.getFinalStack("EvalBot");

    std::ofstream log("training_log.csv", std::ios::app);
    log << epoch << "," << bot_final_stack << "\n";
    log.close();

    if(dashboard) {
        dashboard->addEvalResult(epoch, bot_final_stack);
    }

    net->train();
}

void CheckpointManager::save_checkpoint(PokerNet& net, int epoch) {
    std::string filename = "./logs/epoch_" + std::to_string(epoch) + ".pt";
    torch::save(net, filename);
}
