#include "player.h"
#include "checkpoint.h"
#include "game.h"
#include "ai_rl.h"
#include "ai_smart.h"
#include "host_silent.h"
#include "observer_dashboard.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

CheckpointManager::CheckpointManager(std::string path, int freq)
    : base_path(path), frequency(freq) {}

void CheckpointManager::run_evaluation(PokerNet& net, int epoch, ObserverDashboard* dashboard) {
    torch::NoGradGuard no_grad;
    net->eval();

    HostSilent host;
    Game eval_game(&host);
    eval_game.setSilent(true);

    torch::optim::Adam dummy_opt(net->parameters(), 1e-4);

    AIRL* test_bot = new AIRL(net, dummy_opt, 1000.0f, /*noise=*/0.0f, /*ent_coeff=*/0.0f);
    AISmart* baseline = new AISmart(0.5);

    eval_game.addPlayer(Player(new AIBorrowed(test_bot), "EvalBot"));
    eval_game.addPlayer(Player(new AIBorrowed(baseline), "Baseline"));

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

    delete test_bot;
    delete baseline;

    net->train();
}

void CheckpointManager::save_checkpoint(PokerNet& net, int epoch) {
    std::string filename = base_path + "_epoch_" + std::to_string(epoch) + ".pt";
    torch::save(net, filename);
}

std::string CheckpointManager::find_latest_checkpoint() const {
    namespace fs = std::filesystem;
    std::string dir = "./logs/rl/";
    std::string prefix = "rl_poker_model_epoch_";

    std::vector<std::pair<int, std::string>> found;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) == 0 && entry.path().extension() == ".pt") {
                // Extract epoch number from filename
                std::string num_str = name.substr(prefix.size());
                num_str = num_str.substr(0, num_str.find('.'));
                try {
                    int ep = std::stoi(num_str);
                    found.push_back({ep, entry.path().string()});
                } catch (...) {}
            }
        }
    } catch (...) {}

    if (found.empty()) return "";
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
    return found[0].second;
}

bool CheckpointManager::load_checkpoint(PokerNet& net, const std::string& path) const {
    try {
        torch::load(net, path);
        return true;
    } catch (const c10::Error& e) {
        std::string msg = e.what();
        if (msg.find("shapes cannot be multiplied") != std::string::npos ||
            msg.find("size mismatch") != std::string::npos) {
            std::cerr << "\n[Checkpoint] Architecture mismatch: the saved model was trained\n"
                      << "  with a different input size than the current network (input size changed).\n"
                      << "  Delete old checkpoints in ./logs/rl/ and start fresh.\n";
        } else {
            std::cerr << "[Checkpoint] Failed to load \"" << path << "\": " << e.what() << "\n";
        }
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[Checkpoint] Failed to load \"" << path << "\": " << e.what() << "\n";
        return false;
    }
}
