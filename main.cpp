#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>

#include "genetic_trainer.h"
#include "ai.h"
#include "ai_blindlimp.h"
#include "ai_call.h"
#include "ai_checkfold.h"
#include "ai_human.h"
#include "ai_raise.h"
#include "ai_random.h"
#include "ai_smart.h"
#include "card.h"
#include "combination.h"
#include "game.h"
#include "host_terminal.h"
#include "host_silent.h"
#include "info.h"
#include "io_terminal.h"
#include "observer.h"
#include "observer_terminal.h"
#include "observer_terminal_quiet.h"
#include "observer_dashboard.h"
#include "observer_log.h"
#include "pokermath.h"
#include "random.h"
#include "table.h"
#include "tools_terminal.h"
#include "unittest.h"
#include "rl_trainer.h"
#include "checkpoint.h"

int main(int argc, char** argv)
{
    // Create logs directory
    system("mkdir -p ./logs/ga");

    // Parse CLI flags
    bool runGA = false;
    bool runUT = false;
    if (argc > 1 && strcmp(argv[1], "--ga") == 0) {
        runGA = true;
    }
    if (argc > 1 && strcmp(argv[1], "--unittest") == 0) {
        runUT = true;
    }

    if (runUT) {
        doUnitTest();
        return 0;
    }

    if (runGA) {
        GeneticTrainer::Config config;
        config.population_size       = 50;
        config.num_generations       = 250;
        config.hands_per_session     = 500;
        config.survival_rate         = 0.30f;
        config.mutation_rate_start   = 0.25f;
        config.mutation_rate_end     = 0.05f;
        config.mutation_strength_start = 0.50f;
        config.mutation_strength_end   = 0.10f;
        config.buy_in     = 1000;
        config.big_blind  = 10;
        config.small_blind = 5;
        config.log_dir    = "./logs/ga/";
        config.checkpoint_interval = 10;

        GeneticTrainer trainer(config);
        trainer.train();
        return 0;
    }

    // Default: RL training (REINFORCE with PokerNet)
    // ── Startup Menu ────────────────────────────────────────────────────────
    CheckpointManager menu_cp("./logs/rl/rl_poker_model", 100);
    std::string latest = menu_cp.find_latest_checkpoint();

    std::string checkpoint_to_load = "";

    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════╗\n";
    std::cout << "  ║        PokerNet  ─  Training Setup       ║\n";
    std::cout << "  ╚══════════════════════════════════════════╝\n\n";

    if (latest.empty()) {
        // No saved model found — start fresh automatically
        std::cout << "  No saved checkpoint found.\n";
        std::cout << "  Starting fresh training...\n\n";
    } else {
        std::cout << "  Latest checkpoint: " << latest << "\n\n";
        std::cout << "  [1]  Resume from checkpoint\n";
        std::cout << "  [2]  Start fresh (discards saved weights)\n";
        std::cout << "  [3]  Load a specific checkpoint file\n";
        std::cout << "\n  Choice: ";

        int choice = 0;
        std::cin >> choice;
        std::cin.ignore();

        if (choice == 1) {
            checkpoint_to_load = latest;
            std::cout << "\n  Resuming from: " << latest << "\n\n";
        } else if (choice == 3) {
            std::cout << "  Enter checkpoint file path: ";
            std::getline(std::cin, checkpoint_to_load);
            // Trim surrounding whitespace/quotes the user might paste
            while (!checkpoint_to_load.empty() && (checkpoint_to_load.front() == '"' || checkpoint_to_load.front() == ' '))
                checkpoint_to_load.erase(checkpoint_to_load.begin());
            while (!checkpoint_to_load.empty() && (checkpoint_to_load.back() == '"' || checkpoint_to_load.back() == ' '))
                checkpoint_to_load.pop_back();
            std::cout << "\n  Loading: " << checkpoint_to_load << "\n\n";
        } else {
            std::cout << "\n  Starting fresh training...\n\n";
        }
    }
    // ── End Menu ─────────────────────────────────────────────────────────────

    runRLTraining(checkpoint_to_load);

    return 0;
}
