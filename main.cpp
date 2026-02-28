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

int main(int argc, char** argv)
{
    // Create logs directory
    system("mkdir -p ./logs/ga");

    // Check for CLI flag to run RL training
    bool runRL = false;
    bool runUT = false;
    if (argc > 1 && strcmp(argv[1], "--rl") == 0) {
        runRL = true;
    }
    if (argc > 1 && strcmp(argv[1], "--unittest") == 0) {
        runUT = true;
    }

    if (runUT) {
        doUnitTest();
        return 0;
    }

    if (runRL) {
        runRLTraining();
        return 0;
    }

    // Default to Genetic Algorithm (dashboard branch behavior)
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

    // Create and run trainer
    GeneticTrainer trainer(config);

    trainer.train();

    return 0;
}
