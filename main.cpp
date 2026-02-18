/*
OOPoker

Copyright (c) 2010 Lode Vandevenne
All rights reserved.

This file is part of OOPoker.

OOPoker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OOPoker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OOPoker.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
OOPoker, or "Object Oriented Poker", is a C++ No-Limit Texas Hold'm engine meant
to be used to implement poker AIs for entertainment  or research purposes. These
AIs can be made to battle each other, or a single human can play against the AIs
for his/her enjoyment.
*/

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstdlib>

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
#include <torch/torch.h>
#include "poker_net.h"
#include "ai_rl.h"

int main()
{
    // Create logs directory
    system("mkdir -p ./logs/ga");

    // Configure the genetic algorithm (paper Table 1)
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

    // Optional: resume from a saved population
    // trainer.loadPopulation("./logs/ga/population_gen_100.bin");

    trainer.train();

    return 0;
}
