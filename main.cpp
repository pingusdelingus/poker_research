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
Linux compile command:
g++ *.cpp -W -Wall -Wextra -ansi -O3
g++ *.cpp -W -Wall -Wextra -ansi -g3
*/


/*
OOPoker, or "Object Oriented Poker", is a C++ No-Limit Texas Hold'm engine meant
to be used to implement poker AIs for entertainment  or research purposes. These
AIs can be made to battle each other, or a single human can play against the AIs
for his/her enjoyment.
*/

//In all functions below, when cards are sorted, it's always from high to low

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>


#include "checkpoint.h"
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

// returns whether user wants to quit

bool doGame(PokerNet& net, torch::optim::Optimizer& optimizer, ObserverDashboard* dashboard, int epoch)
{
  // hard coding RL self play
  char c = '6';
  int gameType = (c == '6') ? 6 : (c - '0');
  if(c == 'q') return true;

  Rules rules;
  rules.buyIn = 1000;
  rules.bigBlind = 10;
  rules.smallBlind = 5;
  rules.allowRebuy = false;
  rules.fixedNumberOfDeals = (gameType == 6) ? 1000 : 100;

  HostSilent silent_host;
  HostTerminal terminal_host;
  Host* host = (gameType == 6) ? (Host*)&silent_host : (Host*)&terminal_host;

  Game game(host);
  game.setRules(rules);

  if(gameType == 6) // RL Self-Play Training
  {
    game.setSilent(true);
    dashboard->setEpoch(epoch);
    game.addObserverBorrowed(dashboard);

    AIRL* agent1 = new AIRL(net, optimizer);
    AIRL* agent2 = new AIRL(net, optimizer);

    game.addPlayer(Player(agent1, "RL_Agent_A"));
    game.addPlayer(Player(new AISmart() , "OOPoker Bot"));
  }
 if(gameType == 1) //Human + AI's
  {
    game.addPlayer(Player(new AIHuman(&terminal_host), "You"));

    //choose the AI players here
    game.addPlayer(Player(new AIRandom(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
  }
  else if(gameType == 2) //Human + AI heads-up
  {
    game.addPlayer(Player(new AIHuman(&terminal_host), "You"));

    //choose the AI player here
    game.addPlayer(Player(new AISmart(), getRandomName()));
  }
  else if(gameType == 3) //AI Battle
  {
    //game.addObserver(new ObserverTerminalQuiet());
    game.addObserver(new ObserverTerminal());

    //choose the AI players here (AISmart, AIRandom, AICall, ...)
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
  }
  else if(gameType == 4) //AI heads-up
  {
    game.addObserver(new ObserverTerminalQuiet());

    //choose two AI players here
    game.addPlayer(Player(new AIRandom(), getRandomName()));
    game.addPlayer(Player(new AISmart(), getRandomName()));
  }
  else if(gameType == 5) //random game (human)
  {
    game.addPlayer(Player(new AIHuman(&terminal_host), "You"));

    size_t num = getRandom(1, 9);
    for(size_t i = 0; i < num; i++)
    {
      game.addPlayer(Player(getRandom() < 0.1 ? (AI*)(new AIRandom()) : (AI*)(new AISmart()), getRandomName()));
    }
  }

  game.doGame();

  // save weights after the session
  if (gameType == 6) {
    torch::save(net, "./logs/poker_model.pt");
  }

  return false;
}

int main()
{
    // 1. Initialize the global neural network and optimizer
    PokerNet global_net(23, 128);
    torch::optim::Adam optimizer(global_net->parameters(), 1e-4);
    CheckpointManager cp_manager("poker_model", 1000);

    // 2. Load existing weights if they exist
    std::string model_path = "./logs/poker_model.pt";
    std::ifstream f(model_path.c_str());
    if (f.good()) {
        try {
            torch::load(global_net, model_path);
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load model: " << e.msg() << std::endl;
        }
    }

    // 3. Create persistent dashboard
    ObserverDashboard dashboard(1000);

    // 4. Training Loop
    for(int epoch = 0; ; epoch++) {
        bool quit = doGame(global_net, optimizer, &dashboard, epoch);

        // every 10 sessions, run a formal evaluation
        if (epoch % 10 == 0) {
            cp_manager.run_evaluation(global_net, epoch, &dashboard);
            cp_manager.save_checkpoint(global_net, epoch);
        }

        if(quit) break;
    }
    return 0;
} // end of main
