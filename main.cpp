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
#include "info.h"
#include "io_terminal.h"
#include "observer.h"
#include "observer_terminal.h"
#include "observer_terminal_quiet.h"
#include "observer_log.h"
#include "pokermath.h"
#include "random.h"
#include "table.h"
#include "tools_terminal.h"
#include "unittest.h"
##include <torch/torch.h>
#include "poker_net.h"
#include "ai_rl.h" 

// returns whether user wants to quit

bool doGame(PokerNet& net, torch::optim::Optimizer& optimizer)
{
  std::cout << "Welcome to OOPoker RL Trainer\n" << std::endl;

  std::cout << "Choose Game Type\n\
1: human + AI's\n\
2: human + AI heads-up\n\
3: AI battle\n\
4: AI battle heads-up\n\
6: RL Self-Play Training (NEW)\n\
q: quit" << std::endl;
  
  char c = getChar();
  int gameType = (c == '6') ? 6 : (c - '0');
  if(c == 'q') return true;

  Rules rules;
  rules.buyIn = 1000;
  rules.bigBlind = 10;
  rules.allowRebuy = (gameType == 6); // enable rebuys for training stability
  rules.fixedNumberOfDeals = (gameType == 6) ? 1000 : 100;

  HostTerminal host;
  Game game(&host);
  game.setRules(rules);

  if(gameType == 6) // RL Self-Play Training
  {
    std::cout << "Starting Self-Play Session..." << std::endl;
    // use standard terminal observer to see progress
    game.addObserver(new ObserverTerminalQuiet());

    // both players use the SAME network (shared_ptr) to learn against themselves
    auto agent1 = std::make_shared<AIRL>(net, optimizer);
    auto agent2 = std::make_shared<AIRL>(net, optimizer);

    game.addPlayer(Player(agent1.get(), "RL_Agent_A"));
    game.addPlayer(Player(agent2.get(), "RL_Agent_B"));
  }
  /* ... existing gameType 1-5 logic ... */
  else if(gameType == 3) // example of using the bot in a normal battle
  {
    game.addObserver(new ObserverTerminal());
    game.addPlayer(Player(new AIRL(net, optimizer), "Trained_Bot"));
    for(int i = 0; i < 5; ++i) game.addPlayer(Player(new AISmart(), getRandomName()));
  }

  game.doGame();

  // save weights after the session
  if (gameType == 6) {
    torch::save(net, "poker_model.pt");
    std::cout << "Model saved to poker_model.pt" << std::endl;
  }

  return false;
}

int main()
{
PokerNet global_net(23, 128);
  torch::optim::Adam optimizer(global_net->parameters(), 1e-4);
  CheckpointManager cp_manager("poker_model", 1000); // eval every 1k hands

  for(int epoch = 0; ; epoch++) {
    bool quit = doGame(global_net, optimizer);
    
    // every 10 sessions, run a formal evaluation
    if (epoch % 10 == 0) {
      cp_manager.run_evaluation(global_net);
      cp_manager.save_checkpoint(global_net, epoch);
    }

    if(quit) break;
  }
  return 0;

}// end of main


