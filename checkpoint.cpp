#include "checkpoint.h"
#include "game.h"
#include "ai_rl.h"
#include "ai_smart.h"
#include "host_terminal.h"
#include <fstream>

CheckpointManager::CheckpointManager(std::string path, int freq) 
  : save_path(path), eval_frequency(freq) {}

void CheckpointManager::run_evaluation(PokerNet& net, int eval_hands) {
  torch::NoGradGuard no_grad; // ensure no training happens during eval
  net->eval();

  HostTerminal silent_host; // use a quiet host for speed
  Game eval_game(&silent_host);
  
  // dummy optimizer as AIRL requires it, but it won't be used in eval mode
  torch::optim::Adam dummy_opt(net->parameters(), 0); 
  
  auto test_bot = std::make_shared<AIRL>(net, dummy_opt);
  auto baseline = std::make_shared<AISmart>(0.5);

  eval_game.addPlayer(Player(test_bot.get(), "Evaluated_Bot"));
  eval_game.addPlayer(Player(baseline.get(), "Baseline_Smart"));

  Rules rules;
  rules.fixedNumberOfDeals = eval_hands;
  rules.buyIn = 1000;
  eval_game.setRules(rules);

  std::cout << "\n[EVALUATION] Running " << eval_hands << " hands vs AISmart..." << std::endl;
  eval_game.doGame();
  
  // logic to extract win rate from game stats would go here
  // for now, we simply save the state
  net->train(); 
} // end of run_evaluation

void CheckpointManager::save_checkpoint(PokerNet& net, int epoch) {
  std::string filename = save_path + "_epoch_" + std::to_string(epoch) + ".pt";
  torch::save(net, filename);
  
  // also log to a csv for graphing
  std::ofstream log("training_log.csv", std::ios::app);
  log << epoch << "," << "win_rate_placeholder" << "\n";
  log.close();
} // end of save_checkpoint
