#pragma once
#include "poker_net.h"
#include <string>
#include <vector>

class CheckpointManager {
private:
  std::string save_path;
  int eval_frequency; // how many hands between evals
  std::vector<float> win_rate_history;

public:
  CheckpointManager(std::string path, int freq = 1000);

  // runs a silent set of hands against a baseline to get a clean win rate
  void run_evaluation(PokerNet& net, int eval_hands = 100);

  // saves the weights and the current win rate history
  void save_checkpoint(PokerNet& net, int epoch);
}; // end of checkpointmanager
