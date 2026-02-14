#pragma once
#include "poker_net.h"
#include <string>
#include <vector>


class CheckpointManager {
public:
    CheckpointManager(std::string path, int freq = 100);
    void run_evaluation(PokerNet& net, int epoch);
    void save_checkpoint(PokerNet& net, int epoch);
private:
    std::string base_path;
    int frequency;
};
