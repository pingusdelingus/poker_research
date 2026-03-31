#pragma once
#include "poker_net.h"
#include <string>
#include <vector>

class ObserverDashboard;

class CheckpointManager {
public:
    CheckpointManager(std::string path, int freq = 100);
    void run_evaluation(PokerNet& net, int epoch, ObserverDashboard* dashboard = nullptr);
    void save_checkpoint(PokerNet& net, int epoch);

    // Returns the path to the most recent checkpoint file, or "" if none exists.
    std::string find_latest_checkpoint() const;

    // Loads weights from a .pt file into net. Returns true on success.
    bool load_checkpoint(PokerNet& net, const std::string& path) const;

private:
    std::string base_path;
    int frequency;
};
