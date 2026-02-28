#pragma once
#include "ai.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <deque>
#include <algorithm>
#include "info.h"
#include "action.h"
#include "converter.h" // ActionNode and TensorConverter are here
#include "poker_net.h"

class AIRL: public AI {
public:
    AIRL(PokerNet& n, torch::optim::Optimizer& opt, float buy_in = 1000.0f, float noise = 0.5f, float entropy_coeff = 0.001f);

    // --- Overrides for the AI Interface ---
    Action doTurn(const Info& info) override;
    void onEvent(const Event& event) override;
    std::string getAIName() override;
    bool boastCards(const Info& info) override;
    bool wantsToLeave(const Info& info) override;

    // --- Epoch-level reward ---
    void applyEpochReward(float epoch_reward);

    // --- Helper Methods ---
    void reset_history();
    void add_to_history(int cmd, float amt, int pos);
    torch::Tensor history_to_tensor();
    
    // Opponent Tracking
    std::vector<float> get_opponent_features();
    void update_opponent_stats(const Event& event);

private:
    PokerNet& net;
    torch::optim::Optimizer& optimizer;

    // Opponent Stats Tracking
    struct OpponentTracker {
        std::string name;
        std::deque<int> vpip_history; // 1 = VPIP, 0 = No VPIP
        std::deque<int> pfr_history;  // 1 = PFR, 0 = No PFR
        bool current_hand_vpip = false;
        bool current_hand_pfr = false;
        bool active_in_hand = false; // Did opponent play this hand?
    } opp_tracker;

    // LSTM history
    std::shared_ptr<ActionNode> history_head;
    std::shared_ptr<ActionNode> history_tail;
    torch::Tensor h_state;
    torch::Tensor c_state;

    // Exploration noise and entropy coefficient (both decay over training)
    float noise_scale;
    float entropy_coeff;

    // Per-decision experience buffer (cleared each hand)
    struct Experience {
        torch::Tensor log_prob;
        torch::Tensor entropy;
        float stack;
    };
    std::vector<Experience> hand_experiences;

    // Per-epoch experience buffer (accumulates across all hands)
    struct EpochExperience {
        torch::Tensor log_prob;    // summed log_prob for the hand
        torch::Tensor entropy_sum; // summed entropy for the hand
    };
    std::vector<EpochExperience> epoch_experiences;

    // Per-hand reward tracking
    float buy_in;
    float hand_start_chips;
    float min_stack;
    float last_wager;
    float last_action_cost;
    float total_won;
    bool hand_complete;
    std::string agent_name;

    // Variance reduction baselines
    float reward_baseline;
    float epoch_reward_baseline;

    static constexpr float BASELINE_DECAY = 0.99f;
    static constexpr float MIN_SURVIVAL = 0.01f;
    static constexpr float MAX_GRAD_NORM = 1.0f;
    static constexpr float EPOCH_BASELINE_DECAY = 0.99f;
    static constexpr float EPOCH_REWARD_WEIGHT = 1.0f;
};
