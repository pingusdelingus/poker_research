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
    float applyEpochReward(float epoch_reward);

    // New method for feature importance
    std::vector<std::pair<int, float>> getTopFeatures();

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
        std::deque<int> donk_history; // 1 = Donk, 0 = No Donk
        bool current_hand_vpip = false;
        bool current_hand_pfr = false;
        bool current_hand_donk = false;
        bool active_in_hand = false; // Did opponent play this hand?

        // Hand Ranges: 13x13 matrices
        std::vector<float> assumed_range; // 169 floats
        std::vector<float> seen_range;    // 169 floats
        int hand_bucket = 0;

        OpponentTracker() : assumed_range(169, 1.0f/169.0f), seen_range(169, 0.0f) {}
    } opp_tracker;

    // Last street aggressor (for donk bet detection)
    std::string last_street_aggressor;
    Round current_round_tracker = R_PRE_FLOP;

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
        torch::Tensor state;
        float stack;
    };
    std::vector<Experience> hand_experiences;

    // Per-epoch accumulation
    float accumulated_loss;
    int hand_count_in_epoch;
    
    // Per-hand reward tracking
    float buy_in;
    float hand_start_chips;
    float min_stack;
    float last_wager;
    float last_action_cost;
    float total_won;
    bool hand_complete;
    bool vpip_this_hand;
    std::string agent_name;

    // Variance reduction baselines
    float reward_baseline;
    float epoch_reward_baseline;

    // Feature Importance (Saliency)
    torch::Tensor last_static_state;
    torch::Tensor accumulated_saliency;
    int saliency_count;

    static constexpr float BASELINE_DECAY = 0.99f;
    static constexpr float MIN_SURVIVAL = 0.01f;
    static constexpr float MAX_GRAD_NORM = 1.0f;
    static constexpr float EPOCH_BASELINE_DECAY = 0.99f;
    static constexpr float EPOCH_REWARD_WEIGHT = 1.0f;
};
