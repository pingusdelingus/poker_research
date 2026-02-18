#pragma once
#include "ai.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "info.h"
#include "action.h"
#include "converter.h" // ActionNode and TensorConverter are here
#include "poker_net.h"

class AIRL: public AI {
public:
    AIRL(PokerNet& n, torch::optim::Optimizer& opt, float buy_in = 1000.0f);

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

private:
    PokerNet& net;
    torch::optim::Optimizer& optimizer;

    // LSTM history
    std::shared_ptr<ActionNode> history_head;
    std::shared_ptr<ActionNode> history_tail;
    torch::Tensor h_state;
    torch::Tensor c_state;

    // Per-decision experience buffer (cleared each hand)
    struct Experience {
        torch::Tensor log_prob;
        float stack;
    };
    std::vector<Experience> hand_experiences;

    // Per-epoch experience buffer (accumulates across all hands)
    struct EpochExperience {
        torch::Tensor log_prob; // summed log_prob for the hand
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
