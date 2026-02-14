#pragma once
#include "ai.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include "info.h"
#include "action.h"
#include "converter.h" // ActionNode and TensorConverter are here
#include "poker_net.h"

class AIRL: public AI {
public:
    AIRL(PokerNet& n, torch::optim::Optimizer& opt);
    
    // --- Overrides for the AI Interface ---
    Action doTurn(const Info& info) override;
    void onEvent(const Event& event) override;
    std::string getAIName() override;
    bool boastCards(const Info& info) override;
    bool wantsToLeave(const Info& info) override;

    // --- Helper Methods ---
    void reset_history();
    void add_to_history(int cmd, float amt, int pos);
    torch::Tensor history_to_tensor();

private:
    PokerNet& net;
    torch::optim::Optimizer& optimizer;
    
    std::shared_ptr<ActionNode> history_head;
    std::shared_ptr<ActionNode> history_tail;
    
    torch::Tensor h_state;
    torch::Tensor c_state;
    
    struct Experience {
        torch::Tensor log_prob;
        float stack;
    };
    std::vector<Experience> hand_experiences;
};
