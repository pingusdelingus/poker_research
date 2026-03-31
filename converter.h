#pragma once
#include <memory>
#include <torch/torch.h>
#include "info.h"
#include "action.h"
#include "card.h"


struct ActionNode {
    int command;
    float amount_norm;
    int player_pos;
    std::shared_ptr<ActionNode> next;

    ActionNode(int cmd, float amt, int pos) 
        : command(cmd), amount_norm(amt), player_pos(pos), next(nullptr) {}
}; // end of actionnode


class TensorConverter {
public:
    // 0-23: Static State (24 floats, M-ratio removed)
    // 24-45: Opponent Stats & Metadata (22 floats)
    static constexpr const int INPUT_SIZE = 46;
    static constexpr const int STATIC_SIZE = 24;
    static constexpr const int OPPONENT_SIZE = 22;

    // Converts the game state (Info) + Opponent Stats into a [1, INPUT_SIZE] Tensor
    static torch::Tensor infoToTensor(const Info& info, const std::vector<float>& opponent_stats);

    // Converts the NN's sampled action index and scalar sizing back into a valid Poker Action
    static Action logitsToAction(const Info& info, int action_idx, float sizing);

private:
    // Helper to normalize card ranks (2-14 -> 0-1) and suits (0-3 -> 0-1)
    static void encodeCard(const Card& c, std::vector<float>& features);
    static void encodeEmptyCard(std::vector<float>& features);
};

class GraphConverter {
public:
    // converts the shared_ptr list into a sequence tensor for the RNN
    static torch::Tensor historyToTensor(std::shared_ptr<ActionNode> head) {
        std::vector<float> data;
        int len = 0;
        auto curr = head;
        while (curr) {
            data.push_back((float)curr->command / 3.0f);
            data.push_back(curr->amount_norm);
            data.push_back((float)curr->player_pos / 9.0f);
            len++;
            curr = curr->next;
        }
        if (len == 0) return torch::zeros({1, 1, 3});
        return torch::from_blob(data.data(), {len, 1, 3}, torch::kFloat).clone();
    } // end of historytotensor
};
