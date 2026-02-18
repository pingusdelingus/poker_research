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
    // 4 (Hole Cards) + 10 (Board) + 14 (Game State + Derived) = 28 Floats
    static constexpr const int INPUT_SIZE = 28;

    // Converts the game state (Info) into a [1, 28] Tensor for the NN
    static torch::Tensor infoToTensor(const Info& info);

    // Converts an existing Action (from AISmart) into a Target Vector (x, y)
    // Used for Imitation Learning (Training Phase)
    static torch::Tensor actionToTarget(const Action& action, const Info& info);

    // Converts the NN's Output Vector (x, y) back into a valid Poker Action
    // Used for Gameplay (Inference Phase)
    static Action vectorToAction(const Info& info, float x, float y);

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
