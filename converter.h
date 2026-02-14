#pragma once
#include <memory>
#include <torch/torch.h>
#include "info.h"
#include "action.h"

struct ActionNode {
    int command;
    float amount_norm;
    int player_pos;
    std::shared_ptr<ActionNode> next;

    ActionNode(int cmd, float amt, int pos) 
        : command(cmd), amount_norm(amt), player_pos(pos), next(nullptr) {}
}; // end of actionnode

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
