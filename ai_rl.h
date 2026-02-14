#pragma once
#include <torch/torch.h>
#include <memory>
#include <vector>
#include "info.h"
#include "action.h"

// 1. the sequence node for your graph history
struct ActionNode {
    int command;
    float amount_norm;
    int player_pos;
    std::shared_ptr<ActionNode> next;

    ActionNode(int cmd, float amt, int pos) 
        : command(cmd), amount_norm(amt), player_pos(pos), next(nullptr) {}
}; // end of actionnode

// 2. the static conversion utility
class TensorConverter {
public:
    static constexpr int INPUT_SIZE = 23; 

    static torch::Tensor infoToTensor(const Info& info);
    static torch::Tensor actionToTarget(const Action& action, const Info& info);
    static Action vectorToAction(const Info& info, float x, float y);

private:
    static void encodeCard(const Card& c, std::vector<float>& features);
    static void encodeEmptyCard(std::vector<float>& features);
}; // end of tensorconverter
