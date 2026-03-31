#pragma once
#include <torch/torch.h>
#include <string>
#include "poker_net.h"

// Pass a non-empty checkpoint_path to resume from a saved .pt file.
void runRLTraining(const std::string& checkpoint_path = "");
