#include "ai_rl.h"
#include "converter.h" 
#include "info.h"
#include "event.h"
#include <torch/torch.h>

AIRL::AIRL(PokerNet& n, torch::optim::Optimizer& opt) 
  : net(n), optimizer(opt) 
{
  reset_history();
} // end of constructor

void AIRL::reset_history() 
{
  history_head = nullptr;
  history_tail = nullptr;
  // initialize hidden states for the lstm
  h_state = torch::zeros({1, 1, 128});
  c_state = torch::zeros({1, 1, 128});
  hand_experiences.clear();
} // end of reset_history

void AIRL::add_to_history(int cmd, float amt, int pos) 
{
  auto new_node = std::make_shared<ActionNode>(cmd, amt, pos);
  if (!history_head) {
    history_head = new_node;
    history_tail = new_node;
  } else {
    history_tail->next = new_node;
    history_tail = new_node;
  }
} // end of add_to_history


torch::Tensor AIRL::history_to_tensor() 
{
  std::vector<float> data;
  int len = 0;
  auto curr = history_head;
  while (curr) {
    data.push_back((float)curr->command / 3.0f);
    data.push_back(curr->amount_norm);
    data.push_back((float)curr->player_pos / 9.0f);
    len++;
    curr = curr->next;
  }
  
  // if no history, provide a "zero" action node to keep dimensions consistent
  if (len == 0) return torch::zeros({1, 1, 3});
  
  // create tensor [len, 1, 3]
  return torch::from_blob(data.data(), {len, 1, 3}, torch::kFloat).clone();
}

Action AIRL::doTurn(const Info& info) 
{
  torch::Tensor state = TensorConverter::infoToTensor(info);
  torch::Tensor hist = history_to_tensor();
  torch::Tensor opp = torch::zeros({1, 10});

  // 1. forward pass through the graph rnn
  torch::Tensor out_vec = net->forward_with_history(state, hist, opp);
  
  // 2. stochastic exploration (reparameterization)
  float noise_scale = 0.1f; 
  auto sampled_vec = out_vec + torch::randn_like(out_vec) * noise_scale;

  // 3. calculate log_prob for the policy gradient
  auto log_prob = -0.5 * torch::pow((sampled_vec - out_vec) / noise_scale, 2).sum();

  hand_experiences.push_back({log_prob, static_cast<float> (info.getStack() )});

  return TensorConverter::vectorToAction(info, sampled_vec[0][0].item<float>(), sampled_vec[0][1].item<float>());
} // end of doturn
//


void AIRL::onEvent(const Event& event) {
    // track history
    if (event.type == E_RAISE || event.type == E_CALL || event.type == E_CHECK || event.type == E_FOLD) {
        add_to_history((int)event.type, (float)event.chips / 100.0f, 0); 
    }
    
    if (event.type == E_NEW_DEAL) {
        reset_history();
    }

    // placeholder for rl update logic
    if (event.type == E_POT_DIVISION) {
        // update logic here
    }
}

std::string AIRL::getAIName() {
    return "GraphRL_Bot";
}

bool AIRL::boastCards(const Info& info) {
    return false;
}

bool AIRL::wantsToLeave(const Info& info) {
    return false;
}
