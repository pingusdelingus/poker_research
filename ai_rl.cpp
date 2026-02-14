#include "ai_rl.h"
#include <cmath>
#include "info.h"
#include "event.h"
#include "converter.h"

AIRL::AIRL(PokerNet& n, torch::optim::Optimizer& opt) 
  : net(n), optimizer(opt) 
{
  reset_history();
} // end of constructor

void AIRL::reset_history() 
{
  history_head = nullptr;
  history_tail = nullptr;
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
  if (len == 0) return torch::zeros({1, 1, 3});
  return torch::from_blob(data.data(), {len, 1, 3}, torch::kFloat).clone();
} // end of historytotensor

Action AIRL::doTurn(const Info& info) 
{
  torch::Tensor state = TensorConverter::infoToTensor(info);
  torch::Tensor hist = history_to_tensor();
  torch::Tensor opp = torch::zeros({1, 10}); 

  // we assume PokerNet has the forward_with_history method
  torch::Tensor out_vec = net->forward_with_history(state, hist, opp);

  float scale = 0.1f;
  auto noise = torch::randn_like(out_vec) * scale;
  auto sampled_vec = out_vec + noise;
  auto log_prob = -0.5 * torch::pow((sampled_vec - out_vec) / scale, 2).sum();

  // 3. store experience for reinforcement update later
  hand_experiences.push_back({log_prob, info.getStack()});

  // 4. convert vector to game action
  float x = sampled_vec[0][0].item<float>();
  float y = sampled_vec[0][1].item<float>();
  
  return TensorConverter::vectorToAction(info, x, y);
} // end of doturn

void AIRL::onEvent(const Event& event) 
{
  // update sequence history based on game events
  if (event.type == E_RAISE || event.type == E_CALL || event.type == E_CHECK || event.type == E_FOLD) {
    // find player index from name (simplified here)
    add_to_history((int)event.type, (float)event.chips / 100.0f, 0); 
  }

  if (event.type == E_NEW_DEAL) {
    reset_history();
  }

  // reinforcement learning update at the end of the hand
  if (event.type == E_POT_DIVISION) {
    for (auto& exp : hand_experiences) {
      // simple reward: chips at end vs chips at start
      // in a more advanced version, use advantage estimation
      float reward = (float)(event.chips); 
      
      auto loss = -exp.log_prob * reward;
      
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }
  }
} // end of onevent
