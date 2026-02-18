#include "ai_rl.h"
#include "converter.h"
#include "info.h"
#include "event.h"
#include <torch/torch.h>
#include <cmath>

AIRL::AIRL(PokerNet& n, torch::optim::Optimizer& opt, float buy_in_amount)
  : net(n), optimizer(opt),
    buy_in(buy_in_amount),
    hand_start_chips(0.0f), min_stack(0.0f),
    last_wager(0.0f), last_action_cost(0.0f), total_won(0.0f),
    hand_complete(false),
    reward_baseline(0.0f),
    epoch_reward_baseline(0.0f)
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
  // Track stacks for survival-weighted reward
  if (hand_experiences.empty()) {
    // First decision of the hand: record starting chips (stack + wager includes blinds already posted)
    hand_start_chips = static_cast<float>(info.getStack() + info.getWager());
    min_stack = static_cast<float>(info.getStack());
    agent_name = info.getYou().getName();
  } else {
    min_stack = std::min(min_stack, static_cast<float>(info.getStack()));
  }
  last_wager = static_cast<float>(info.getWager());

  // 1. forward pass through the graph rnn
  torch::Tensor state = TensorConverter::infoToTensor(info);
  torch::Tensor hist = history_to_tensor();
  torch::Tensor opp = torch::zeros({1, 10});
  torch::Tensor out_vec = net->forward_with_history(state, hist, opp);

  // 2. stochastic exploration (reparameterization)
  float noise_scale = 0.1f;
  auto sampled_vec = out_vec + torch::randn_like(out_vec) * noise_scale;

  // 3. calculate log_prob for the policy gradient
  auto log_prob = -0.5 * torch::pow((sampled_vec - out_vec) / noise_scale, 2).sum();

  hand_experiences.push_back({log_prob, static_cast<float>(info.getStack())});

  Action action = TensorConverter::vectorToAction(info, sampled_vec[0][0].item<float>(), sampled_vec[0][1].item<float>());

  // Track the cost of this action for final wager calculation
  if (action.command == A_RAISE) {
    last_action_cost = static_cast<float>(action.amount);
  } else if (action.command == A_CALL) {
    last_action_cost = static_cast<float>(info.getCallAmount());
  } else {
    last_action_cost = 0.0f;
  }

  return action;
} // end of doturn


void AIRL::onEvent(const Event& event) {
    // Track betting history for LSTM input
    if (event.type == E_RAISE || event.type == E_CALL || event.type == E_CHECK || event.type == E_FOLD) {
        add_to_history((int)event.type, (float)event.chips / 100.0f, 0);
    }

    // Accumulate winnings for this agent
    if (event.type == E_WIN && event.player == agent_name) {
        total_won += static_cast<float>(event.chips);
    }

    // Mark hand as complete (E_WIN events follow in the same dispatch batch)
    if (event.type == E_POT_DIVISION) {
        hand_complete = true;
    }

    // At the start of a new hand: compute reward for the completed hand and update
    if (event.type == E_NEW_DEAL) {
        // If the previous hand had decisions, compute reward and do REINFORCE update
        // Skip if gradients are disabled (e.g. during evaluation with NoGradGuard)
        if (hand_complete && !hand_experiences.empty() &&
            hand_experiences[0].log_prob.requires_grad()) {
            // Total chips the agent put into the pot this hand
            float total_invested = last_wager + last_action_cost;

            // Net chip gain normalized by buy-in
            float chip_gain = (total_won - total_invested) / buy_in;

            // Survival factor: penalizes hands where stack dipped low (all-ins)
            float survival_factor = std::max(MIN_SURVIVAL, min_stack / buy_in);

            // Final survival-weighted reward
            float reward = chip_gain * survival_factor;

            // Advantage over baseline for variance reduction
            float advantage = reward - reward_baseline;
            reward_baseline = BASELINE_DECAY * reward_baseline + (1.0f - BASELINE_DECAY) * reward;

            // Accumulate hand's summed log_prob for epoch-level reward
            torch::Tensor hand_log_prob_sum = torch::zeros({1});
            for (const auto& exp : hand_experiences) {
                hand_log_prob_sum = hand_log_prob_sum + exp.log_prob;
            }
            epoch_experiences.push_back({hand_log_prob_sum});

            // REINFORCE: loss = -sum(log_prob_i * advantage)
            torch::Tensor loss = torch::zeros({1});
            for (const auto& exp : hand_experiences) {
                loss = loss - exp.log_prob * advantage;
            }

            optimizer.zero_grad();
            loss.backward({}, /*retain_graph=*/true);
            torch::nn::utils::clip_grad_norm_(net->parameters(), MAX_GRAD_NORM);
            optimizer.step();
        }

        // Reset for new hand
        reset_history();
        total_won = 0.0f;
        hand_complete = false;
        last_action_cost = 0.0f;
        hand_start_chips = 0.0f;
        min_stack = 0.0f;
        last_wager = 0.0f;
    }
}

void AIRL::applyEpochReward(float epoch_reward)
{
    // Flush the last hand's log_probs (no E_NEW_DEAL follows the final hand)
    if (!hand_experiences.empty() && hand_experiences[0].log_prob.requires_grad()) {
        torch::Tensor hand_log_prob_sum = torch::zeros({1});
        for (const auto& exp : hand_experiences) {
            hand_log_prob_sum = hand_log_prob_sum + exp.log_prob;
        }
        epoch_experiences.push_back({hand_log_prob_sum});
    }

    if (epoch_experiences.empty()) return;

    // Skip if gradients are disabled (evaluation mode)
    if (!epoch_experiences[0].log_prob.requires_grad()) {
        epoch_experiences.clear();
        return;
    }

    // Epoch advantage with separate baseline
    float epoch_advantage = epoch_reward - epoch_reward_baseline;
    epoch_reward_baseline = EPOCH_BASELINE_DECAY * epoch_reward_baseline
                          + (1.0f - EPOCH_BASELINE_DECAY) * epoch_reward;

    // REINFORCE over the entire epoch, normalized by num_hands
    float num_hands = static_cast<float>(epoch_experiences.size());
    torch::Tensor loss = torch::zeros({1});
    for (const auto& exp : epoch_experiences) {
        loss = loss - exp.log_prob * epoch_advantage;
    }
    loss = loss * (EPOCH_REWARD_WEIGHT / num_hands);

    optimizer.zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(net->parameters(), MAX_GRAD_NORM);
    optimizer.step();

    epoch_experiences.clear();
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
