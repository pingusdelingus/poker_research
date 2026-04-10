#include "ai_rl.h"
#include "converter.h"
#include "info.h"
#include "event.h"
#include "pokermath.h"
#include <torch/torch.h>
#include <cmath>

AIRL::AIRL(PokerNet& n, torch::optim::Optimizer& opt, float buy_in_amount, float noise, float ent_coeff)
  : net(n), optimizer(opt),
    buy_in(buy_in_amount), noise_scale(noise), entropy_coeff(ent_coeff),
    hand_start_chips(0.0f), min_stack(0.0f),
    last_wager(0.0f), last_action_cost(0.0f), total_won(0.0f),
    hand_complete(false), vpip_this_hand(false),
    reward_baseline(0.0f),
    epoch_reward_baseline(0.0f),
    current_round_tracker(R_PRE_FLOP),
    accumulated_loss(0.0f),
    hand_count_in_epoch(0),
    current_big_blind(10),
    last_range_type_raw(1.0f)
{
  reset_history();
} // end of constructor

void AIRL::reset_history()
{
  history_head = nullptr;
  history_tail = nullptr;
  history_length = 0;
  history_position = 0;
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
  history_length++;
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

// Returns only the actions from index `start` onward as a tensor.
// Used for incremental LSTM feeding: only new actions since the last doTurn.
torch::Tensor AIRL::history_to_tensor_from(int start)
{
  std::vector<float> data;
  int idx = 0;
  auto curr = history_head;
  while (curr) {
    if (idx >= start) {
      data.push_back((float)curr->command / 3.0f);
      data.push_back(curr->amount_norm);
      data.push_back((float)curr->player_pos / 9.0f);
    }
    idx++;
    curr = curr->next;
  }
  int len = (int)data.size() / 3;
  if (len == 0) return torch::zeros({1, 1, 3});
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
    
    // Identify opponent name (the other player in head-to-head)
    opp_tracker.name = "";
    for (int i=0; i<info.getNumPlayers(); ++i) {
        if (i != info.yourIndex) {
            opp_tracker.name = info.players[i].name;
            break;
        }
    }
  } else {
    min_stack = std::min(min_stack, static_cast<float>(info.getStack()));
  }
  last_wager = static_cast<float>(info.getWager());

  // 1. Get current opponent features
  std::vector<float> opp_feats = get_opponent_features();

  // Capture range type for EMA update at end of hand (Info only available here).
  // RANGE_UNKNOWN=2, LINEAR=1, POLAR=0 — normalized by /2.
  {
      int oppIdx = (info.yourIndex == 0) ? 1 : 0;
      last_range_type_raw = (float)getOpponentRangeType(info, oppIdx) / 2.0f;
  }
  
  // 2. forward pass through the graph rnn
  torch::Tensor state_full = TensorConverter::infoToTensor(info, opp_feats);
  // track saliency for all 38 features (23 static + 15 opponent)
  torch::Tensor state = state_full.clone().set_requires_grad(true); 
  
  torch::Tensor static_part = state.slice(1, 0, TensorConverter::STATIC_SIZE);
  torch::Tensor opp_part = state.slice(1, TensorConverter::STATIC_SIZE, TensorConverter::INPUT_SIZE);
  
  // Feed only new actions (since last doTurn) into the LSTM, carrying h/c state forward.
  // This is O(delta) per decision instead of O(n^2) over the full history each time.
  torch::Tensor delta = history_to_tensor_from(history_position);
  auto fwd = net->forward_with_state(static_part, delta, opp_part, h_state, c_state);
  torch::Tensor out_vec = std::get<0>(fwd);
  h_state = std::get<1>(fwd);
  c_state = std::get<2>(fwd);
  history_position = history_length;

  // Guard: if weights are corrupted (NaN/Inf), skip this decision safely
  if (torch::any(torch::isnan(out_vec)).item<bool>() ||
      torch::any(torch::isinf(out_vec)).item<bool>()) {
    // Can't learn from this step; return a safe action without touching gradients
    return info.getCheckFoldAction();
  }

  // 3. stochastic exploration (reparameterization / discrete sampling)
  torch::Tensor logits = out_vec.slice(1, 0, 3);
  torch::Tensor sizing = out_vec.slice(1, 3, 4);

  // Clamp logits to prevent softmax overflow/underflow (e.g. after a large raise squeezes one class)
  logits = torch::clamp(logits, -10.0f, 10.0f);

  // Discrete action probability
  torch::Tensor probs = torch::softmax(logits, 1);
  torch::Tensor log_probs = torch::log_softmax(logits, 1);

  // Safety clamp on probs before multinomial - avoids 0-probability elements causing NaN
  probs = torch::clamp(probs, 1e-6f, 1.0f);
  // Re-renormalize so they sum to 1 after clamping
  probs = probs / probs.sum(1, true);

  // Sample action
  torch::Tensor action_idx;
  if (noise_scale > 1e-6f) { // training
      action_idx = torch::multinomial(probs, 1);
  } else {
      action_idx = torch::argmax(probs, 1, true);
  }

  // Explore sizing
  torch::Tensor sampled_sizing;
  if (noise_scale > 1e-6f) {
      torch::Tensor noise = torch::randn_like(sizing) * noise_scale;
      sampled_sizing = sizing + noise;
  } else {
      sampled_sizing = sizing;
  }

  // Selected action log_prob 
  torch::Tensor chosen_log_prob = log_probs[0][action_idx[0].item<int64_t>()].unsqueeze(0);
  if (action_idx[0].item<int64_t>() == 2) { // RAISE
      torch::Tensor p_sz = -0.5f * torch::pow((sampled_sizing.detach() - sizing) / noise_scale, 2).sum();
      chosen_log_prob = chosen_log_prob + p_sz;
  }

  // 4. entropy bonus: - sum( p * log(p) ) for discrete
  // Using probs + 1e-8 avoids 0 * -inf producing NaN
  auto entropy = -(probs * torch::log(probs + 1e-8f)).sum();

  hand_experiences.push_back({chosen_log_prob, entropy, state, static_cast<float>(info.getStack())});

  Action action = TensorConverter::logitsToAction(info, action_idx[0].item<int64_t>(), sampled_sizing[0][0].item<float>());

  // Track VPIP (Voluntarily Put Money In Pot)
  if (action.command == A_RAISE || (action.command == A_CALL && info.getCallAmount() > 0)) {
      vpip_this_hand = true;
  }

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
    if (event.type == E_NEW_DEAL) {
        if (event.bigBlind > 0) current_big_blind = event.bigBlind;
        current_board.clear();
        opp_tracker.hb_probs_dirty = true; // reset cache for new hand
    } else if (event.type == E_FLOP) {
        current_board.push_back(event.card1);
        current_board.push_back(event.card2);
        current_board.push_back(event.card3);
        opp_tracker.hb_probs_dirty = true; // new board = stale cache
    } else if (event.type == E_TURN) {
        current_board.push_back(event.card4);
        opp_tracker.hb_probs_dirty = true;
    } else if (event.type == E_RIVER) {
        current_board.push_back(event.card5);
        opp_tracker.hb_probs_dirty = true;
    }

    // Update opponent stats from events
    update_opponent_stats(event);

    // Track round changes for donk bet detection
    if (event.type == E_FLOP || event.type == E_TURN || event.type == E_RIVER) {
        current_round_tracker = (Round)(current_round_tracker + 1);
    }

    // Track betting history for LSTM input
    if (event.type == E_RAISE || event.type == E_CALL || event.type == E_CHECK || event.type == E_FOLD) {
        add_to_history((int)event.type, (float)event.chips / 100.0f, 0);
        if (event.type == E_RAISE) {
            last_street_aggressor = event.player;
        }
        // hb_probs_dirty is set inside update_opponent_stats when the range changes
    }

    // Accumulate winnings for this agent
    if (event.type == E_WIN && event.player == agent_name) {
        total_won += static_cast<float>(event.chips);
    }

    // Mark hand as complete and lock in this hand's range type for the cross-hand EMA.
    if (event.type == E_POT_DIVISION) {
        hand_complete = true;
        opp_tracker.range_type_ema = 0.9f * opp_tracker.range_type_ema
                                   + 0.1f * last_range_type_raw;
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
            // Uses total_won (what we won from pot) vs total_invested (what we put in)
            float chip_gain = (total_won - total_invested) / buy_in;

            // Reward is the raw chip gain signal — no survival multiplier.
            // The chip gain already penalizes losing chips; a separate survival factor
            // suppressed legitimate aggression (e.g. profitable all-ins) by design.
            float reward = chip_gain;

            // Advantage over baseline for variance reduction
            float advantage = reward - reward_baseline;
            reward_baseline = BASELINE_DECAY * reward_baseline + (1.0f - BASELINE_DECAY) * reward;

            // REINFORCE: loss = -sum(log_prob_i * advantage) - entropy_coeff * sum(entropy_i)
            torch::Tensor loss = torch::zeros({1});
            for (const auto& exp : hand_experiences) {
                loss = loss - exp.log_prob * advantage - entropy_coeff * exp.entropy;
            }

            // Guard: skip backward entirely if loss is non-finite
            // (can happen when weights have exploded, avoids propagating NaN into grads)
            float loss_val = loss.item<float>();
            if (!std::isfinite(loss_val)) {
                // Don't backward, don't accumulate — just drop this hand
            } else {
                loss.backward();

                // Per-hand gradient clip: prevents a single bad hand from
                // poisoning the epoch's accumulated gradient buffer
                torch::nn::utils::clip_grad_norm_(net->parameters(), MAX_GRAD_NORM);

                // Saliency accumulation — only add if gradient is fully finite
                for (const auto& exp : hand_experiences) {
                    if (exp.state.grad().defined()) {
                        auto grad_abs = exp.state.grad().abs().sum(0).detach();
                        if (!torch::any(torch::isnan(grad_abs)).item<bool>() &&
                            !torch::any(torch::isinf(grad_abs)).item<bool>()) {
                            if (!accumulated_saliency.defined()) {
                                accumulated_saliency = torch::zeros({TensorConverter::INPUT_SIZE});
                                saliency_count = 0;
                            }
                            accumulated_saliency += grad_abs;
                            saliency_count++;
                        }
                    }
                }

                accumulated_loss += loss_val;
                hand_count_in_epoch++;
            }
        }

        // Reset for new hand
        reset_history();
        total_won = 0.0f;
        hand_complete = false;
        vpip_this_hand = false;
        last_action_cost = 0.0f;
        hand_start_chips = 0.0f;
        min_stack = 0.0f;
        last_wager = 0.0f;
        current_round_tracker = R_PRE_FLOP;
        last_street_aggressor = "";
    }
}

void AIRL::update_opponent_stats(const Event& event) {
    if (event.type == E_NEW_DEAL) {
        std::fill(opp_tracker.assumed_range.begin(), opp_tracker.assumed_range.end(), 1.0f/1326.0f);
        opp_tracker.hb_probs_dirty = true; // range was reset, invalidate cache
        if (opp_tracker.active_in_hand) {
            opp_tracker.vpip_history.push_front(opp_tracker.current_hand_vpip ? 1 : 0);
            opp_tracker.pfr_history.push_front(opp_tracker.current_hand_pfr ? 1 : 0);
            opp_tracker.donk_history.push_front(opp_tracker.current_hand_donk ? 1 : 0);
            if (opp_tracker.vpip_history.size() > 100) opp_tracker.vpip_history.pop_back();
            if (opp_tracker.pfr_history.size() > 100) opp_tracker.pfr_history.pop_back();
            if (opp_tracker.donk_history.size() > 100) opp_tracker.donk_history.pop_back();
        }
        opp_tracker.current_hand_vpip = false;
        opp_tracker.current_hand_pfr = false;
        opp_tracker.current_hand_donk = false;
        opp_tracker.active_in_hand = false;
    }

    if (event.player == opp_tracker.name) {
        opp_tracker.active_in_hand = true;

        if (event.type == E_RAISE) {
            // Update sizing tell EMA: event.chips is the raise amount above call, in chips.
            // We normalise by the big blind so the signal is stake-independent.
            float raise_bb = (current_big_blind > 0)
                             ? (float)event.chips / (float)current_big_blind
                             : 1.0f;
            opp_tracker.avg_raise_bb = 0.9f * opp_tracker.avg_raise_bb + 0.1f * raise_bb;

            updateOpponentRange(opp_tracker.assumed_range, current_board, 2);
            opp_tracker.hb_probs_dirty = true;
        } else if (event.type == E_CALL) {
            updateOpponentRange(opp_tracker.assumed_range, current_board, 1);
            opp_tracker.hb_probs_dirty = true;
        } else if (event.type == E_CHECK) {
            updateOpponentRange(opp_tracker.assumed_range, current_board, 0);
            opp_tracker.hb_probs_dirty = true;
        }

        if (current_round_tracker == R_PRE_FLOP) {
            if (event.type == E_RAISE) {
                opp_tracker.current_hand_vpip = true;
                opp_tracker.current_hand_pfr = true;
            } else if (event.type == E_CALL) {
                opp_tracker.current_hand_vpip = true;
            }
        } else {
            // Post-flop donk bet detection: leading into the previous street aggressor
            if (event.type == E_RAISE && !last_street_aggressor.empty() && last_street_aggressor != opp_tracker.name) {
                // If the opponent is the first to act aggressively this street, it's a donk bet
                opp_tracker.current_hand_donk = true;
            }
        }
    }

    if ((event.type == E_PLAYER_SHOWDOWN || event.type == E_BOAST) && event.player == opp_tracker.name) {
        // Record seen hand in 13x13 matrix
        if (event.card1.isValid() && event.card2.isValid()) {
            int r1 = event.card1.value - 2;
            int r2 = event.card2.value - 2;
            bool suited = (event.card1.suit == event.card2.suit);
            int idx = suited ? (std::max(r1, r2) * 13 + std::min(r1, r2)) : (std::min(r1, r2) * 13 + std::max(r1, r2));
            opp_tracker.seen_range[idx] += 1.0f;
        }
    }
}
std::vector<float> AIRL::get_opponent_features() {
    std::vector<float> feats;
    feats.reserve(TensorConverter::OPPONENT_SIZE);

    // 0-6: HandBand Probs — computed once per street change, cached for all decisions on that street.
    // Previously this ran 1326 full hand evaluations on EVERY doTurn call (O(1326) per decision),
    // now it only recomputes when the board or range changes.
    if (opp_tracker.hb_probs_dirty) {
        opp_tracker.cached_hb_probs = getOpponentHandBandProbabilities(current_board, opp_tracker.assumed_range);
        opp_tracker.hb_probs_dirty = false;
    }
    for (float p : opp_tracker.cached_hb_probs) {
        feats.push_back(std::min(1.0f, p * 2.0f));
    }

    // 7: Hand VPIP (Live), 8: Hand PFR (Live), 9: History Is Empty
    feats.push_back(opp_tracker.current_hand_vpip ? 1.0f : 0.0f);
    feats.push_back(opp_tracker.current_hand_pfr ? 1.0f : 0.0f);
    feats.push_back((history_head == nullptr) ? 1.0f : 0.0f);

    auto calc_rate = [](const std::deque<int>& history, size_t window) {
        if (history.empty()) return 0.5f;
        int count = 0;
        size_t n = std::min(history.size(), window);
        for (size_t i = 0; i < n; ++i) count += history[i];
        return (float)count / (float)n;
    };

    // 10-13: VPIP 10/30/50/100
    feats.push_back(calc_rate(opp_tracker.vpip_history, 10));
    feats.push_back(calc_rate(opp_tracker.vpip_history, 30));
    feats.push_back(calc_rate(opp_tracker.vpip_history, 50));
    feats.push_back(calc_rate(opp_tracker.vpip_history, 100));

    // 14-17: PFR 10/30/50/100
    feats.push_back(calc_rate(opp_tracker.pfr_history, 10));
    feats.push_back(calc_rate(opp_tracker.pfr_history, 30));
    feats.push_back(calc_rate(opp_tracker.pfr_history, 50));
    feats.push_back(calc_rate(opp_tracker.pfr_history, 100));

    // 18-21: Donk 10/30/50/100
    feats.push_back(calc_rate(opp_tracker.donk_history, 10));
    feats.push_back(calc_rate(opp_tracker.donk_history, 30));
    feats.push_back(calc_rate(opp_tracker.donk_history, 50));
    feats.push_back(calc_rate(opp_tracker.donk_history, 100));

    // 22: Opponent average raise size in BB (EMA, capped at 10 BB, normalized 0-1).
    // A high value flags opponents who habitually over-bet strong hands —
    // the model can exploit this by folding more to large bets and calling down light on small ones.
    feats.push_back(std::min(opp_tracker.avg_raise_bb / 10.0f, 1.0f));

    // 23: Cross-hand range type EMA (0=polar, 0.5=linear, 1=unknown).
    // Per-decision range type (static feature 23) is noisy early in a hand;
    // this smoothed version provides a stable prior of the opponent's overall tendencies.
    feats.push_back(opp_tracker.range_type_ema);

    return feats;
}

float AIRL::applyEpochReward(float epoch_reward)
{
    // 1. Flush the last hand's results (similar logic to E_NEW_DEAL in onEvent)
    if (!hand_experiences.empty() && hand_experiences[0].log_prob.requires_grad()) {
        // Simple final hand update using its own chip gain signal
        float total_invested = last_wager + last_action_cost;
        float chip_gain = (total_won - total_invested) / buy_in;
        float reward = chip_gain;
        float advantage = reward - reward_baseline;
        reward_baseline = BASELINE_DECAY * reward_baseline + (1.0f - BASELINE_DECAY) * reward;

        torch::Tensor loss = torch::zeros({1});
        for (const auto& exp : hand_experiences) {
            loss = loss - exp.log_prob * advantage - entropy_coeff * exp.entropy;
        }
        // Guard: skip backward if loss is non-finite
        float loss_val = loss.item<float>();
        if (!std::isfinite(loss_val)) {
            // Drop the last hand silently
        } else {
            loss.backward();
            torch::nn::utils::clip_grad_norm_(net->parameters(), MAX_GRAD_NORM);

            // Final hand saliency (NaN-guarded)
            for (const auto& exp : hand_experiences) {
                if (exp.state.grad().defined()) {
                    auto grad_abs = exp.state.grad().abs().sum(0).detach();
                    if (!torch::any(torch::isnan(grad_abs)).item<bool>() &&
                        !torch::any(torch::isinf(grad_abs)).item<bool>()) {
                        if (!accumulated_saliency.defined()) {
                            accumulated_saliency = torch::zeros({TensorConverter::INPUT_SIZE});
                            saliency_count = 0;
                        }
                        accumulated_saliency += grad_abs;
                        saliency_count++;
                    }
                }
            }

            accumulated_loss += loss_val;
            hand_count_in_epoch++;
        }
    }

    // 2. Calculate average loss for the epoch
    float avg_loss = (hand_count_in_epoch > 0) ? (accumulated_loss / hand_count_in_epoch) : 0.0f;

    // 3. Reset for next epoch
    accumulated_loss = 0.0f;
    hand_count_in_epoch = 0;
    hand_experiences.clear();

    // Use epoch_reward (final stack) to update the long-term baseline
    epoch_reward_baseline = EPOCH_BASELINE_DECAY * epoch_reward_baseline
                          + (1.0f - EPOCH_BASELINE_DECAY) * epoch_reward;

    return avg_loss;
}

std::vector<std::pair<int, float>> AIRL::getTopFeatures() {
    std::vector<std::pair<int, float>> result;
    if (!accumulated_saliency.defined() || saliency_count == 0) return result;

    torch::Tensor mean_saliency = (accumulated_saliency / (float)saliency_count);

    // Normalize to relative importance: max feature = 1.0
    // This makes the dashboard readable regardless of absolute gradient scale
    float max_val = mean_saliency.max().item<float>();
    if (max_val > 1e-10f) {
        mean_saliency = mean_saliency / max_val;
    }

    for (int i = 0; i < TensorConverter::INPUT_SIZE; ++i) {
        result.push_back({i, mean_saliency[i].item<float>()});
    }

    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Clear saliency after fetching for the epoch dashboard
    accumulated_saliency = torch::Tensor();
    saliency_count = 0;

    return result;
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
