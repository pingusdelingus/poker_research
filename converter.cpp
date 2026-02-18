#include "converter.h"
#include <cmath>
#include <algorithm>
#include "action.h"
#include "info.h"
#include "ai_rl.h"
// helper to normalize money values by the big blind

const double pi = 3.1415926535;

float normalize(int value, int big_blind)
{
  if (big_blind == 0) return 0.0f;
  return (float)value / (float)big_blind;
} // end of normalize

void TensorConverter::encodeCard(const Card& c, std::vector<float>& features)
{
  // normalize rank 2-14 to 0.0-1.0 and suit 0-3 to 0.0-1.0
  features.push_back((c.value - 2.0f) / 12.0f);
  features.push_back((float)c.suit / 3.0f);
} // end of encodecard

void TensorConverter::encodeEmptyCard(std::vector<float>& features)
{
  features.push_back(-1.0f);
  features.push_back(-1.0f);
} // end of encodeemptycard

torch::Tensor TensorConverter::infoToTensor(const Info& info)
{
  std::vector<float> features;
  // total size: 4 (hole) + 10 (board) + 14 (game state + derived) = 28
  features.reserve(28);

  int bb = info.getBigBlind();

  //  hole cards
  const auto& hole = info.getHoleCards();
  if (hole.size() >= 2)
  {
    encodeCard(hole[0], features);
    encodeCard(hole[1], features);
  }
  else
  {
    encodeEmptyCard(features);
    encodeEmptyCard(features);
  }

  //  board cards
  for (int i = 0; i < 5; ++i)
  {
    if (i < (int)info.boardCards.size())
    {
      encodeCard(info.boardCards[i], features);
    }
    else
    {
      encodeEmptyCard(features);
    }
  }

  //  game state features
  features.push_back(normalize(info.getPot(), bb));                                       // [14] pot
  features.push_back(normalize(info.getStack(), bb));                                     // [15] stack
  features.push_back(normalize(info.getCallAmount(), bb));                                // [16] call amount
  features.push_back(normalize(info.getWager(), bb));                                     // [17] wager

  float pos = (float)info.getPosition() / (float)std::max(1, info.getNumPlayers() - 1);
  features.push_back(pos);                                                                // [18] position

  float equity = (float)info.getPotEquity();
  float potOddsPct = (float)info.getPotOddsPercentage();
  features.push_back(equity);                                                             // [19] pot equity
  features.push_back(potOddsPct);                                                         // [20] pot odds pct
  features.push_back((float)info.getMRatio() / 50.0f);                                    // [21] m-ratio
  features.push_back((float)info.getNumActivePlayers() / 9.0f);                           // [22] active players

  //  new derived features
  // betting round: 0=preflop, 0.33=flop, 0.67=turn, 1.0=river
  float round_norm = 0.0f;
  if (info.round == R_FLOP) round_norm = 0.33f;
  else if (info.round == R_TURN) round_norm = 0.67f;
  else if (info.round == R_RIVER || info.round == R_SHOWDOWN) round_norm = 1.0f;
  features.push_back(round_norm);                                                         // [23] betting round

  // equity vs pot odds gap: positive = calling is +EV
  features.push_back(equity - potOddsPct);                                                // [24] equity gap

  // pot commitment ratio: how much of starting chips are already in the pot
  float total_chips = (float)(info.getStack() + info.getWager());
  float commit_ratio = (total_chips > 0) ? (float)info.getWager() / total_chips : 0.0f;
  features.push_back(commit_ratio);                                                       // [25] pot commitment

  // stack-to-pot ratio: room for maneuvering (capped at 20 for normalization)
  int pot = info.getPot();
  float spr = (pot > 0) ? std::min(20.0f, (float)info.getStack() / (float)pot) / 20.0f : 1.0f;
  features.push_back(spr);                                                                // [26] SPR

  // turn number within the current betting round (how many re-raises)
  features.push_back((float)info.turn / 1000.0f);                                        // [27] turn number

  return torch::from_blob(features.data(), {1, 28}, torch::kFloat).clone();
} // end of infototensor

torch::Tensor TensorConverter::actionToTarget(const Action& action, const Info& info)
{
  float x = 0.0f, y = 0.0f;

  // mapping: fold = up, call/check = left, raise = right
  switch (action.command)
  {
    case A_FOLD:
      x = 0.0f; y = 1.0f;
      break;
    case A_CHECK:
    case A_CALL:
      x = -1.0f; y = 0.0f;
      break;
    case A_RAISE:
      x = 1.0f;
      // use y to represent raise sizing relative to stack
      // this teaches the network sizing during imitation
      float total_stack = (float)info.getStack();
      y = (total_stack > 0) ? (float)action.amount / total_stack : 0.0f;
      break;
  }

  float target[] = {x, y};
  return torch::from_blob(target, {1, 2}, torch::kFloat).clone();
} // end of actiontotarget

Action TensorConverter::vectorToAction(const Info& info, float x, float y)
{
  float angle = std::atan2(y, x);
  float magnitude = std::sqrt(x*x + y*y);

  // fold zone
  if (angle > pi/3 && angle < 2*pi/3)
  {
    return info.getCheckFoldAction();
  }
  
  // call zone
  if (std::abs(angle) > 2*pi/3)
  {
    return info.getCallAction();
  }

  // raise zone logic
  // map strength to a stack percentage
  double strength = 1.0 / (1.0 + std::exp(-magnitude)); 
  
  int min_r = info.getMinChipsToRaise();
  int max_r = info.getStack();

  if (min_r > max_r) return info.getAllInAction();

  int amount = min_r + (int)((max_r - min_r) * strength);
  return info.amountToAction(amount);
} // end of vectortoaction
