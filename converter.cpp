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

torch::Tensor TensorConverter::infoToTensor(const Info& info, const std::vector<float>& opponent_stats)
{
  std::vector<float> features;
  // total size: 23 (Static) + 11 (Opponent) = 34
  features.reserve(INPUT_SIZE);

  int bb = info.getBigBlind();

  //  hole cards
  const auto& hole = info.getHoleCards();
  if (hole.size() >= 2)
  {
    Card c = hole[0];
    features.push_back((c.value - 2.0f) / 12.0f);
    features.push_back((float)c.suit / 3.0f);
    c = hole[1];
    features.push_back((c.value - 2.0f) / 12.0f);
    features.push_back((float)c.suit / 3.0f);
  }
  else
  {
    features.push_back(-1.0f); features.push_back(-1.0f);
    features.push_back(-1.0f); features.push_back(-1.0f);
  }

  //  board cards
  for (int i = 0; i < 5; ++i)
  {
    if (i < (int)info.boardCards.size())
    {
      Card c = info.boardCards[i];
      features.push_back((c.value - 2.0f) / 12.0f);
      features.push_back((float)c.suit / 3.0f);
    }
    else
    {
      features.push_back(-1.0f); features.push_back(-1.0f);
    }
  }

  //  game state features [14-22]
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

  // Append Opponent Stats [23-33]
  if (opponent_stats.size() == OPPONENT_SIZE) {
      features.insert(features.end(), opponent_stats.begin(), opponent_stats.end());
  } else {
      // Fallback if size mismatch (should warn or error, but filling with 0s for safety)
      for(int i=0; i<OPPONENT_SIZE; ++i) features.push_back(0.0f);
  }

  return torch::from_blob(features.data(), {1, INPUT_SIZE}, torch::kFloat).clone();
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
