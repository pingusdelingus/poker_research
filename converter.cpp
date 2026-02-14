#include "converter.h"
#include <cmath>
#include <algorithm>
#include "action.h"
#include "info.h"
#include "ai_rl.h"
// helper to normalize money values by the big blind
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
  // expanded input size to include equity and m-ratio
  // total size: 4 (hole) + 10 (board) + 9 (stats) = 23
  features.reserve(23);

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

  //  game state and mathematical features
  features.push_back(normalize(info.getPot(), bb));
  features.push_back(normalize(info.getStack(), bb));
  features.push_back(normalize(info.getCallAmount(), bb));
  features.push_back(normalize(info.getWager(), bb));
  
  // normalized position 0 to 1
  float pos = (float)info.getPosition() / (float)std::max(1, info.getNumPlayers() - 1);
  features.push_back(pos);

  // use oopoker built in math for better learning
  features.push_back((float)info.getPotEquity());
  features.push_back((float)info.getPotOddsPercentage());
  features.push_back((float)info.getMRatio() / 50.0f); // cap m-ratio at 50 for normalization
  features.push_back((float)info.getNumActivePlayers() / 9.0f);

  return torch::from_blob(features.data(), {1, 23}, torch::kFloat).clone();
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
  double pi = 3.1415926535;

  // fold zone
  if (angle > pi/4 && angle < 3*pi/4)
  {
    return info.getCheckFoldAction();
  }
  
  // call zone
  if (std::abs(angle) > 3*pi/4)
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
