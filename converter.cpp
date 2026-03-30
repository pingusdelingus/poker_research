#include "converter.h"
#include <cmath>
#include <algorithm>
#include "action.h"
#include "info.h"
#include "ai_rl.h"
#include "pokermath.h"

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
  // total size: 25 (Static) + 22 (Opponent) = 47
  features.reserve(INPUT_SIZE);

  int bb = info.getBigBlind();
  const auto& hole = info.getHoleCards();
  const auto& board = info.boardCards;

  // 0-3: Hole cards
  if (hole.size() >= 2)
  {
    features.push_back((hole[0].value - 2.0f) / 12.0f);
    features.push_back((float)hole[0].suit / 3.0f);
    features.push_back((hole[1].value - 2.0f) / 12.0f);
    features.push_back((float)hole[1].suit / 3.0f);
  }
  else
  {
    for(int i=0; i<4; ++i) features.push_back(-1.0f);
  }

  // 4-13: Board cards
  for (int i = 0; i < 5; ++i)
  {
    if (i < (int)board.size())
    {
      features.push_back((board[i].value - 2.0f) / 12.0f);
      features.push_back((float)board[i].suit / 3.0f);
    }
    else
    {
      features.push_back(-1.0f); features.push_back(-1.0f);
    }
  }

  // 14-22: Game state
  features.push_back(normalize(info.getPot(), bb));                                       // [14] pot
  features.push_back(normalize(info.getStack(), bb));                                     // [15] stack
  features.push_back(normalize(info.getCallAmount(), bb));                                // [16] call amount
  features.push_back(normalize(info.getWager(), bb));                                     // [17] wager
  features.push_back((float)info.getPosition() / (float)std::max(1, info.getNumPlayers() - 1)); // [18] position
  features.push_back((float)info.getPotEquity());                                         // [19] pot equity
  features.push_back((float)info.getPotOddsPercentage());                                 // [20] pot odds pct
  features.push_back((float)info.getMRatio() / 50.0f);                                    // [21] m-ratio
  features.push_back((float)info.getNumActivePlayers() / 9.0f);                           // [22] active players

  // 23: Board Texture (normalized 0-1)
  BoardTexture bt = getBoardTexture(board);
  features.push_back((float)bt / 6.0f); // 0-6 range

  // 24: Opponent Range Type (normalized 0-1)
  int oppIdx = (info.yourIndex == 0) ? 1 : 0;
  if (info.getNumPlayers() > 2) {
      // Find first active opponent that is not you
      for (int i = 0; i < info.getNumPlayers(); i++) {
          if (i != info.yourIndex && !info.players[i].folded) {
              oppIdx = i;
              break;
          }
      }
  }
  RangeType rt = getOpponentRangeType(info, oppIdx);
  features.push_back((float)rt / 2.0f); // 0-2 range

  // Append Opponent Stats [26-40]
  if (opponent_stats.size() == OPPONENT_SIZE) {
      features.insert(features.end(), opponent_stats.begin(), opponent_stats.end());
  } else {
      for(int i=0; i<OPPONENT_SIZE; ++i) features.push_back(0.0f);
  }

  return torch::from_blob(features.data(), {1, INPUT_SIZE}, torch::kFloat).clone();
} // end of infototensor

Action TensorConverter::logitsToAction(const Info& info, int action_idx, float sizing)
{
  // 0: Fold/Check
  // 1: Call
  // 2: Raise
  
  if (action_idx == 0)
  {
    return info.getCheckFoldAction();
  }
  
  if (action_idx == 1)
  {
    return info.getCallAction();
  }

  // Raise Action
  // map continuous sizing parameter to a percentage of the remaining max stack using sigmoid
  double strength = 1.0 / (1.0 + std::exp(-sizing)); 
  
  int min_r = info.getMinChipsToRaise();
  int max_r = info.getStack();

  if (min_r > max_r) return info.getAllInAction();

  int amount = min_r + (int)((max_r - min_r) * strength);
  return info.amountToAction(amount);
} // end of logitsToAction
