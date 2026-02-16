#include "observer_dashboard.h"
#include "event.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

ObserverDashboard::ObserverDashboard(int totalDeals)
: epochKeeper(new StatKeeper())
, dealCount(0)
, totalDeals(totalDeals)
, epoch(0)
{
  trainingStart = std::chrono::steady_clock::now();
  epochStart = trainingStart;
  // Clear screen once at start
  std::cout << "\033[2J\033[H" << std::flush;
}

ObserverDashboard::~ObserverDashboard()
{
  delete epochKeeper;
}

void ObserverDashboard::tallyEpochWinner()
{
  // Find who won most chips this epoch
  std::vector<std::string> players;
  epochKeeper->getAllPlayers(players);

  std::string winner;
  int bestChipsWon = -1;
  for(const auto& name : players) {
    const PlayerStats* stats = epochKeeper->getPlayerStats(name);
    if(!stats) continue;
    if(stats->chips_won > bestChipsWon) {
      bestChipsWon = stats->chips_won;
      winner = name;
    }
  }
  if(!winner.empty()) {
    epochWins[winner]++;
  }
}

void ObserverDashboard::setEpoch(int e)
{
  // Tally the winner of the previous epoch before resetting
  if(e > 0) {
    tallyEpochWinner();
  }
  epoch = e;
  dealCount = 0;
  delete epochKeeper;
  epochKeeper = new StatKeeper();
  epochStart = std::chrono::steady_clock::now();
}

void ObserverDashboard::addEvalResult(int ep, int botStack)
{
  evalHistory.push_back({ep, botStack});
  if(evalHistory.size() > 10) {
    evalHistory.erase(evalHistory.begin());
  }
  redraw();
}

StatKeeper& ObserverDashboard::getEpochStatKeeper()
{
  return *epochKeeper;
}

void ObserverDashboard::onEvent(const Event& event)
{
  epochKeeper->onEvent(event);
  allTimeKeeper.onEvent(event);

  if(event.type == E_NEW_DEAL) {
    dealCount++;
  }

  if(event.type == E_POT_DIVISION) {
    if(dealCount % 10 == 0 || dealCount == totalDeals) {
      redraw();
    }
  }
}

std::string ObserverDashboard::makeProgressBar(int current, int total, int width)
{
  double ratio = (total > 0) ? (double)current / total : 0.0;
  int filled = (int)(ratio * width);
  std::string bar = "[";
  for(int i = 0; i < width; i++) {
    bar += (i < filled) ? "#" : "-";
  }
  bar += "]";
  return bar;
}

void ObserverDashboard::renderStatsTable(std::stringstream& ss, StatKeeper& keeper, bool showEpochWins)
{
  std::string lastColHeader = showEpochWins ? "Ep Wins " : "Chips+/-";
  ss << "  +-----------------+--------+--------+--------+---------+\n";
  ss << "  | Player          |  VPIP  |  PFR   |   AF   | " << lastColHeader << "|\n";
  ss << "  +-----------------+--------+--------+--------+---------+\n";

  std::vector<std::string> players;
  keeper.getAllPlayers(players);

  // Calculate total chips_won across all players to find the average (for per-epoch view)
  int totalChipsWon = 0;
  int numPlayers = 0;
  for(const auto& name : players) {
    const PlayerStats* stats = keeper.getPlayerStats(name);
    if(!stats || stats->deals == 0) continue;
    totalChipsWon += stats->chips_won;
    numPlayers++;
  }
  int avgChipsWon = (numPlayers > 0) ? totalChipsWon / numPlayers : 0;

  for(const auto& name : players) {
    const PlayerStats* stats = keeper.getPlayerStats(name);
    if(!stats || stats->deals == 0) continue;

    std::string displayName = name.substr(0, 15);

    ss << "  | " << std::left << std::setw(15) << displayName << " | ";

    // VPIP
    double vpip = stats->getVPIP();
    if(std::isnan(vpip) || std::isinf(vpip)) ss << std::right << std::setw(5) << "--" << "% | ";
    else ss << std::right << std::setw(5) << std::fixed << std::setprecision(1) << (vpip * 100) << "% | ";

    // PFR
    double pfr = stats->getPFR();
    if(std::isnan(pfr) || std::isinf(pfr)) ss << std::setw(5) << "--" << "% | ";
    else ss << std::setw(5) << std::fixed << std::setprecision(1) << (pfr * 100) << "% | ";

    // AF
    double af = stats->getAF();
    if(std::isnan(af) || std::isinf(af)) ss << std::setw(6) << "--" << " | ";
    else ss << std::setw(6) << std::fixed << std::setprecision(2) << af << " | ";

    // Last column
    if(showEpochWins) {
      auto it = epochWins.find(name);
      int wins = (it != epochWins.end()) ? it->second : 0;
      ss << std::setw(7) << wins << " |\n";
    } else {
      int chipsPnl = stats->chips_won - avgChipsWon;
      ss << std::showpos << std::setw(7) << chipsPnl << std::noshowpos << " |\n";
    }
  }

  ss << "  +-----------------+--------+--------+--------+---------+\n";
}

void ObserverDashboard::redraw()
{
  auto now = std::chrono::steady_clock::now();
  double epochElapsed = std::chrono::duration<double>(now - epochStart).count();
  double totalElapsed = std::chrono::duration<double>(now - trainingStart).count();

  double dps = (epochElapsed > 0 && dealCount > 0) ? dealCount / epochElapsed : 0;
  double eta = (dps > 0 && totalDeals > dealCount) ? (totalDeals - dealCount) / dps : 0;

  // Format total time
  int th = (int)(totalElapsed / 3600);
  int tm = (int)(std::fmod(totalElapsed, 3600) / 60);
  int ts = (int)(std::fmod(totalElapsed, 60));

  // Format ETA
  int em = (int)(eta / 60);
  int es = (int)(std::fmod(eta, 60));

  std::stringstream ss;

  // Move cursor home (no clear — avoids flicker)
  ss << "\033[H";

  // Header
  ss << "======================================================\n";
  ss << "          OOPoker RL Training Dashboard\n";
  ss << "======================================================\n\n";

  // Epoch and timing
  ss << "  Epoch: " << epoch << "    |    Total Time: ";
  if(th > 0) ss << th << "h ";
  if(tm > 0 || th > 0) ss << tm << "m ";
  ss << ts << "s\n";

  ss << "  Deals: " << dealCount << " / " << totalDeals
     << "    |    " << std::fixed << std::setprecision(0) << dps << " deals/sec"
     << "    |    ETA: " << em << "m " << es << "s\n";

  double pct = (totalDeals > 0) ? (100.0 * dealCount / totalDeals) : 0;
  ss << "  " << makeProgressBar(dealCount, totalDeals, 40)
     << " " << std::fixed << std::setprecision(1) << pct << "%\n\n";

  // Per-epoch stats
  ss << "  --- This Epoch ---\n";
  renderStatsTable(ss, *epochKeeper, false);
  ss << "\n";

  // All-time stats
  ss << "  --- All Time (" << epoch + 1 << " epochs) ---\n";
  renderStatsTable(ss, allTimeKeeper, true);
  ss << "\n";

  // Evaluation history
  if(!evalHistory.empty()) {
    ss << "  Recent Evaluations (vs OOPoker Bot baseline):\n";
    ss << "  +----------+-------------+\n";
    ss << "  |  Epoch   |  Bot Stack  |\n";
    ss << "  +----------+-------------+\n";
    for(const auto& r : evalHistory) {
      ss << "  | " << std::setw(8) << r.epoch
         << " | " << std::setw(11) << r.botStack << " |\n";
    }
    ss << "  +----------+-------------+\n\n";
  }

  // Footer
  ss << "  Press 'q' to quit training.\n";

  // Replace \n with \033[K\n (clear to end of line) to avoid stale chars
  std::string output = ss.str();
  std::string cleaned;
  cleaned.reserve(output.size() + 200);
  for(char ch : output) {
    if(ch == '\n') cleaned += "\033[K\n";
    else cleaned += ch;
  }
  // Clear any remaining lines from previous longer output
  cleaned += "\033[J";
  std::cout << cleaned << std::flush;
}
