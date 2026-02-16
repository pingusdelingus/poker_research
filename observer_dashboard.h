#pragma once

#include "observer.h"
#include "statistics.h"
#include <string>
#include <vector>
#include <map>
#include <chrono>

class ObserverDashboard : public Observer
{
  private:
    StatKeeper* epochKeeper;
    StatKeeper allTimeKeeper;

    int dealCount;
    int totalDeals;
    int epoch;

    std::chrono::steady_clock::time_point epochStart;
    std::chrono::steady_clock::time_point trainingStart;

    struct EvalResult {
      int epoch;
      int botStack;
    };
    std::vector<EvalResult> evalHistory;

    std::map<std::string, int> epochWins; // track how many epochs each player won

    void redraw();
    void renderStatsTable(std::stringstream& ss, StatKeeper& keeper, bool showEpochWins);
    std::string makeProgressBar(int current, int total, int width);
    void tallyEpochWinner();

  public:
    ObserverDashboard(int totalDeals);
    ~ObserverDashboard();

    virtual void onEvent(const Event& event);

    void setEpoch(int epoch);
    void addEvalResult(int epoch, int botStack);
    StatKeeper& getEpochStatKeeper();
};
