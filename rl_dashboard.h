#pragma once

#include "observer.h"
#include "observer_statkeeper.h"
#include <vector>
#include <string>
#include <chrono>

class RLDashboard : public Observer {
public:
    RLDashboard();

    // Observer interface — receives live game events
    void onEvent(const Event& event) override;

    // Called once at training start
    void init(int total_epochs, int hands_per_epoch);

    // Called at start of each epoch
    void beginEpoch(int epoch, float lr = 0.0f, float noise = 0.0f);

    void setPhase(const std::string& phase, const std::string& opponent);

    // Called after epoch game completes.
    void endEpoch(float agent_stack, float opponent_stack,
                  float loss_value, float learning_rate, float noise_scale,
                  const std::vector<std::pair<int, float>>& saliency = {},
                  int sub_wins = 0, int sub_games = 1);

    // Called after checkpoint evaluation
    void addEvalResult(int epoch, float eval_stack);

    // Render the dashboard
    void render();

    // Write a plain-text final summary to FINAL_DASHBOARD_SUMMARY.txt
    void writeFinalSummary(const std::string& output_path = "FINAL_DASHBOARD_SUMMARY.txt");

    // Log metrics to a CSV file
    void logMetrics();

private:
    int total_epochs;
    int hands_per_epoch;

    // Current epoch
    int epoch;
    int hands_this_epoch;
    ObserverStatKeeper observer;

    // Per-epoch results
    float agent_stack;
    float opponent_stack;
    float loss_value;
    float learning_rate;
    float noise_scale;
    std::vector<std::pair<int, float>> current_saliency;
    std::string training_phase;
    std::string opponent_name;

    // History
    struct EpochSnapshot {
        float agent_stack;
        float opponent_stack;
        float win_rate; // agent_stack > opponent_stack
    };
    std::vector<EpochSnapshot> history;

    // Eval history (every N epochs vs AISmart baseline)
    struct EvalSnapshot {
        int epoch;
        float stack;
    };
    std::vector<EvalSnapshot> eval_history;

    // Running stats
    int total_wins;
    int total_games;
    int total_epochs_completed;

    // Timing
    std::chrono::steady_clock::time_point train_start;
    std::chrono::steady_clock::time_point epoch_start;

    // Helpers
    static std::string makeProgressBar(int current, int total, int width);
    static std::string makeSparkline(const std::vector<float>& values, int max_width);
    static std::string formatTime(double seconds);
    static std::string formatFloat(float val, int precision, bool show_sign);
};
