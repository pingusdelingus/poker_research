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
    void beginEpoch(int epoch);

    // Called after epoch game completes
    void endEpoch(float agent_stack, float opponent_stack,
                  float loss_value, float learning_rate, float noise_scale);

    // Called after checkpoint evaluation
    void addEvalResult(int epoch, float eval_stack);

    // Render the dashboard
    void render();

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
