#pragma once

#include "observer_statkeeper.h"
#include "genetic_trainer.h"
#include <array>
#include <vector>
#include <string>
#include <chrono>

class GADashboard {
public:
    GADashboard();

    // Called once at training start
    void init(int total_generations, int population_size, int genome_size,
              int hands_per_session);

    // Called at the start of each generation
    void beginGeneration(int gen, float mutation_rate, float mutation_strength);

    // Called before/after each evaluation session
    void beginSession(int individual_idx, OpponentType opp, int seat_num);
    void endSession(float earnings, bool evolved_won);

    // Get the observer to attach to Game via addObserverBorrowed()
    ObserverStatKeeper& getObserver();

    // Called after evaluation is complete with population results
    void setPopulationResults(const std::vector<Individual>& sorted_pop,
                              float avg_fitness,
                              int num_elites, int num_survivors);

    // Render the full dashboard to stdout
    void render();

private:
    // --- Config (set once) ---
    int total_generations;
    int population_size;
    int genome_size;
    int hands_per_session;
    int total_sessions_per_gen; // pop * 4 opponents * 2 seats

    // --- Per-generation state ---
    int generation;
    float mutation_rate;
    float mutation_strength;
    int sessions_completed;

    // Current session info (for progress display)
    int current_individual;
    OpponentType current_opponent;
    int current_seat; // 1 or 2

    // Per-generation stats observer (reset each generation)
    ObserverStatKeeper observer;

    // Per-opponent tracking for current generation (best individual's results)
    struct OpponentRecord {
        int evolved_wins;
        int opponent_wins;
        int total_sessions;
        float total_earnings; // sum of raw earnings before ANE normalization

        OpponentRecord() : evolved_wins(0), opponent_wins(0),
                          total_sessions(0), total_earnings(0.0f) {}
    };
    std::array<OpponentRecord, OPP_COUNT> opponent_records;

    // Population results (set after evaluation)
    float best_fitness, avg_fitness, worst_fitness;
    std::array<float, OPP_COUNT> best_per_opponent; // best individual's per-opponent earnings
    int num_elites, num_survivors;

    // --- Generation history ---
    struct GenSnapshot {
        float best_fitness;
        float avg_fitness;
        float worst_fitness;
        std::array<float, OPP_COUNT> per_opponent;
    };
    std::vector<GenSnapshot> history;

    // --- Timing ---
    std::chrono::steady_clock::time_point train_start;
    std::chrono::steady_clock::time_point gen_start;
    std::chrono::steady_clock::time_point session_start;

    // --- Rendering helpers ---
    static std::string makeProgressBar(int current, int total, int width);
    static std::string makeSparkline(const std::vector<float>& values, int max_width);
    static const char* opponentName(OpponentType opp);
    static std::string formatTime(double seconds);
    static std::string formatPct(double val); // format as XX.X%
    static std::string formatFloat(float val, int precision, bool show_sign);
};
