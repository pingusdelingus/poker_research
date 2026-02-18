#pragma once
#include <vector>
#include <string>
#include <array>

class GADashboard;

// =========================================================
// Opponent types (maps to paper's 4 rule-based opponents)
// =========================================================
enum OpponentType {
    OPP_CHECKFOLD = 0, // "Scared Limper"
    OPP_CALL      = 1, // "Calling Machine"
    OPP_RAISE     = 2, // "Hothead Maniac"
    OPP_SMART     = 3, // "Candid Statistician"
    OPP_COUNT     = 4
};

// =========================================================
// Individual: one member of the population
// =========================================================
struct Individual {
    std::vector<float> genome;
    float fitness;                               // Average Normalized Earnings (ANE)
    std::array<float, OPP_COUNT> earnings;       // cumulative earnings per opponent

    Individual();
    void randomize(int genome_size);
};

// =========================================================
// GeneticTrainer: GA training loop with TSER selection
// =========================================================
class GeneticTrainer {
public:
    struct Config {
        int population_size       = 50;
        int num_generations       = 250;
        int hands_per_session     = 500;
        float survival_rate       = 0.30f;

        // Mutation schedule (linear decay over generations)
        float mutation_rate_start     = 0.25f;
        float mutation_rate_end       = 0.05f;
        float mutation_strength_start = 0.50f;
        float mutation_strength_end   = 0.10f;

        // Game rules
        int buy_in     = 1000;
        int big_blind  = 10;
        int small_blind = 5;

        // Logging
        std::string log_dir       = "./logs/ga/";
        int checkpoint_interval   = 10;
    };

    GeneticTrainer(const Config& config);

    void train();

    const Individual& getBest() const;

    void saveBestGenome(const std::string& path) const;
    void savePopulation(const std::string& path) const;
    void loadPopulation(const std::string& path);

private:
    Config config;
    std::vector<Individual> population;
    int genome_size;
    int generation;

    // --- Core GA ---
    void evaluatePopulation(GADashboard& dashboard);
    float runSession(const std::vector<float>& genome, OpponentType opp_type,
                     bool evolved_is_player0, GADashboard& dashboard);
    void select();
    std::vector<float> crossover(const std::vector<float>& a,
                                  const std::vector<float>& b);
    void mutate(std::vector<float>& genome, float rate, float strength);

    // Mutation schedule
    float getMutationRate() const;
    float getMutationStrength() const;

    // Logging
    void logGeneration();
};
