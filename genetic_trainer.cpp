#include "genetic_trainer.h"
#include "ga_dashboard.h"
#include "ai_evolved.h"
#include "ai_call.h"
#include "ai_checkfold.h"
#include "ai_raise.h"
#include "ai_smart.h"
#include "game.h"
#include "player.h"
#include "host_silent.h"
#include "random.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =========================================================
// Individual
// =========================================================
Individual::Individual() : fitness(0.0f)
{
    earnings.fill(0.0f);
}

void Individual::randomize(int genome_size)
{
    genome.resize(genome_size);
    // Gaussian initialization: mean=0, std=0.5
    for (int i = 0; i < genome_size; i++) {
        float u1 = static_cast<float>(getRandom()) + 1e-10f;
        float u2 = static_cast<float>(getRandom());
        float noise = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * static_cast<float>(M_PI) * u2);
        genome[i] = 0.5f * noise;
    }
    fitness = 0.0f;
    earnings.fill(0.0f);
}

// =========================================================
// GeneticTrainer
// =========================================================
GeneticTrainer::GeneticTrainer(const Config& config)
    : config(config), generation(0)
{
    // Determine genome size from a template network
    AIEvolved template_agent;
    genome_size = template_agent.genomeSize();

    // Initialize population
    population.resize(config.population_size);
    for (int i = 0; i < config.population_size; i++) {
        population[i].randomize(genome_size);
    }
}

// =========================================================
// Run a single session: evolved player vs one opponent
// Returns normalized earnings: (final_stack - buy_in) / buy_in
// =========================================================
float GeneticTrainer::runSession(const std::vector<float>& genome,
                                  OpponentType opp_type,
                                  bool evolved_is_player0,
                                  GADashboard& dashboard)
{
    HostSilent host;
    Game game(&host);
    game.setSilent(true);

    Rules rules;
    rules.buyIn = config.buy_in;
    rules.bigBlind = config.big_blind;
    rules.smallBlind = config.small_blind;
    rules.allowRebuy = false;
    rules.fixedNumberOfDeals = config.hands_per_session;
    game.setRules(rules);

    // Attach the dashboard's observer for stat tracking
    game.addObserverBorrowed(&dashboard.getObserver());

    // Create AIs
    AIEvolved* evolved = new AIEvolved(genome);
    AI* opponent = nullptr;
    switch (opp_type) {
        case OPP_CHECKFOLD: opponent = new AICheckFold(); break;
        case OPP_CALL:      opponent = new AICall();      break;
        case OPP_RAISE:     opponent = new AIRaise();     break;
        case OPP_SMART:     opponent = new AISmart();     break;
        default:            opponent = new AISmart();      break;
    }

    std::string evolved_name = "Evolved";
    std::string opp_name = "Opponent";

    if (evolved_is_player0) {
        game.addPlayer(Player(evolved, evolved_name));
        game.addPlayer(Player(opponent, opp_name));
    } else {
        game.addPlayer(Player(opponent, opp_name));
        game.addPlayer(Player(evolved, evolved_name));
    }

    game.doGame();

    int final_stack = game.getFinalStack(evolved_name);
    return static_cast<float>(final_stack - config.buy_in) / static_cast<float>(config.buy_in);
}

// =========================================================
// Evaluate all individuals against all opponent types
// Fitness = Average Normalized Earnings (ANE)
// =========================================================
void GeneticTrainer::evaluatePopulation(GADashboard& dashboard)
{
    for (int i = 0; i < static_cast<int>(population.size()); i++) {
        auto& ind = population[i];
        ind.earnings.fill(0.0f);

        for (int opp = 0; opp < OPP_COUNT; opp++) {
            // Session 1: evolved as player 0
            dashboard.beginSession(i, static_cast<OpponentType>(opp), 1);
            float e1 = runSession(ind.genome, static_cast<OpponentType>(opp), true, dashboard);
            bool won1 = (e1 > 0.0f);
            dashboard.endSession(e1, won1);

            // Session 2: evolved as player 1 (swapped seats)
            dashboard.beginSession(i, static_cast<OpponentType>(opp), 2);
            float e2 = runSession(ind.genome, static_cast<OpponentType>(opp), false, dashboard);
            bool won2 = (e2 > 0.0f);
            dashboard.endSession(e2, won2);

            // Cumulative earnings across both sessions
            ind.earnings[opp] = e1 + e2;
        }

        // Render progress periodically (every 5 individuals)
        if (i % 5 == 0 || i == static_cast<int>(population.size()) - 1) {
            dashboard.render();
        }
    }

    // Compute ANE: f(i) = (1/m) * sum_j(e_ij / n_j)
    // where n_j = max(BB_normalized, max_i(e_ij))
    float bb_norm = static_cast<float>(config.big_blind) / static_cast<float>(config.buy_in);

    for (int opp = 0; opp < OPP_COUNT; opp++) {
        float max_earning = bb_norm;
        for (const auto& ind : population) {
            max_earning = std::max(max_earning, ind.earnings[opp]);
        }

        // Normalize each individual's earnings for this opponent
        for (auto& ind : population) {
            ind.earnings[opp] /= max_earning;
        }
    }

    // Final fitness = average of normalized earnings
    for (auto& ind : population) {
        float sum = 0.0f;
        for (int opp = 0; opp < OPP_COUNT; opp++) {
            sum += ind.earnings[opp];
        }
        ind.fitness = sum / static_cast<float>(OPP_COUNT);
    }
}

// =========================================================
// Selection: TSER (Tiered Survival and Elite Reproduction)
// =========================================================
void GeneticTrainer::select()
{
    // Sort by fitness (descending)
    std::sort(population.begin(), population.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness > b.fitness;
              });

    int num_survivors = std::max(2, static_cast<int>(config.population_size * config.survival_rate));

    // Compute average fitness among survivors
    float avg_fitness = 0.0f;
    for (int i = 0; i < num_survivors; i++) {
        avg_fitness += population[i].fitness;
    }
    avg_fitness /= static_cast<float>(num_survivors);

    // Classify into elites (above average) and second tier
    int num_elites = 0;
    for (int i = 0; i < num_survivors; i++) {
        if (population[i].fitness >= avg_fitness) num_elites++;
    }
    // Ensure at least 2 elites for crossover
    num_elites = std::max(2, num_elites);
    num_elites = std::min(num_elites, num_survivors);

    float rate = getMutationRate();
    float strength = getMutationStrength();

    // Build new population
    std::vector<Individual> new_pop;
    new_pop.reserve(config.population_size);

    // Elites: survive unchanged (immune to mutation)
    for (int i = 0; i < num_elites; i++) {
        new_pop.push_back(population[i]);
    }

    // Second tier: survive but get mutated
    for (int i = num_elites; i < num_survivors; i++) {
        Individual ind = population[i];
        mutate(ind.genome, rate, strength);
        new_pop.push_back(ind);
    }

    // Fill remaining with offspring from elite crossover + mutation
    while (static_cast<int>(new_pop.size()) < config.population_size) {
        int pa = getRandom(0, num_elites - 1);
        int pb = getRandom(0, num_elites - 1);
        while (pb == pa && num_elites > 1) {
            pb = getRandom(0, num_elites - 1);
        }

        Individual child;
        child.genome = crossover(population[pa].genome, population[pb].genome);
        mutate(child.genome, rate, strength);
        new_pop.push_back(child);
    }

    population = new_pop;
}

// =========================================================
// Crossover: interleave odd/even genome indices
// =========================================================
std::vector<float> GeneticTrainer::crossover(const std::vector<float>& a,
                                              const std::vector<float>& b)
{
    std::vector<float> child(a.size());
    for (size_t i = 0; i < child.size(); i++) {
        child[i] = (i % 2 == 0) ? a[i] : b[i];
    }
    return child;
}

// =========================================================
// Mutation: per-gene Gaussian noise
// =========================================================
void GeneticTrainer::mutate(std::vector<float>& genome, float rate, float strength)
{
    for (size_t i = 0; i < genome.size(); i++) {
        if (static_cast<float>(getRandom()) < rate) {
            // Box-Muller for Gaussian noise
            float u1 = static_cast<float>(getRandom()) + 1e-10f;
            float u2 = static_cast<float>(getRandom());
            float noise = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * static_cast<float>(M_PI) * u2);
            genome[i] += strength * noise;
        }
    }
}

// =========================================================
// Mutation schedule (linear decay)
// =========================================================
float GeneticTrainer::getMutationRate() const
{
    if (config.num_generations <= 1) return config.mutation_rate_start;
    float t = static_cast<float>(generation) / static_cast<float>(config.num_generations - 1);
    return config.mutation_rate_start + t * (config.mutation_rate_end - config.mutation_rate_start);
}

float GeneticTrainer::getMutationStrength() const
{
    if (config.num_generations <= 1) return config.mutation_strength_start;
    float t = static_cast<float>(generation) / static_cast<float>(config.num_generations - 1);
    return config.mutation_strength_start + t * (config.mutation_strength_end - config.mutation_strength_start);
}

// =========================================================
// Main training loop
// =========================================================
void GeneticTrainer::train()
{
    GADashboard dashboard;
    dashboard.init(config.num_generations, config.population_size,
                   genome_size, config.hands_per_session);

    for (generation = 0; generation < config.num_generations; generation++) {
        // 1. Begin generation
        dashboard.beginGeneration(generation, getMutationRate(), getMutationStrength());

        // 2. Evaluate
        evaluatePopulation(dashboard);

        // 3. Sort for results
        std::sort(population.begin(), population.end(),
                  [](const Individual& a, const Individual& b) {
                      return a.fitness > b.fitness;
                  });

        float avg_fit = 0.0f;
        for (const auto& ind : population) avg_fit += ind.fitness;
        avg_fit /= static_cast<float>(population.size());

        // Compute elites/survivors for display
        int num_survivors = std::max(2, static_cast<int>(config.population_size * config.survival_rate));
        float survivor_avg = 0.0f;
        for (int i = 0; i < num_survivors; i++) survivor_avg += population[i].fitness;
        survivor_avg /= static_cast<float>(num_survivors);
        int num_elites = 0;
        for (int i = 0; i < num_survivors; i++) {
            if (population[i].fitness >= survivor_avg) num_elites++;
        }
        num_elites = std::max(2, std::min(num_elites, num_survivors));

        // 4. Update dashboard with results and render
        dashboard.setPopulationResults(population, avg_fit, num_elites, num_survivors);
        dashboard.render();

        // 5. Log to CSV
        logGeneration();

        // 6. Checkpoint
        if (generation % config.checkpoint_interval == 0) {
            saveBestGenome(config.log_dir + "best_gen_" + std::to_string(generation) + ".bin");
        }

        // 7. Select + reproduce (skip after last generation)
        if (generation < config.num_generations - 1) {
            select();
        }
    }

    // Save final best
    saveBestGenome(config.log_dir + "best_final.bin");
}

// =========================================================
// Get best individual
// =========================================================
const Individual& GeneticTrainer::getBest() const
{
    return *std::max_element(population.begin(), population.end(),
                             [](const Individual& a, const Individual& b) {
                                 return a.fitness < b.fitness;
                             });
}

// =========================================================
// Logging
// =========================================================
void GeneticTrainer::logGeneration()
{
    std::sort(population.begin(), population.end(),
              [](const Individual& a, const Individual& b) {
                  return a.fitness > b.fitness;
              });

    float avg = 0.0f;
    for (const auto& ind : population) avg += ind.fitness;
    avg /= static_cast<float>(population.size());

    std::ofstream log(config.log_dir + "training_log.csv", std::ios::app);
    if (generation == 0) {
        log << "generation,best_fitness,avg_fitness,worst_fitness,mutation_rate,mutation_strength,"
            << "best_vs_checkfold,best_vs_call,best_vs_raise,best_vs_smart\n";
    }
    log << generation << ","
        << population[0].fitness << ","
        << avg << ","
        << population.back().fitness << ","
        << getMutationRate() << ","
        << getMutationStrength();
    for (int opp = 0; opp < OPP_COUNT; opp++) {
        log << "," << population[0].earnings[opp];
    }
    log << "\n";
    log.close();
}

// =========================================================
// Save/Load
// =========================================================
void GeneticTrainer::saveBestGenome(const std::string& path) const
{
    auto best = std::max_element(population.begin(), population.end(),
                                 [](const Individual& a, const Individual& b) {
                                     return a.fitness < b.fitness;
                                 });

    std::ofstream file(path, std::ios::binary);
    int size = static_cast<int>(best->genome.size());
    file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    file.write(reinterpret_cast<const char*>(best->genome.data()), size * sizeof(float));
    file.close();
}

void GeneticTrainer::savePopulation(const std::string& path) const
{
    std::ofstream file(path, std::ios::binary);
    int pop_size = static_cast<int>(population.size());
    int g_size = genome_size;
    file.write(reinterpret_cast<const char*>(&pop_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&g_size), sizeof(int));
    for (const auto& ind : population) {
        file.write(reinterpret_cast<const char*>(ind.genome.data()), g_size * sizeof(float));
        file.write(reinterpret_cast<const char*>(&ind.fitness), sizeof(float));
    }
    file.close();
}

void GeneticTrainer::loadPopulation(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Could not load population from: " << path << std::endl;
        return;
    }

    int pop_size, g_size;
    file.read(reinterpret_cast<char*>(&pop_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&g_size), sizeof(int));

    if (g_size != genome_size) {
        std::cerr << "Genome size mismatch: file has " << g_size
                  << ", expected " << genome_size << std::endl;
        return;
    }

    population.resize(pop_size);
    for (auto& ind : population) {
        ind.genome.resize(g_size);
        file.read(reinterpret_cast<char*>(ind.genome.data()), g_size * sizeof(float));
        file.read(reinterpret_cast<char*>(&ind.fitness), sizeof(float));
        ind.earnings.fill(0.0f);
    }
    file.close();

    config.population_size = pop_size;
    std::cout << "Loaded population of " << pop_size << " from " << path << std::endl;
}
