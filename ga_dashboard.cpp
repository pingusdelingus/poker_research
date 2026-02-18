#include "ga_dashboard.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// =========================================================
// Constructor
// =========================================================
GADashboard::GADashboard()
    : total_generations(0), population_size(0), genome_size(0)
    , hands_per_session(0), total_sessions_per_gen(0)
    , generation(0), mutation_rate(0), mutation_strength(0)
    , sessions_completed(0), current_individual(0)
    , current_opponent(OPP_CHECKFOLD), current_seat(1)
    , best_fitness(0), avg_fitness(0), worst_fitness(0)
    , num_elites(0), num_survivors(0)
{
    best_per_opponent.fill(0.0f);
    train_start = std::chrono::steady_clock::now();
    gen_start = train_start;
    session_start = train_start;
}

// =========================================================
// Init (called once)
// =========================================================
void GADashboard::init(int total_gen, int pop_size, int g_size, int hands)
{
    total_generations = total_gen;
    population_size = pop_size;
    genome_size = g_size;
    hands_per_session = hands;
    total_sessions_per_gen = pop_size * OPP_COUNT * 2; // 2 seats per opponent

    train_start = std::chrono::steady_clock::now();

    // Clear screen once
    std::cout << "\033[2J\033[H" << std::flush;
}

// =========================================================
// Begin/End generation
// =========================================================
void GADashboard::beginGeneration(int gen, float mut_rate, float mut_strength)
{
    generation = gen;
    mutation_rate = mut_rate;
    mutation_strength = mut_strength;
    sessions_completed = 0;
    current_individual = 0;
    current_opponent = OPP_CHECKFOLD;
    current_seat = 1;

    // Reset per-generation stats
    observer = ObserverStatKeeper();
    for (auto& rec : opponent_records) {
        rec = OpponentRecord();
    }

    gen_start = std::chrono::steady_clock::now();
}

// =========================================================
// Session tracking
// =========================================================
void GADashboard::beginSession(int ind_idx, OpponentType opp, int seat)
{
    current_individual = ind_idx;
    current_opponent = opp;
    current_seat = seat;
    session_start = std::chrono::steady_clock::now();
}

void GADashboard::endSession(float earnings, bool evolved_won)
{
    auto& rec = opponent_records[current_opponent];
    rec.total_sessions++;
    rec.total_earnings += earnings;
    if (evolved_won) rec.evolved_wins++;
    else rec.opponent_wins++;

    sessions_completed++;
}

ObserverStatKeeper& GADashboard::getObserver()
{
    return observer;
}

// =========================================================
// Population results (called after evaluation)
// =========================================================
void GADashboard::setPopulationResults(const std::vector<Individual>& sorted_pop,
                                        float avg_fit,
                                        int elites, int survivors)
{
    best_fitness = sorted_pop.front().fitness;
    worst_fitness = sorted_pop.back().fitness;
    avg_fitness = avg_fit;
    num_elites = elites;
    num_survivors = survivors;

    for (int opp = 0; opp < OPP_COUNT; opp++) {
        best_per_opponent[opp] = sorted_pop.front().earnings[opp];
    }

    // Save to history
    GenSnapshot snap;
    snap.best_fitness = best_fitness;
    snap.avg_fitness = avg_fitness;
    snap.worst_fitness = worst_fitness;
    for (int opp = 0; opp < OPP_COUNT; opp++) {
        snap.per_opponent[opp] = best_per_opponent[opp];
    }
    history.push_back(snap);
}

// =========================================================
// Static helpers
// =========================================================
const char* GADashboard::opponentName(OpponentType opp)
{
    switch (opp) {
        case OPP_CHECKFOLD: return "Scared Limper (CheckFold)";
        case OPP_CALL:      return "Calling Machine (Call)";
        case OPP_RAISE:     return "Hothead Maniac (Raise)";
        case OPP_SMART:     return "Candid Statistician (Smart)";
        default:            return "Unknown";
    }
}

std::string GADashboard::formatTime(double seconds)
{
    int h = static_cast<int>(seconds / 3600);
    int m = static_cast<int>(std::fmod(seconds, 3600) / 60);
    int s = static_cast<int>(std::fmod(seconds, 60));

    std::stringstream ss;
    if (h > 0) ss << h << "h ";
    if (m > 0 || h > 0) ss << m << "m ";
    ss << s << "s";
    return ss.str();
}

std::string GADashboard::formatPct(double val)
{
    if (std::isnan(val) || std::isinf(val)) return "  --  ";
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << (val * 100.0) << "%";
    return ss.str();
}

std::string GADashboard::formatFloat(float val, int precision, bool show_sign)
{
    std::stringstream ss;
    if (show_sign) ss << std::showpos;
    ss << std::fixed << std::setprecision(precision) << val;
    if (show_sign) ss << std::noshowpos;
    return ss.str();
}

std::string GADashboard::makeProgressBar(int current, int total, int width)
{
    double ratio = (total > 0) ? static_cast<double>(current) / total : 0.0;
    int filled = static_cast<int>(ratio * width);
    std::string bar = "[";
    for (int i = 0; i < width; i++) {
        bar += (i < filled) ? "#" : "-";
    }
    bar += "]";
    return bar;
}

std::string GADashboard::makeSparkline(const std::vector<float>& values, int max_width)
{
    if (values.empty()) return "";

    // Use last max_width values
    int start = std::max(0, static_cast<int>(values.size()) - max_width);
    std::vector<float> window(values.begin() + start, values.end());

    float min_val = *std::min_element(window.begin(), window.end());
    float max_val = *std::max_element(window.begin(), window.end());
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;

    const char* blocks[] = {"\u2581", "\u2582", "\u2583", "\u2584",
                            "\u2585", "\u2586", "\u2587", "\u2588"};

    std::string result;
    for (float v : window) {
        int level = static_cast<int>(((v - min_val) / range) * 7.0f);
        level = std::max(0, std::min(7, level));
        result += blocks[level];
    }
    return result;
}

// =========================================================
// Main render
// =========================================================
void GADashboard::render()
{
    auto now = std::chrono::steady_clock::now();
    double total_secs = std::chrono::duration<double>(now - train_start).count();
    double gen_secs = std::chrono::duration<double>(now - gen_start).count();

    // ETA for current generation
    double sessions_per_sec = (gen_secs > 0 && sessions_completed > 0)
        ? sessions_completed / gen_secs : 0;
    int remaining_sessions = total_sessions_per_gen - sessions_completed;
    double gen_eta = (sessions_per_sec > 0) ? remaining_sessions / sessions_per_sec : 0;

    // ETA for entire training
    double avg_gen_time = (generation > 0) ? total_secs / generation : gen_secs;
    int remaining_gens = total_generations - generation - 1;
    double total_eta = remaining_gens * avg_gen_time + gen_eta;

    std::stringstream ss;
    ss << "\033[H"; // cursor home

    // ── Header ──
    ss << "\033[1m"; // bold
    ss << "  ================================================================\n";
    ss << "            OOPoker Neuroevolution Training Dashboard\n";
    ss << "  ================================================================\n";
    ss << "\033[0m"; // reset
    ss << "\n";

    // ── Generation info ──
    ss << "  Generation: \033[1m" << generation + 1 << " / " << total_generations << "\033[0m"
       << "          Total Time: " << formatTime(total_secs) << "\n";

    ss << "  Population: " << population_size;
    if (num_elites > 0 || num_survivors > 0) {
        ss << " (" << num_elites << " elites, " << num_survivors << " survivors)";
    }
    ss << "     Genome: " << genome_size << " params\n";

    ss << "  Gen Time: " << std::fixed << std::setprecision(1) << gen_secs << "s"
       << "   |   Mutation: rate=" << std::setprecision(3) << mutation_rate
       << "  str=" << mutation_strength << "\n";
    ss << "\n";

    // ── Current Generation Progress ──
    bool evaluating = (sessions_completed < total_sessions_per_gen);
    ss << "  \033[4m" << "Current Generation Progress" << "\033[0m\n";

    if (evaluating) {
        ss << "  Evaluating: Individual " << current_individual + 1 << " / " << population_size
           << "    vs " << opponentName(current_opponent)
           << " (seat " << current_seat << "/2)\n";
    } else {
        ss << "  Evaluation complete (" << total_sessions_per_gen << " sessions)\n";
    }

    float gen_pct = static_cast<float>(sessions_completed) / static_cast<float>(total_sessions_per_gen);
    ss << "  " << makeProgressBar(sessions_completed, total_sessions_per_gen, 50)
       << " " << std::fixed << std::setprecision(1) << (gen_pct * 100.0f) << "%";
    if (evaluating && gen_eta > 0) {
        ss << "    ETA: " << formatTime(gen_eta);
    }
    ss << "\n\n";

    // ── Fitness ──
    ss << "  \033[4m" << "Fitness (ANE)" << "\033[0m\n";
    ss << "  Best:  " << formatFloat(best_fitness, 4, true)
       << "    Avg:  " << formatFloat(avg_fitness, 4, true)
       << "    Worst: " << formatFloat(worst_fitness, 4, true) << "\n";
    ss << "\n";

    // ── Opponent Table ──
    ss << "  \033[4m" << "Best Player vs Opponents" << "\033[0m\n";
    ss << "  +----------------------------+----------+--------+-------+----------+\n";
    ss << "  | Opponent                    | Earnings |  W / L | Win%  | Trend    |\n";
    ss << "  +----------------------------+----------+--------+-------+----------+\n";

    for (int opp = 0; opp < OPP_COUNT; opp++) {
        const auto& rec = opponent_records[opp];
        ss << "  | " << std::left << std::setw(27) << opponentName(static_cast<OpponentType>(opp)) << " | ";

        // Earnings (from best individual's normalized earnings)
        ss << std::right << std::setw(8)
           << formatFloat(best_per_opponent[opp], 4, true) << " | ";

        // W/L for this generation (across all individuals)
        ss << std::setw(3) << rec.evolved_wins << "/" << std::left << std::setw(3) << rec.opponent_wins << " | ";

        // Win%
        float win_pct = (rec.total_sessions > 0)
            ? 100.0f * rec.evolved_wins / rec.total_sessions : 0.0f;
        ss << std::right << std::setw(4) << std::fixed << std::setprecision(0) << win_pct << "% | ";

        // Sparkline of best individual's per-opponent earnings across generations
        if (!history.empty()) {
            std::vector<float> trend;
            for (const auto& snap : history) {
                trend.push_back(snap.per_opponent[opp]);
            }
            std::string spark = makeSparkline(trend, 8);
            ss << std::left << std::setw(8) << spark;
        } else {
            ss << "        ";
        }
        ss << " |\n";
    }
    ss << "  +----------------------------+----------+--------+-------+----------+\n";
    ss << "\n";

    // ── Play Style ──
    const PlayerStats* evolved_stats = observer.getStatKeeper().getPlayerStats("Evolved");
    ss << "  \033[4m" << "Evolved Agent Play Style (This Gen)" << "\033[0m\n";
    ss << "  +--------+--------+--------+--------+--------+--------+\n";
    ss << "  |  VPIP  |  PFR   |   AF   |  WSD   |  WSDW  | Deals  |\n";
    ss << "  +--------+--------+--------+--------+--------+--------+\n";

    if (evolved_stats && evolved_stats->deals > 0) {
        ss << "  | " << std::right << std::setw(5) << formatPct(evolved_stats->getVPIP())
           << " | " << std::setw(5) << formatPct(evolved_stats->getPFR())
           << " | ";

        double af = evolved_stats->getAF();
        if (std::isnan(af) || std::isinf(af))
            ss << std::setw(6) << "--";
        else
            ss << std::setw(6) << std::fixed << std::setprecision(2) << af;

        ss << " | " << std::setw(5) << formatPct(evolved_stats->getWSD())
           << " | " << std::setw(5) << formatPct(evolved_stats->getWSDW())
           << " | " << std::setw(6) << evolved_stats->deals
           << " |\n";
    } else {
        ss << "  |   --   |   --   |   --   |   --   |   --   |   --   |\n";
    }
    ss << "  +--------+--------+--------+--------+--------+--------+\n";

    // Action breakdown
    if (evolved_stats && evolved_stats->actions > 0) {
        int total = evolved_stats->actions;
        ss << "  Actions: "
           << "fold " << std::fixed << std::setprecision(0)
           << (100.0 * evolved_stats->folds / total) << "%"
           << "  check " << (100.0 * evolved_stats->checks / total) << "%"
           << "  call " << (100.0 * evolved_stats->calls / total) << "%"
           << "  raise " << (100.0 * (evolved_stats->bets + evolved_stats->raises) / total) << "%"
           << "  allin " << (100.0 * evolved_stats->allins / total) << "%\n";
    }
    ss << "\n";

    // ── Fitness Trend ──
    if (history.size() >= 2) {
        ss << "  \033[4m" << "Fitness Trend" << "\033[0m" << " (last "
           << std::min(static_cast<int>(history.size()), 30) << " gens)\n";

        std::vector<float> best_trend, avg_trend;
        for (const auto& snap : history) {
            best_trend.push_back(snap.best_fitness);
            avg_trend.push_back(snap.avg_fitness);
        }
        ss << "  Best: " << makeSparkline(best_trend, 30) << "\n";
        ss << "  Avg:  " << makeSparkline(avg_trend, 30) << "\n";
        ss << "\n";
    }

    // ── Overall Progress ──
    ss << "  \033[4m" << "Overall Progress" << "\033[0m\n";
    float overall_pct = static_cast<float>(generation + 1) / static_cast<float>(total_generations);
    ss << "  " << makeProgressBar(generation + 1, total_generations, 50)
       << " " << std::fixed << std::setprecision(1) << (overall_pct * 100.0f) << "%";
    if (total_eta > 0 && generation < total_generations - 1) {
        ss << "    ETA: " << formatTime(total_eta);
    }
    ss << "\n\n";

    ss << "  Press Ctrl+C to stop training.\n";

    // Clean output: add \033[K (clear to EOL) before each newline
    std::string output = ss.str();
    std::string cleaned;
    cleaned.reserve(output.size() + 500);
    for (char ch : output) {
        if (ch == '\n') cleaned += "\033[K\n";
        else cleaned += ch;
    }
    cleaned += "\033[J"; // clear to end of screen
    std::cout << cleaned << std::flush;
}
