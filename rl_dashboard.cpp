#include "rl_dashboard.h"
#include "event.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <fstream>

RLDashboard::RLDashboard()
    : total_epochs(0), hands_per_epoch(0)
    , epoch(0), hands_this_epoch(0)
    , agent_stack(0), opponent_stack(0)
    , loss_value(0), learning_rate(0), noise_scale(0)
    , total_wins(0), total_games(0), total_epochs_completed(0)
{
    train_start = std::chrono::steady_clock::now();
    epoch_start = train_start;
}

void RLDashboard::init(int total_ep, int hands)
{
    total_epochs = total_ep;
    hands_per_epoch = hands;
    std::cout << "\033[2J\033[H" << std::flush;
}

void RLDashboard::beginEpoch(int ep, float lr, float noise)
{
    epoch = ep;
    hands_this_epoch = 0;
    observer = ObserverStatKeeper();
    epoch_start = std::chrono::steady_clock::now();
    learning_rate = lr;
    noise_scale = noise;
    agent_stack = 0;
    opponent_stack = 0;
}

void RLDashboard::setPhase(const std::string& phase, const std::string& opponent)
{
    training_phase = phase;
    opponent_name = opponent;
}

void RLDashboard::onEvent(const Event& event)
{
    // Forward to stat keeper
    observer.onEvent(event);

    // Count hands and render periodically
    if (event.type == E_NEW_DEAL) {
        hands_this_epoch++;
        if (hands_this_epoch % 25 == 0) {
            render();
        }
    }
}

void RLDashboard::endEpoch(float a_stack, float o_stack, float loss, float lr, float noise, const std::vector<std::pair<int, float>>& saliency, int sub_wins, int sub_games)
{
    agent_stack = a_stack;
    opponent_stack = o_stack;
    loss_value = loss;
    learning_rate = lr;
    noise_scale = noise;
    current_saliency = saliency;

    total_wins += sub_wins;
    total_games += std::max(1, sub_games);
    total_epochs_completed++;

    EpochSnapshot snap;
    snap.agent_stack = a_stack;
    snap.opponent_stack = o_stack;
    snap.win_rate = (total_games > 0)
        ? static_cast<float>(total_wins) / total_games : 0.0f;
    history.push_back(snap);
    
    logMetrics();
}

void RLDashboard::logMetrics()
{
    std::ofstream log("./logs/rl/training_metrics.csv", std::ios::app);
    if (epoch == 0) {
        log << "epoch,agent_stack,opp_stack,net_chips,win_rate,loss,lr,noise,vpip,pfr,af,wsd,wsdw,deals\n";
    }

    const PlayerStats* agent_stats = observer.getStatKeeper().getPlayerStats("RL_Agent");
    
    float vpip = 0.0f, pfr = 0.0f, af = 0.0f, wsd = 0.0f, wsdw = 0.0f;
    int deals = 0;

    if (agent_stats && agent_stats->deals > 0) {
        vpip = agent_stats->getVPIP();
        pfr = agent_stats->getPFR();
        
        int postflop_aggr = (agent_stats->bets - agent_stats->preflop_bets)
                          + (agent_stats->raises - agent_stats->preflop_raises);
        int postflop_calls = std::max(1, agent_stats->calls - agent_stats->preflop_calls);
        af = static_cast<float>(postflop_aggr) / postflop_calls;

        wsd = agent_stats->getWSD();
        wsdw = agent_stats->getWSDW();
        deals = agent_stats->deals;
    }

    float net_chips = agent_stack - opponent_stack;
    float win_rate = history.empty() ? 0.0f : history.back().win_rate;

    log << epoch << ","
        << agent_stack << ","
        << opponent_stack << ","
        << net_chips << ","
        << win_rate << ","
        << loss_value << ","
        << learning_rate << ","
        << noise_scale << ","
        << vpip << ","
        << pfr << ","
        << af << ","
        << wsd << ","
        << wsdw << ","
        << deals << "\n";
    
    log.close();
}

void RLDashboard::addEvalResult(int ep, float stack)
{
    eval_history.push_back({ep, stack});
}

// =========================================================
// Helpers
// =========================================================
std::string RLDashboard::formatTime(double seconds)
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

std::string RLDashboard::formatFloat(float val, int precision, bool show_sign)
{
    std::stringstream ss;
    if (show_sign) ss << std::showpos;
    ss << std::fixed << std::setprecision(precision) << val;
    return ss.str();
}

std::string RLDashboard::makeProgressBar(int current, int total, int width)
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

std::string RLDashboard::makeSparkline(const std::vector<float>& values, int max_width)
{
    if (values.empty()) return "";

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
// Render
// =========================================================
void RLDashboard::render()
{
    auto now = std::chrono::steady_clock::now();
    double total_secs = std::chrono::duration<double>(now - train_start).count();
    double epoch_secs = std::chrono::duration<double>(now - epoch_start).count();

    double avg_epoch_time = (total_epochs_completed > 0)
        ? total_secs / total_epochs_completed : epoch_secs;
    int remaining = total_epochs - epoch - 1;
    double eta = remaining * avg_epoch_time;

    std::stringstream ss;
    ss << "\033[H"; // cursor home

    // Header
    ss << "\033[1m";
    ss << "  ================================================================\n";
    ss << "         OOPoker REINFORCE Training Dashboard (PokerNet)\n";
    ss << "  ================================================================\n";
    ss << "\033[0m\n";

    // Training Phase and Opponent
    ss << "  \033[4mTraining Status\033[0m\n";
    ss << "  Phase:    \033[1m" << std::left << std::setw(15) << training_phase << "\033[0m"
       << "     Opponent: \033[1m" << opponent_name << "\033[0m\n\n";

    // Epoch info
    ss << "  Epoch: \033[1m" << epoch + 1 << " / " << total_epochs << "\033[0m"
       << "          Total Time: " << formatTime(total_secs) << "\n";
    ss << "  Hand: " << hands_this_epoch << " / " << hands_per_epoch
       << "     Epoch Time: " << std::fixed << std::setprecision(1) << epoch_secs << "s"
       << "     Noise: " << std::setprecision(3) << noise_scale
       << "     LR: " << std::setprecision(6) << learning_rate << "\n\n";

    // Current epoch results
    ss << "  \033[4mCurrent Epoch Results\033[0m\n";
    ss << "  Agent Stack:    " << formatFloat(agent_stack, 0, false)
       << "     Opponent Stack: " << formatFloat(opponent_stack, 0, false) << "\n";

    float net_chips = agent_stack - opponent_stack;
    ss << "  Net Chips:      " << formatFloat(net_chips, 0, true);
    if (net_chips > 0) ss << " (W)";
    else if (net_chips < 0) ss << " (L)";
    else ss << " (D)";
    ss << "\n\n";

    // Win rate
    ss << "  \033[4mWin Rate\033[0m\n";
    float win_pct = (total_epochs_completed > 0)
        ? 100.0f * total_wins / total_epochs_completed : 0.0f;
    ss << "  Wins: " << total_wins << " / " << total_epochs_completed
       << "  (" << std::fixed << std::setprecision(1) << win_pct << "%)\n";

    // Win rate sparkline
    if (history.size() >= 2) {
        // Rolling win rate over last 50 epochs
        std::vector<float> rolling_wr;
        int window = 20;
        for (int i = 0; i < static_cast<int>(history.size()); i++) {
            int start_idx = std::max(0, i - window + 1);
            int wins = 0;
            int count = 0;
            for (int j = start_idx; j <= i; j++) {
                if (history[j].agent_stack > history[j].opponent_stack) wins++;
                count++;
            }
            rolling_wr.push_back(static_cast<float>(wins) / count);
        }
        ss << "  Trend (rolling " << window << "): " << makeSparkline(rolling_wr, 40) << "\n";
    }
    ss << "\n";

    // Agent stack trend
    if (history.size() >= 2) {
        ss << "  \033[4mAgent Stack Trend\033[0m (last "
           << std::min(static_cast<int>(history.size()), 40) << " epochs)\n";
        std::vector<float> stack_trend;
        for (const auto& snap : history) {
            stack_trend.push_back(snap.agent_stack);
        }
        ss << "  " << makeSparkline(stack_trend, 40) << "\n\n";
    }

    // Play style
    const PlayerStats* agent_stats = observer.getStatKeeper().getPlayerStats("RL_Agent");
    ss << "  \033[4mAgent Play Style (This Epoch)\033[0m\n";
    ss << "  +--------+--------+--------+--------+--------+--------+\n";
    ss << "  |  VPIP  |  PFR   |   AF   |  WSD   |  WSDW  | Deals  |\n";
    ss << "  +--------+--------+--------+--------+--------+--------+\n";

    if (agent_stats && agent_stats->deals > 0) {
        auto fmtPct = [](double val) -> std::string {
            if (std::isnan(val) || std::isinf(val)) return "  --  ";
            std::stringstream s;
            s << std::fixed << std::setprecision(1) << (val * 100.0) << "%";
            return s.str();
        };

        ss << "  | " << std::right << std::setw(5) << fmtPct(agent_stats->getVPIP())
           << " | " << std::setw(5) << fmtPct(agent_stats->getPFR())
           << " | ";

        // AF = (bets + raises) / calls, post-flop only
        // Use max(1, calls) to avoid div-by-zero when bot never calls post-flop
        int postflop_aggr = (agent_stats->bets - agent_stats->preflop_bets)
                          + (agent_stats->raises - agent_stats->preflop_raises);
        int postflop_calls = std::max(1, agent_stats->calls - agent_stats->preflop_calls);
        double af = static_cast<double>(postflop_aggr) / postflop_calls;
        ss << std::setw(6) << std::fixed << std::setprecision(2) << af;

        ss << " | " << std::setw(5) << fmtPct(agent_stats->getWSD())
           << " | " << std::setw(5) << fmtPct(agent_stats->getWSDW())
           << " | " << std::setw(6) << agent_stats->deals
           << " |\n";
    } else {
        ss << "  |   --   |   --   |   --   |   --   |   --   |   --   |\n";
    }
    ss << "  +--------+--------+--------+--------+--------+--------+\n";

    if (agent_stats && agent_stats->actions > 0) {
        double total = static_cast<double>(agent_stats->actions);
        auto fmtPct = [](double val, double total) -> std::string {
            if (total <= 0) return " 0.0%";
            std::stringstream s;
            s << std::fixed << std::setprecision(1) << std::right << std::setw(4) << (100.0 * val / total) << "%";
            return s.str();
        };

        ss << "  \033[4mAction Distribution (This Epoch)\033[0m\n";
        ss << "  Fold:  " << fmtPct(agent_stats->folds, total) << " " << makeProgressBar(agent_stats->folds, total, 20) << "\n";
        ss << "  Check: " << fmtPct(agent_stats->checks, total) << " " << makeProgressBar(agent_stats->checks, total, 20) << "\n";
        ss << "  Call:  " << fmtPct(agent_stats->calls, total) << " " << makeProgressBar(agent_stats->calls, total, 20) << "\n";
        ss << "  Raise: " << fmtPct(agent_stats->bets + agent_stats->raises, total) << " " << makeProgressBar(agent_stats->bets + agent_stats->raises, total, 20) << "\n";
    }
    ss << "\n";

    // Feature Importance (Saliency)
    if (!current_saliency.empty()) {
        static const std::vector<std::string> feature_names = {
            // 0-3: Hole cards
            "Hole1_Rank", "Hole1_Suit", "Hole2_Rank", "Hole2_Suit",
            // 4-13: Board cards
            "Board1_Rank", "Board1_Suit", "Board2_Rank", "Board2_Suit",
            "Board3_Rank", "Board3_Suit", "Board4_Rank", "Board4_Suit", "Board5_Rank", "Board5_Suit",
            // 14-22: Game state (M-Ratio removed)
            "Pot", "Stack", "Call_Amt", "Wager", "Position", "Equity", "PotOdds%", "ActivePl",
            // 22-23: Board derived features
            "BoardTexture", "OppRangeType",
            // 24-30: Opponent HandBand probabilities (×2 scaled)
            "OppHB_Air", "OppHB_DrawWk", "OppHB_DrawStr", "OppHB_MadeWk", "OppHB_MadeMd", "OppHB_MadeStr", "OppHB_Nuts",
            // 31-33: Live opponent stats
            "OppVPIP_Live", "OppPFR_Live", "Hist_Empty",
            // 34-37: VPIP history
            "OppVPIP_10", "OppVPIP_30", "OppVPIP_50", "OppVPIP_100",
            // 38-41: PFR history
            "OppPFR_10", "OppPFR_30", "OppPFR_50", "OppPFR_100",
            // 42-45: Donk history
            "OppDonk_10", "OppDonk_30", "OppDonk_50", "OppDonk_100"
        };

        ss << "  \033[4mTop Feature Importance (Saliency)\033[0m\n";
        for (int i = 0; i < std::min(10, (int)current_saliency.size()); ++i) {
            int idx = current_saliency[i].first;
            float val = current_saliency[i].second;
            std::string name = (idx < (int)feature_names.size()) ? feature_names[idx] : "Unknown";
            ss << "  " << std::left << std::setw(15) << name << ": " << std::fixed << std::setprecision(4) << val << "\n";
        }
        ss << "\n";
    }

    // Eval vs AISmart baseline
    if (!eval_history.empty()) {
        ss << "  \033[4mEvaluation vs AISmart(0.5) Baseline\033[0m\n";
        auto& latest = eval_history.back();
        float net_vs_baseline = latest.stack - 1000.0f;
        ss << "  Latest (epoch " << latest.epoch << "): "
           << formatFloat(latest.stack, 0, false) << " chips"
           << "  (net " << formatFloat(net_vs_baseline, 0, true) << ")\n";

        if (eval_history.size() >= 2) {
            std::vector<float> eval_trend;
            for (const auto& e : eval_history) {
                eval_trend.push_back(e.stack);
            }
            ss << "  Eval Trend: " << makeSparkline(eval_trend, 30) << "\n";
        }
        ss << "\n";
    }

    // Overall progress
    ss << "  \033[4mOverall Progress\033[0m\n";
    float pct = static_cast<float>(epoch + 1) / static_cast<float>(total_epochs);
    ss << "  " << makeProgressBar(epoch + 1, total_epochs, 50)
       << " " << std::fixed << std::setprecision(1) << (pct * 100.0f) << "%";
    if (eta > 0 && epoch < total_epochs - 1) {
        ss << "    ETA: " << formatTime(eta);
    }
    ss << "\n\n";
    ss << "  Press Ctrl+C to stop training.\n";

    // Clear trailing lines
    std::string output = ss.str();
    std::string cleaned;
    cleaned.reserve(output.size() + 500);
    for (char ch : output) {
        if (ch == '\n') cleaned += "\033[K\n";
        else cleaned += ch;
    }
    cleaned += "\033[J";
    std::cout << cleaned << std::flush;
}

void RLDashboard::writeFinalSummary(const std::string& output_path)
{
    auto now = std::chrono::steady_clock::now();
    double total_secs = std::chrono::duration<double>(now - train_start).count();

    static const std::vector<std::string> feature_names = {
        "Hole1_Rank", "Hole1_Suit", "Hole2_Rank", "Hole2_Suit",
        "Board1_Rank", "Board1_Suit", "Board2_Rank", "Board2_Suit",
        "Board3_Rank", "Board3_Suit", "Board4_Rank", "Board4_Suit", "Board5_Rank", "Board5_Suit",
        "Pot", "Stack", "Call_Amt", "Wager", "Position", "Equity", "PotOdds%", "ActivePl",
        "BoardTexture", "OppRangeType",
        "OppHB_Air", "OppHB_DrawWk", "OppHB_DrawStr", "OppHB_MadeWk", "OppHB_MadeMd", "OppHB_MadeStr", "OppHB_Nuts",
        "OppVPIP_Live", "OppPFR_Live", "Hist_Empty",
        "OppVPIP_10", "OppVPIP_30", "OppVPIP_50", "OppVPIP_100",
        "OppPFR_10", "OppPFR_30", "OppPFR_50", "OppPFR_100",
        "OppDonk_10", "OppDonk_30", "OppDonk_50", "OppDonk_100"
    };

    std::ofstream f(output_path);
    if (!f.is_open()) return;

    float win_pct = (total_epochs_completed > 0)
        ? 100.0f * total_wins / total_epochs_completed : 0.0f;
    float avg_epoch_time = (total_epochs_completed > 0) ? total_secs / total_epochs_completed : 0.0f;

    f << "================================================================\n";
    f << "       PokerNet Training - Final Summary\n";
    f << "================================================================\n\n";

    f << "Training completed.\n";
    f << "  Total epochs:    " << total_epochs_completed << " / " << total_epochs << "\n";
    f << "  Total time:      " << formatTime(total_secs) << "\n";
    f << "  Avg epoch time:  " << formatFloat(avg_epoch_time, 1, false) << "s\n\n";

    f << "Win Rate\n";
    f << "  Wins: " << total_wins << " / " << total_epochs_completed
      << "  (" << formatFloat(win_pct, 1, false) << "%)\n\n";

    f << "Last Epoch Results\n";
    f << "  Phase:           " << training_phase << "\n";
    f << "  Opponent:        " << opponent_name << "\n";
    f << "  Agent Stack:     " << formatFloat(agent_stack, 0, false) << "\n";
    f << "  Opponent Stack:  " << formatFloat(opponent_stack, 0, false) << "\n";
    float net = agent_stack - opponent_stack;
    f << "  Net Chips:       " << formatFloat(net, 0, true);
    if (net > 0)       f << " (W)";
    else if (net < 0)  f << " (L)";
    else               f << " (D)";
    f << "\n\n";

    const PlayerStats* agent_stats = observer.getStatKeeper().getPlayerStats("RL_Agent");
    if (agent_stats && agent_stats->deals > 0) {
        f << "Agent Play Style (Last Epoch)\n";
        auto fmtPct = [](double v) {
            std::stringstream s;
            s << std::fixed << std::setprecision(1) << (v * 100.0) << "%";
            return s.str();
        };
        int postflop_aggr = (agent_stats->bets - agent_stats->preflop_bets)
                          + (agent_stats->raises - agent_stats->preflop_raises);
        int postflop_calls = std::max(1, agent_stats->calls - agent_stats->preflop_calls);
        double af = static_cast<double>(postflop_aggr) / postflop_calls;
        double total_actions = static_cast<double>(agent_stats->actions);

        f << "  VPIP:   " << fmtPct(agent_stats->getVPIP()) << "\n";
        f << "  PFR:    " << fmtPct(agent_stats->getPFR()) << "\n";
        f << "  AF:     " << std::fixed << std::setprecision(2) << af << "\n";
        f << "  WSD:    " << fmtPct(agent_stats->getWSD()) << "\n";
        f << "  WSDW:   " << fmtPct(agent_stats->getWSDW()) << "\n";
        f << "  Deals:  " << agent_stats->deals << "\n\n";

        if (total_actions > 0) {
            f << "Action Distribution (Last Epoch)\n";
            f << "  Fold:  " << fmtPct((double)agent_stats->folds / total_actions) << "\n";
            f << "  Check: " << fmtPct((double)agent_stats->checks / total_actions) << "\n";
            f << "  Call:  " << fmtPct((double)agent_stats->calls / total_actions) << "\n";
            f << "  Raise: " << fmtPct((double)(agent_stats->bets + agent_stats->raises) / total_actions) << "\n\n";
        }
    }

    if (!current_saliency.empty()) {
        f << "Top Feature Importance (Saliency, normalized)\n";
        for (int i = 0; i < std::min(20, (int)current_saliency.size()); ++i) {
            int idx = current_saliency[i].first;
            float val = current_saliency[i].second;
            std::string name = (idx < (int)feature_names.size()) ? feature_names[idx] : "Unknown";
            f << "  " << std::left << std::setw(16) << name
              << ": " << std::fixed << std::setprecision(4) << val << "\n";
        }
        f << "\n";
    }

    if (!eval_history.empty()) {
        f << "Evaluation vs AISmart Baseline\n";
        f << "  Latest (epoch " << eval_history.back().epoch << "): "
          << formatFloat(eval_history.back().stack, 0, false) << " chips\n";
        f << "  Full history:\n";
        for (const auto& e : eval_history) {
            f << "    Epoch " << std::setw(5) << e.epoch
              << " -> " << formatFloat(e.stack, 0, false) << " chips\n";
        }
        f << "\n";
    }

    f << "================================================================\n";
    f.close();

    std::cout << "\n[Summary] Written to: " << output_path << "\n";
}
