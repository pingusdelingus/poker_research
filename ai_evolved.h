#pragma once
#include "ai.h"
#include "info.h"
#include "action.h"
#include <vector>
#include <string>
#include <cmath>

// =========================================================
// Raw C++ LSTM Layer (no libtorch)
// Standard LSTM: 4 gates (input, forget, cell, output)
// Weight layout per gate: W[in*hid] + R[hid*hid] + b[hid]
// Total: 4 * (in*hid + hid*hid + hid)
// =========================================================
struct LSTMLayer {
    int input_size;
    int hidden_size;

    std::vector<float> weights; // all gate weights concatenated
    std::vector<float> h;      // hidden state [hidden_size]
    std::vector<float> c;      // cell state [hidden_size]

    LSTMLayer();
    LSTMLayer(int input_size, int hidden_size);

    void forward(const std::vector<float>& input);
    void reset_state();

    int num_parameters() const;
    void set_weights(const float* src);
    void get_weights(float* dst) const;
};

// =========================================================
// Fully Connected Layer with tanh activation
// =========================================================
struct FCLayer {
    int input_size;
    int output_size;
    bool use_tanh;
    std::vector<float> weights; // [input_size * output_size + output_size]

    FCLayer();
    FCLayer(int input_size, int output_size, bool use_tanh = true);

    std::vector<float> forward(const std::vector<float>& input) const;

    int num_parameters() const;
    void set_weights(const float* src);
    void get_weights(float* dst) const;
};

// =========================================================
// Dual-LSTM Poker Network (Li & Miikkulainen 2017)
// Game LSTM (8->50) + Opponent LSTM (8->10) + FC (60->32->1)
// =========================================================
struct EvolvedNet {
    LSTMLayer game_lstm;       // resets every hand
    LSTMLayer opponent_lstm;   // persists across hands within a session
    FCLayer decision_hidden;   // 60 -> 32, tanh
    FCLayer decision_output;   // 32 -> 1, tanh

    EvolvedNet();

    // Forward: takes 8-feature input, returns scalar output in [-1, 1]
    float forward(const std::vector<float>& features);

    void reset_game_state();
    void reset_opponent_state();

    int genome_size() const;
    std::vector<float> get_genome() const;
    void set_genome(const std::vector<float>& genome);
};

// =========================================================
// AIEvolved: AI interface for the evolved network
// =========================================================
class AIEvolved : public AI {
public:
    AIEvolved();
    AIEvolved(const std::vector<float>& genome);

    Action doTurn(const Info& info) override;
    void onEvent(const Event& event) override;
    std::string getAIName() override;

    // Genome access
    std::vector<float> getGenome() const;
    void setGenome(const std::vector<float>& genome);
    int genomeSize() const;

    // State management (called by trainer between matchups)
    void resetForNewHand();
    void resetForNewOpponent();

private:
    EvolvedNet net;
    int starting_stack;  // rules.buyIn
    int big_blind;       // rules.bigBlind
    bool initialized;    // set on first doTurn call

    // Feature extraction (8 floats, paper Section 3.1)
    std::vector<float> extractFeatures(const Info& info) const;

    // Decision algorithm (paper Algorithm 1)
    Action scalarToAction(float o, const Info& info) const;
};
