#include "ai_evolved.h"
#include "event.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// =========================================================
// Helper: sigmoid and tanh
// =========================================================
static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static float tanh_f(float x)
{
    return std::tanh(x);
}

// =========================================================
// LSTMLayer
// =========================================================
LSTMLayer::LSTMLayer() : input_size(0), hidden_size(0) {}

LSTMLayer::LSTMLayer(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size)
{
    weights.resize(num_parameters(), 0.0f);
    h.resize(hidden_size, 0.0f);
    c.resize(hidden_size, 0.0f);
}

int LSTMLayer::num_parameters() const
{
    // 4 gates, each: W[in*hid] + R[hid*hid] + b[hid]
    return 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size);
}

void LSTMLayer::reset_state()
{
    std::fill(h.begin(), h.end(), 0.0f);
    std::fill(c.begin(), c.end(), 0.0f);
}

void LSTMLayer::set_weights(const float* src)
{
    std::memcpy(weights.data(), src, weights.size() * sizeof(float));
}

void LSTMLayer::get_weights(float* dst) const
{
    std::memcpy(dst, weights.data(), weights.size() * sizeof(float));
}

void LSTMLayer::forward(const std::vector<float>& input)
{
    // Weight layout for each gate g:
    //   W_g: [input_size * hidden_size] (input weights)
    //   R_g: [hidden_size * hidden_size] (recurrent weights)
    //   b_g: [hidden_size] (bias)
    // Gates order: input(i), forget(f), cell candidate(g), output(o)

    int gate_params = input_size * hidden_size + hidden_size * hidden_size + hidden_size;
    const float* w = weights.data();

    // Temporary storage for gate pre-activations
    std::vector<float> gate_i(hidden_size);
    std::vector<float> gate_f(hidden_size);
    std::vector<float> gate_g(hidden_size);
    std::vector<float> gate_o(hidden_size);

    float* gates[4] = { gate_i.data(), gate_f.data(), gate_g.data(), gate_o.data() };

    for (int g = 0; g < 4; g++) {
        const float* W_g = w + g * gate_params;
        const float* R_g = W_g + input_size * hidden_size;
        const float* b_g = R_g + hidden_size * hidden_size;

        for (int j = 0; j < hidden_size; j++) {
            float val = b_g[j];

            // W_g * input
            for (int k = 0; k < input_size; k++) {
                val += W_g[j * input_size + k] * input[k];
            }

            // R_g * h_prev
            for (int k = 0; k < hidden_size; k++) {
                val += R_g[j * hidden_size + k] * h[k];
            }

            gates[g][j] = val;
        }
    }

    // Apply activations and compute new c and h
    for (int j = 0; j < hidden_size; j++) {
        float i_gate = sigmoid(gate_i[j]);
        float f_gate = sigmoid(gate_f[j]);
        float g_gate = tanh_f(gate_g[j]);
        float o_gate = sigmoid(gate_o[j]);

        c[j] = f_gate * c[j] + i_gate * g_gate;
        h[j] = o_gate * tanh_f(c[j]);
    }
}

// =========================================================
// FCLayer
// =========================================================
FCLayer::FCLayer() : input_size(0), output_size(0), use_tanh(true) {}

FCLayer::FCLayer(int input_size, int output_size, bool use_tanh)
    : input_size(input_size), output_size(output_size), use_tanh(use_tanh)
{
    weights.resize(num_parameters(), 0.0f);
}

int FCLayer::num_parameters() const
{
    return input_size * output_size + output_size;
}

void FCLayer::set_weights(const float* src)
{
    std::memcpy(weights.data(), src, weights.size() * sizeof(float));
}

void FCLayer::get_weights(float* dst) const
{
    std::memcpy(dst, weights.data(), weights.size() * sizeof(float));
}

std::vector<float> FCLayer::forward(const std::vector<float>& input) const
{
    std::vector<float> output(output_size);
    const float* W = weights.data();
    const float* b = W + input_size * output_size;

    for (int j = 0; j < output_size; j++) {
        float val = b[j];
        for (int k = 0; k < input_size; k++) {
            val += W[j * input_size + k] * input[k];
        }
        output[j] = use_tanh ? tanh_f(val) : val;
    }

    return output;
}

// =========================================================
// EvolvedNet
// =========================================================
EvolvedNet::EvolvedNet()
    : game_lstm(8, 50)
    , opponent_lstm(8, 10)
    , decision_hidden(60, 32, true)
    , decision_output(32, 1, true)
{
}

float EvolvedNet::forward(const std::vector<float>& features)
{
    // Feed features to both LSTMs
    game_lstm.forward(features);
    opponent_lstm.forward(features);

    // Concatenate: game_h (50) + opp_h (10) = 60
    std::vector<float> combined(60);
    std::copy(game_lstm.h.begin(), game_lstm.h.end(), combined.begin());
    std::copy(opponent_lstm.h.begin(), opponent_lstm.h.end(), combined.begin() + 50);

    // Decision network
    auto hidden = decision_hidden.forward(combined);
    auto output = decision_output.forward(hidden);

    return output[0]; // scalar in [-1, 1]
}

void EvolvedNet::reset_game_state()
{
    game_lstm.reset_state();
}

void EvolvedNet::reset_opponent_state()
{
    opponent_lstm.reset_state();
}

int EvolvedNet::genome_size() const
{
    return game_lstm.num_parameters()
         + opponent_lstm.num_parameters()
         + decision_hidden.num_parameters()
         + decision_output.num_parameters();
}

std::vector<float> EvolvedNet::get_genome() const
{
    std::vector<float> genome(genome_size());
    float* dst = genome.data();

    game_lstm.get_weights(dst);
    dst += game_lstm.num_parameters();

    opponent_lstm.get_weights(dst);
    dst += opponent_lstm.num_parameters();

    decision_hidden.get_weights(dst);
    dst += decision_hidden.num_parameters();

    decision_output.get_weights(dst);

    return genome;
}

void EvolvedNet::set_genome(const std::vector<float>& genome)
{
    const float* src = genome.data();

    game_lstm.set_weights(src);
    src += game_lstm.num_parameters();

    opponent_lstm.set_weights(src);
    src += opponent_lstm.num_parameters();

    decision_hidden.set_weights(src);
    src += decision_hidden.num_parameters();

    decision_output.set_weights(src);
}

// =========================================================
// AIEvolved
// =========================================================
AIEvolved::AIEvolved()
    : starting_stack(0), big_blind(0), initialized(false)
{
}

AIEvolved::AIEvolved(const std::vector<float>& genome)
    : starting_stack(0), big_blind(0), initialized(false)
{
    net.set_genome(genome);
}

std::vector<float> AIEvolved::getGenome() const
{
    return net.get_genome();
}

void AIEvolved::setGenome(const std::vector<float>& genome)
{
    net.set_genome(genome);
}

int AIEvolved::genomeSize() const
{
    return net.genome_size();
}

void AIEvolved::resetForNewHand()
{
    net.reset_game_state();
}

void AIEvolved::resetForNewOpponent()
{
    net.reset_game_state();
    net.reset_opponent_state();
}

// --- Feature extraction (8 floats, paper Section 3.1) ---
std::vector<float> AIEvolved::extractFeatures(const Info& info) const
{
    std::vector<float> f(8, 0.0f);

    // [0-3] One-hot game stage
    if (info.round == R_PRE_FLOP)                         f[0] = 1.0f;
    else if (info.round == R_FLOP)                         f[1] = 1.0f;
    else if (info.round == R_TURN)                         f[2] = 1.0f;
    else /* R_RIVER or R_SHOWDOWN */                       f[3] = 1.0f;

    // [4] Win probability (pot equity)
    f[4] = static_cast<float>(info.getPotEquity());

    // [5] Player chips committed, normalized by starting stack
    float ss = static_cast<float>(starting_stack);
    f[5] = (ss > 0) ? static_cast<float>(info.getWager()) / ss : 0.0f;

    // [6] Opponent chips committed, normalized by starting stack
    int opp_idx = (info.yourIndex == 0) ? 1 : 0;
    f[6] = (ss > 0) ? static_cast<float>(info.getWager(opp_idx)) / ss : 0.0f;

    // [7] Pot odds percentage
    f[7] = static_cast<float>(info.getPotOddsPercentage());

    return f;
}

// --- Decision algorithm (paper Algorithm 1) ---
Action AIEvolved::scalarToAction(float o, const Info& info) const
{
    int call_amount = info.getCallAmount();
    int stack = info.getStack();

    // Negative output: fold or check
    if (o < 0.0f) {
        if (call_amount == 0) return Action(A_CHECK, 0);
        else return Action(A_FOLD, 0);
    }

    // Call threshold: o < call_amount / starting_stack
    float call_threshold = (starting_stack > 0)
        ? static_cast<float>(call_amount) / static_cast<float>(starting_stack)
        : 0.0f;

    if (o < call_threshold) {
        // Call (or all-in if can't afford)
        if (stack <= call_amount) return info.getAllInAction();
        return info.getCallAction();
    }

    // Raise zone: k = floor(o * starting_stack / big_blind)
    int k = static_cast<int>(std::floor(o * static_cast<float>(starting_stack) / static_cast<float>(big_blind)));
    int raise_amount = k * big_blind;

    // If raise is below minimum, just call
    if (raise_amount < info.minRaiseAmount) {
        return info.getCallAction();
    }

    // Total chips needed: call_amount + raise_amount
    int total_needed = call_amount + raise_amount;
    if (stack <= total_needed) {
        return info.getAllInAction();
    }

    return info.getRaiseAction(raise_amount);
}

// --- AI interface ---
Action AIEvolved::doTurn(const Info& info)
{
    // Initialize game parameters on first call
    if (!initialized) {
        starting_stack = info.rules.buyIn;
        big_blind = info.getBigBlind();
        initialized = true;
    }

    std::vector<float> features = extractFeatures(info);
    float o = net.forward(features);
    return scalarToAction(o, info);
}

void AIEvolved::onEvent(const Event& event)
{
    if (event.type == E_NEW_DEAL) {
        net.reset_game_state();
        // opponent LSTM is NOT reset — it persists across hands
    }
}

std::string AIEvolved::getAIName()
{
    return "Evolved";
}
