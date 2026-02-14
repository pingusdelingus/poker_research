#include "ai.h"
#include "poker_net.h"
#include <cmath>
#include "info.h"
class AIRL : public AI {
private:
    PokerNet model;
    torch::Tensor rnn_hidden_state; 
    torch::Tensor opponent_stats; // Persistent memory of opponent tendencies

public:
    AIRL() : model(PokerNet(50, 128)) {
        // Initialize stats (e.g., VPIP 50%, Aggression 50% initially)
        opponent_stats = torch::full({1, 10}, 0.5); 
    }

    std::string getAIName() override { return "AlphaVector"; }

    // Convert game info to Tensor
    torch::Tensor stateToTensor(const Info& info) {
        // You must implement a parser that converts cards/pot/position to floats
        // Example: [Card1_Rank, Card1_Suit, Pot_Size, Position, ...]
        std::vector<float> data = { /* ... extract features from info ... */ };
        return torch::from_blob(data.data(), {1, (long)data.size()}).clone();
    }

    Action doTurn(const Info& info) override {
        // 1. Get Network Output
        torch::Tensor input = stateToTensor(info);
        torch::NoGradGuard no_grad; // No training during gameplay
        
        // Pass empty tensors for simplicity in this snippet, but you'd manage hidden states here
        torch::Tensor output = model->forward(input, rnn_hidden_state, opponent_stats);
        
        // 2. Decode the Vector (Your Logic)
        float x = output[0][0].item<float>();
        float y = output[0][1].item<float>();

        // Calculate Angle (radians) and Magnitude
        float angle = std::atan2(y, x); // Result between -PI and PI
        float magnitude = std::sqrt(x*x + y*y);

        // 3. Map Geometry to Poker Actions
        // Imagine a unit circle:
        // Top Sector (45 to 135 deg) -> Fold (Straight Up)
        // Left Sector (135 to -135 deg) -> Call (Left)
        // Right Sector (-45 to 45 deg) -> Raise (Right)
        
        double pi = 3.14159;
        
        if (angle > pi/4 && angle < 3*pi/4) {
            return info.getCheckFoldAction(); // "Straight Up"
        } 
        else if (std::abs(angle) > 3*pi/4) {
            return info.getCallAction(); // "Left"
        } 
        else {
            // "Right" -> Raise
            // Use magnitude to determine size (sigmoid to map 0-infinity to 0-1)
            double raise_percentage = 1.0 / (1.0 + std::exp(-magnitude)); 
            
            int min_r = info.getMinChipsToRaise();
            int max_r = info.getStack();
            int amount = min_r + (max_r - min_r) * raise_percentage;
            
            return info.amountToAction(amount);
        }
    }

    // CRITICAL: Update opponent model based on what happened
    void onEvent(const Event& event) override {
        // If opponent showed weakness or strength, update 'opponent_stats' tensor.
        // Also step the LSTM forward so it remembers the flow of the hand.
    }
};
