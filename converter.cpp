#include "converter.h"
#include <cmath>
#include <algorithm>

// Helper to normalize money values by the Big Blind (makes training stable)
float normalize(int value, int bigBlind) {
    if (bigBlind == 0) return 0.0f;
    return (float)value / (float)bigBlind;
}

void TensorConverter::encodeCard(const Card& c, std::vector<float>& features) {
    // Assumption: Card.rank is 2-14 (Ace), Card.suit is 0-3
    features.push_back((c.value - 2.0f) / 12.0f); // Normalize Rank 0.0-1.0
    features.push_back(c.suit / 3.0f);           // Normalize Suit 0.0-1.0
}

void TensorConverter::encodeEmptyCard(std::vector<float>& features) {
    features.push_back(-1.0f);
    features.push_back(-1.0f);
}

torch::Tensor TensorConverter::infoToTensor(const Info& info) {
    std::vector<float> features;
    features.reserve(INPUT_SIZE);

    int bb = info.getBigBlind();

    // --- 1. Hole Cards (4 floats) ---
    const auto& hole = info.getHoleCards();
    if (hole.size() >= 2) {
        encodeCard(hole[0], features);
        encodeCard(hole[1], features);
    } else {
        // Should not happen in play, but safe fallback
        encodeEmptyCard(features);
        encodeEmptyCard(features);
    }

    // --- 2. Board Cards (10 floats) ---
    // We always fill 5 slots. If it's the Flop, the last 2 are empty (-1).
    for (int i = 0; i < 5; ++i) {
        if (i < info.boardCards.size()) {
            encodeCard(info.boardCards[i], features);
        } else {
            encodeEmptyCard(features);
        }
    }

    // --- 3. Game State (6 floats) ---
    // All money is normalized to Big Blinds so the bot learns relative strength
    features.push_back(normalize(info.getPot(), bb));
    features.push_back(normalize(info.getStack(), bb));         // My Stack
    features.push_back(normalize(info.getCallAmount(), bb));    // Cost to Call
    features.push_back(normalize(info.getWager(), bb));         // Already invested
    
    // Position: Normalized 0.0 (Dealer) to 1.0 (Last)
    float pos = (float)info.getPosition() / (float)std::max(1, info.getNumPlayers() - 1);
    features.push_back(pos);

    // Active Players: Normalized
    features.push_back((float)info.getNumActivePlayers() / 9.0f);

    // Convert to Torch Tensor [1, 20]
    return torch::from_blob(features.data(), {1, INPUT_SIZE}, torch::kFloat).clone();
}

torch::Tensor TensorConverter::actionToTarget(const Action& action, const Info& info) {
    float x = 0.0f, y = 0.0f;

    // --- Geometric Mapping ---
    // FOLD  = (0, 1)   [Up]
    // CALL  = (-1, 0)  [Left]
    // RAISE = (1, 0)   [Right]

    switch (action.type) { // Assuming Action has a 'type' enum member
        case A_FOLD:
            x = 0.0f; y = 1.0f; 
            break;
        case A_CHECK: // Treat check like a call (passive continue) or fold (weak)?
            // Usually Check is "Passive", let's map it to Call (Left) or a specific "Up-Left" zone.
            // For now, let's map Check to "Straight Left" (Call 0)
            x = -1.0f; y = 0.0f;
            break;
        case A_CALL:
            x = -1.0f; y = 0.0f;
            break;
        case A_RAISE:
        case A_ALL_IN: // All-in is just a huge raise
            x = 1.0f; y = 0.0f;
            
            // Advanced: You could scale 'y' based on raise size here 
            // to teach the bot sizing during imitation learning.
            // For now, we keep it simple (direction only).
            break;
    }

    float target[] = {x, y};
    return torch::from_blob(target, {1, 2}, torch::kFloat).clone();
}

Action TensorConverter::vectorToAction(const Info& info, float x, float y) {
    // 1. Calculate Angle and Magnitude
    float angle = std::atan2(y, x); // -PI to PI
    float magnitude = std::sqrt(x*x + y*y);
    double pi = 3.1415926535;

    // 2. Decode Zones
    // Fold Zone: Top wedge (45° to 135°)
    if (angle > pi/4 && angle < 3*pi/4) {
        return info.getCheckFoldAction();
    }
    
    // Call Zone: Left Side (abs(angle) > 135°)
    if (std::abs(angle) > 3*pi/4) {
        return info.getCallAction();
    }

    // Raise Zone: Right Side (-45° to 45°)
    // If we are here, we want to raise. But how much?
    
    // 3. Decode Magnitude to Bet Size
    // We map magnitude (0 to inf) to a percentage of the remaining stack.
    // Sigmoid function: 1 / (1 + e^-x) -> maps to 0.0 - 1.0
    double strength = 1.0 / (1.0 + std::exp(-magnitude)); 

    // However, we probably want the "base" raise to be at magnitude ~1.0.
    // Let's scale it so stronger vectors = bigger bets.
    
    int minR = info.getMinChipsToRaise();
    int maxR = info.getStack(); // All-in

    if (minR > maxR) return info.getAllInAction(); // Edge case

    // Lerp between MinRaise and All-In based on vector strength
    int amount = minR + (int)((maxR - minR) * strength);

    // Ensure we are legally raising
    return info.amountToAction(amount);
}
