#include <torch/torch.h>
#include <iostream>
#include "poker_net.h"
#include "ai_smart.h"
#include "game.h"
#include "host_terminal.h" // Required for the Host
#include "player.h"        // Required for Player struct
#include "converter.h"

// A special AI that wraps the Expert (AISmart)
// It asks the expert what to do, trains the Neural Net on that decision,
// and then executes the action.
class AITrainer : public AI {
private:
    AISmart expert;       // The Teacher
    PokerNet& net;        // The Student (Reference to net in main)
    torch::optim::Optimizer& optimizer;
    std::string name;

    // Dummy hidden states for LSTM (reset every turn for now for simplicity)
    torch::Tensor h, c;

public:
    AITrainer(PokerNet& n, torch::optim::Optimizer& opt, std::string name) 
        : expert(0.5), net(n), optimizer(opt), name(name) 
    {
        // Initialize hidden states
        h = torch::zeros({1, 1, 128});
        c = torch::zeros({1, 1, 128});
    }

    std::string getAIName() override { return "TrainerBot"; }

    // This function is called by Game automatically when it's our turn
    Action doTurn(const Info& info) override {
        // 1. Ask the Expert what to do
        Action expertAction = expert.doTurn(info);

        // 2. Prepare Data for the Network
        torch::Tensor input_state = TensorConverter::infoToTensor(info);
        torch::Tensor target_vector = TensorConverter::actionToTarget(expertAction, info);

        // 3. Train the Network (Imitation Learning Step)
        net->train();
        optimizer.zero_grad();
        
        // Pass dummy Opponent Stats (zeros) for now
        torch::Tensor opp_stats = torch::zeros({1, 10}); 
        
        // Forward pass
        torch::Tensor output_vector = net->forward(input_state, h, opp_stats);

        // Calculate Loss (MSE between Bot's vector and Expert's vector)
        torch::Tensor loss = torch::mse_loss(output_vector, target_vector);

        // Backward pass
        loss.backward();
        optimizer.step();

        // Optional: Print loss occasionally
        // std::cout << "Loss: " << loss.item<float>() << std::endl;

        // 4. Return the Expert's action to the game so play continues legally
        return expertAction;
    }

    // Required boilerplate
    void onEvent(const Event& event) override { expert.onEvent(event); }
    bool boastCards(const Info& info) override { return expert.boastCards(info); }
    bool wantsToLeave(const Info& info) override { return false; } // Never leave, keep training
};


/*
 *
 * not using anymore 
int main() {
    std::cout << "Initializing Neural Network..." << std::endl;
    PokerNet net(TensorConverter::INPUT_SIZE, 128);
    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-3));

    // 2. Setup Game Environment
    // The Game class requires a 'Host' to handle output/logging
    HostTerminal host; 
    Game game(&host); // Pass address of host

    // 3. Create our Training Agents
    // We pass the Net and Optimizer by reference so they share the same brain
    AITrainer bot1(net, optimizer, "Bot_1");
    AITrainer bot2(net, optimizer, "Bot_2");

    // 4. Add Players to the Game
    // OOPoker requires wrapping AI in a 'Player' object
    Player p1(&bot1, "Trainer_1");
    Player p2(&bot2, "Trainer_2");
    
    // Give them chips (default stack usually handled by rules, but we initialize player)
    p1.stack = 1000;
    p2.stack = 1000;

    game.addPlayer(p1);
    game.addPlayer(p2);

    // 5. Configure Rules
    Rules rules;
    rules.smallBlind = 10;
    rules.bigBlind = 20;
    game.setRules(rules);

    // 6. Run the Training Loop
    std::cout << "Starting Training Loop (Press Ctrl+C to stop)..." << std::endl;
    
    // doGame() runs until the game ends (players bust or quit).
    // To train for many episodes, we loop calls to doGame
    for(int i = 0; i < 1000; ++i) {
        // Reset stacks for new game
        // Note: We might need to access players via the game or reset manually
        // Since Game copies players internally, we just run doGame repeatedly.
        // If players go bust, OOPoker might remove them.
        // For a simple test, let's just run it once.
        game.doGame();
        
        if (i % 10 == 0) {
            std::cout << " --- Game Batch " << i << " Finished ---" << std::endl;
            // Save checkpoint
            torch::save(net, "poker_bot_checkpoint.pt");
        }
    }

    return 0;
}

*/
