#pragma once
#include <torch/torch.h>
/*
struct PokerNetImpl : torch::nn::Module {
    // 1. Feature extraction layers
    torch::nn::Linear card_embedding{nullptr};
    torch::nn::Linear action_embedding{nullptr};
    torch::nn::LSTM rnn{nullptr};          // Processes the sequence of actions (history)
    torch::nn::Linear opponent_context{nullptr}; // Processes long-term opponent stats

    // 2. The "Vector" Head (Your custom geometry)
    torch::nn::Linear action_head{nullptr}; 

    PokerNetImpl(int input_size, int hidden_size) {
        // Embed cards/game state into a vector
        card_embedding = register_module("card_embed", torch::nn::Linear(input_size, 64));
        action_embedding = register_module("action_embed", torch::nn::Linear(3, 64));
        
        // LSTM takes (seq_len, batch, input_size)
        rnn = register_module("rnn", torch::nn::LSTM(torch::nn::LSTMOptions(64, hidden_size).num_layers(1)));
        
        // Compress opponent stats
        opponent_context = register_module("opp_ctx", torch::nn::Linear(10, 32)); // Assuming 10 stats

        // Output: 2 values (X, Y) for your vector regression
        action_head = register_module("action_head", torch::nn::Linear(hidden_size + 32, 2));
    }

    torch::Tensor forward(torch::Tensor game_state, torch::Tensor hidden_state, torch::Tensor opp_stats) {
        // 1. Process current game state
        auto x = torch::relu(card_embedding(game_state));
        
        // 2. Process history via LSTM
        // Note: In real impl, you manage the LSTM hidden tuple (h_n, c_n)
        auto rnn_out = rnn(x.unsqueeze(0)); 
        auto rnn_last_step = std::get<0>(rnn_out).squeeze(0);

        // 3. Process opponent context (Exploitative part)
        auto ctx = torch::relu(opponent_context(opp_stats));

        // 4. Combine (Concatenate)
        auto combined = torch::cat({rnn_last_step, ctx}, 1);

        // 5. Output Vector (x, y)
        return action_head(combined); 
    }
torch::Tensor forward_with_history(torch::Tensor static_feat, torch::Tensor history_seq, torch::Tensor opp_ctx)
{
  // encode static cards
  auto x_static = torch::relu(card_embedding(static_feat));
  auto x_history = torch::relu(action_embedding(history_seq));

  // process history sequence through rnn
  // rnn returns a tuple: {output, {h_n, c_n}}
  auto rnn_output = rnn(history_seq);
  auto last_hidden = std::get<0>(rnn_output)[-1]; // get final timestep

  auto x_opp = torch::relu(opponent_context(opp_ctx));
  auto combined = torch::cat({x_static, last_hidden, x_opp}, 1);

  return action_head(combined);
} // end of forward_with_history

};

*/
struct PokerNetImpl : torch::nn::Module {
    torch::nn::Linear card_embedding{nullptr};
    torch::nn::Linear action_embedding{nullptr}; // add this
    torch::nn::LSTM rnn{nullptr};
    torch::nn::Linear opponent_context{nullptr};
    torch::nn::Linear action_head{nullptr}; 

    PokerNetImpl(int input_size, int hidden_size) {
        card_embedding = register_module("card_embed", torch::nn::Linear(input_size, 64));
        
        // this maps our 3 history features to the 64 LSTM input features
        action_embedding = register_module("action_embed", torch::nn::Linear(3, 64));
        
        rnn = register_module("rnn", torch::nn::LSTM(torch::nn::LSTMOptions(64, hidden_size).num_layers(1)));
        
        opponent_context = register_module("opp_ctx", torch::nn::Linear(10, 32));
        action_head = register_module("action_head", torch::nn::Linear(hidden_size + 32 + 64, 2)); // update size: hidden + opp + static
    }

    torch::Tensor forward_with_history(torch::Tensor static_feat, torch::Tensor history_seq, torch::Tensor opp_ctx) {
        auto x_static = torch::relu(card_embedding(static_feat));

        // 1. embed the history sequence
        // history_seq is [seq_len, 1, 3] -> becomes [seq_len, 1, 64]
        auto x_history = torch::relu(action_embedding(history_seq));

        // 2. process through lstm
        auto rnn_output = rnn(x_history);
        auto last_hidden = std::get<0>(rnn_output)[-1]; 

        auto x_opp = torch::relu(opponent_context(opp_ctx));
        
        // concatenate static features, rnn history context, and opponent context
        auto combined = torch::cat({x_static, last_hidden, x_opp}, 1);

        return action_head(combined);
    }
};



TORCH_MODULE(PokerNet);
