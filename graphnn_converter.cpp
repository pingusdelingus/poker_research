#include "converter.h"
#include <torch/torch.h>
#include <vector>
#include "poker_net.h"
// structures for the betting history linked list
/*
 *   DEFINED IN converter.h
 *
 * struct ActionNode
{
  int command;
  float amount_norm;
  float pot_norm;
  int player_index;
  ActionNode* next;
}; // end of actionnode
*/


// helper to push action to history using your arena
/* ActionNode* push_history_node(Arena* a, Action act, const Info& info)
{
  ActionNode* node = (ActionNode*)ArenaPush(a, sizeof(ActionNode));
  node->command = (int)act.command;
  node->amount_norm = (float)act.amount / (float)info.getBigBlind();
  node->pot_norm = (float)info.getPot() / (float)info.getBigBlind();
  node->player_index = info.current;
  node->next = nullptr;
  return node;
} // end of push_history_node
*/

/*
class GraphNNConverter
{
public:
  // converts the linked list history into a sequence tensor [seq_len, 1, features]
  static torch::Tensor history_to_tensor(ActionNode* head)
  {
    std::vector<float> sequence_data;
    int seq_len = 0;

    ActionNode* curr = head;
    while (curr != nullptr)
    {
      // feature vector for each action node: [command, amount, pot, player_pos]
      sequence_data.push_back((float)curr->command / 3.0f);
      sequence_data.push_back(std::min(curr->amount_norm / 100.0f, 1.0f));
      sequence_data.push_back(std::min(curr->pot_norm / 500.0f, 1.0f));
      sequence_data.push_back((float)curr->player_index / 9.0f);
      
      seq_len++;
      curr = curr->next;
    }

    if (seq_len == 0)
    {
      return torch::zeros({1, 1, 4});
    }

    // create tensor and reshape to [seq_len, 1, 4] for torch lstm
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor t = torch::from_blob(sequence_data.data(), {seq_len, 4}, options).clone();
    return t.unsqueeze(1); 
  } // end of history_to_tensor

  // combines static state (cards/stack) with the dynamic history
  static void forward_pass(PokerNet& net, const Info& info, ActionNode* history, torch::Tensor opp_stats)
  {
    //  get static features (cards, position, current stack)
    torch::Tensor static_state = TensorConverter::infoToTensor(info);

    //  get dynamic history tensor from linked list
    torch::Tensor history_seq = history_to_tensor(history);

    //  run rnn over history to get "context" vector
    // net->rnn will process history_seq and return the final hidden state
    // this final state represents the "vibe" of the betting round
    auto output = net->forward_with_history(static_state, history_seq, opp_stats);
  } // end of forward_pass
}; // end of graphnnconverter
*/
