# PokerNet Model Card

> **Status**: Implemented. Based on Li & Miikkulainen (2017), "Evolving Adaptive LSTM Poker Players for Effective Opponent Exploitation."

## Reference

Li, X. and Miikkulainen, R. (2017). *Evolving Adaptive LSTM Poker Players for Effective Opponent Exploitation.* University of Texas at Austin. AAAI 2017. See `references/utaustin_lstm.pdf`.

---

## Input Features — 8 floats

Adapted from Section 3.1 of the reference paper. All features are normalized to [0, 1]. Raw card encodings are not used; hand strength is represented solely by win probability (pot equity). Chip amounts are normalized by starting stack (buy-in).

| Index | Feature | Formula / Source | Range |
|-------|---------|-----------------|-------|
| 0 | Preflop | 1 if `round == R_PRE_FLOP`, else 0 | {0, 1} |
| 1 | Flop | 1 if `round == R_FLOP`, else 0 | {0, 1} |
| 2 | Turn | 1 if `round == R_TURN`, else 0 | {0, 1} |
| 3 | River | 1 if `round == R_RIVER`, else 0 | {0, 1} |
| 4 | Win probability | `getPotEquity()` — Monte Carlo equity vs random hand | [0, 1] |
| 5 | Player chips committed | `getWager() / buy_in` | [0, 1] |
| 6 | Opponent chips committed | `getWager(opp_index) / buy_in` | [0, 1] |
| 7 | Pot odds | `getPotOddsPercentage()` = call / (call + pot) | [0, 1) |

### Rationale

The paper demonstrates that a compact feature set focusing on *strategic information* (equity, chip commitment, pot odds) outperforms richer feature sets that include raw card data. The network's job is to learn *strategy*, not hand evaluation — pot equity already encodes hand strength. Fewer features also reduce the search space for the genetic algorithm.

---

## Architecture — Dual-LSTM with Decision Network

Adapted from Section 3.1 and Figure 2 of the reference paper. Implemented in raw C++ (no libtorch) so the genetic algorithm can directly access weights as a flat `vector<float>` for crossover and mutation.

### Game Module (LSTM)
- **Purpose**: Extract features from the sequence of moves within a single hand
- **Structure**: Standard LSTM, input size 8, hidden size 50 (11,800 parameters)
- **Input**: 8-feature state vector, fed at each decision point within a hand
- **Reset**: Hidden and cell state cleared at the start of every new hand (`E_NEW_DEAL`)
- **Gates**: Input, forget, cell candidate, output — sigmoid/tanh activations
- **Weight layout per gate**: W[8×50] + R[50×50] + b[50]

### Opponent Module (LSTM)
- **Purpose**: Model the opponent by extracting patterns across the entire history of hands played
- **Structure**: Standard LSTM, input size 8, hidden size 10 (760 parameters)
- **Input**: Same 8-feature state vector
- **Reset**: Hidden state cleared only when facing a new opponent (persists across all hands in a session)
- **Gates**: Same as Game Module

### Decision Network (Feed-forward)
- **Purpose**: Combine game context and opponent model to select an action
- **Input**: Concatenated hidden states — game (50) + opponent (10) = 60
- **Hidden layer**: FC 60→32, tanh activation (1,952 parameters)
- **Output layer**: FC 32→1, tanh activation (33 parameters)
- **Output**: Single scalar `o ∈ [-1, 1]`

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Game LSTM (8→50) | 11,800 |
| Opponent LSTM (8→10) | 760 |
| FC hidden (60→32) | 1,952 |
| FC output (32→1) | 33 |
| **Total genome** | **14,545 floats (~57 KB)** |

### Architecture Diagram

```
                    8-feature input vector
                    (at each decision point)
                           |
                    +------+------+
                    |             |
              Game LSTM      Opponent LSTM
              (8→50,          (8→10,
               resets          persists
               each hand)      across hands)
                    |             |
                 [50]           [10]
                    |             |
                    +------+------+
                           |
                        [60] (concatenated)
                           |
                     FC 60→32 (tanh)
                           |
                     FC 32→1 (tanh)
                           |
                     scalar output o ∈ [-1, 1]
                           |
                     Decision Algorithm
                           |
                     FOLD / CHECK / CALL / RAISE
```

### Decision Algorithm (Algorithm 1 from paper)

```
Given: o (network output), x (call amount), ss (starting stack), BB (big blind), r_min (minimum raise)

if o < 0:
    if x == 0: CHECK
    else: FOLD
else if o < x / ss:
    CALL (or ALL-IN if stack ≤ call amount)
else:
    k = floor(o * ss / BB)
    RAISE by k * BB (or ALL-IN if stack ≤ total needed)
    if raise < r_min: CALL instead
```

The output `o` can be interpreted as the fraction of starting stack the player is willing to commit. Negative values indicate unwillingness to put money in.

---

## Training — Genetic Algorithm with TSER

Adapted from Section 3.2 and Table 1 of the reference paper. No gradients — fitness is the sole training signal.

### Overview

A population of 50 LSTM players is evolved using a genetic algorithm. Each player's genome is the flat concatenation of all network weights (14,545 floats). Fitness is evaluated by playing against 4 rule-based opponents, and selection uses **Tiered Survival and Elite Reproduction (TSER)**.

### Opponent Pool

The paper uses 4 rule-based opponents requiring *different and opposite* counter-strategies:

| Paper Opponent | Strategy | Counter-Strategy | Implementation |
|---------------|----------|-----------------|----------------|
| Scared Limper | Ultra-conservative, folds to any raise | Raise every hand | `AICheckFold` |
| Calling Machine | Calls everything, never raises | Never bluff, value bet strong hands | `AICall` |
| Hothead Maniac | Raises blindly every hand | Fold weak, trap with strong hands | `AIRaise` |
| Candid Statistician | Bets proportional to hand strength | Bluff when passive, fold when aggressive | `AISmart` |

### Fitness Evaluation (per generation)

Each individual plays against all 4 opponents:
- 2 sessions per opponent (swapped seats), 500 hands each
- Total: 8 sessions × 500 hands = 4,000 hands per individual per generation
- Stacks reset to buy-in at the start of each session
- No rebuys — games end on bust-out
- Game LSTM resets each hand; opponent LSTM resets each session

**Fitness function — Average Normalized Earnings (ANE)**:

```
f(i) = (1/m) * sum_j(e_ij / n_j)

where:
  e_ij = cumulative earnings of player i vs opponent j (sum of both sessions)
  n_j  = max(BB/buy_in, max_i(e_ij))  — normalization factor per opponent
  m    = number of opponents (4)
```

ANE prevents specialization: no matter how much a player earns against one opponent, that opponent can contribute at most 1/m to the fitness.

### Selection: Tiered Survival and Elite Reproduction (TSER)

1. **Evaluate**: All 50 players play against all 4 opponents (400 total sessions); compute ANE
2. **Survive**: Top 30% of population (15 players) survives to next generation
3. **Classify**: Survivors with ANE ≥ mean(survivor ANE) become **elites**; rest are **second tier**
4. **Reproduce**: Only elites reproduce via crossover between random elite pairs
5. **Mutate**: Children and second-tier survivors are mutated; elites are immune
6. **Fill**: New children fill remaining 35 population slots

### Crossover

Interleave odd/even genome indices from two elite parents:
```
child[i] = parent_a[i] if i is even, parent_b[i] if i is odd
```

### Mutation

Per-gene Gaussian noise with linearly decaying schedule:
- **Rate**: Probability each gene is mutated. 0.25 → 0.05 over 250 generations.
- **Strength**: Standard deviation of Gaussian noise. 0.50 → 0.10 over 250 generations.
- Uses Box-Muller transform for Gaussian sampling.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Generations | 250 |
| Population size | 50 |
| Survival rate | 0.30 |
| Mutation rate | 0.25 → 0.05 (linear decay) |
| Mutation strength (std dev) | 0.50 → 0.10 (linear decay) |
| Crossover method | Interleave odd/even indices |
| Hands per session | 500 |
| Sessions per opponent | 2 (swapped seats) |
| Starting stack (buy-in) | 1,000 |
| Big blind | 10 |
| Small blind | 5 |

### Training Loop

```
initialize population of 50 players (Gaussian random weights, mean=0, std=0.5)

for generation in 1..250:
    for each player in population:
        for each opponent in [CheckFold, Call, Raise, Smart]:
            session1 = play 500 hands (player=seat0, opponent=seat1)
            session2 = play 500 hands (player=seat1, opponent=seat0)
            earnings[opponent] = (finalStack1 - buyIn)/buyIn + (finalStack2 - buyIn)/buyIn

    compute ANE fitness for each player
    rank players by fitness

    survivors = top 30% of population
    elites = survivors with ANE ≥ mean(survivor ANE)
    second_tier = survivors - elites

    children = []
    while len(children) < 50 - len(survivors):
        parent1, parent2 = random pair from elites
        child = crossover(parent1, parent2)
        child = mutate(child, rate, strength)
        children.append(child)

    mutate each second_tier member (elites are immune)
    population = elites + second_tier + children
    decay mutation rate and strength linearly

champion = highest fitness player in final generation
```

---

## Key Files

| File | Purpose |
|------|---------|
| `ai_evolved.h/cpp` | LSTMLayer, FCLayer, EvolvedNet, AIEvolved (network + AI interface) |
| `genetic_trainer.h/cpp` | Individual, GeneticTrainer (GA loop, TSER, crossover, mutation) |
| `ga_dashboard.h/cpp` | GADashboard (live terminal dashboard with stats tracking) |
| `main.cpp` | Entry point — configures and runs GeneticTrainer |

---

## Dashboard Metrics

The training dashboard displays live statistics collected via `ObserverStatKeeper` attached to each game session:

| Metric | Description |
|--------|-------------|
| Fitness (ANE) | Best, average, worst across population |
| Per-opponent earnings | Best individual's normalized earnings vs each opponent |
| W/L record | Win/loss count across all evaluation sessions |
| Win % | Percentage of sessions won per opponent |
| VPIP | Voluntary Put Money In Pot — how often the agent voluntarily enters the hand |
| PFR | Pre-Flop Raise — how often the agent raises pre-flop |
| AF | Aggression Factor — ratio of aggressive to passive actions post-flop |
| WSD | Went to Showdown — how often the agent sees showdown |
| WSDW | Won at Showdown — win rate when reaching showdown |
| Action breakdown | Fold/check/call/raise/all-in percentages |
| Fitness trend | Sparkline of best and average fitness across generations |

---

## Previous Implementation (for reference)

The old REINFORCE-based system remains in the codebase but is no longer used for training.

| Aspect | Old (REINFORCE) | Current (Neuroevolution) |
|--------|-----------------|--------------------------|
| Input | 28 features (raw cards + game state) | 8 features (equity + strategic info) |
| Network | Single LSTM on action history + static embedding | Dual-LSTM (game + opponent) + FC head |
| Output | 2D vector, angle-based action zones | Single scalar, threshold-based decisions |
| Training | Policy gradient (Adam, lr=1e-4) | Genetic algorithm (no gradients) |
| Opponent | AISmart only | 4 opponent archetypes |
| Framework | libtorch | Raw C++ |
| Files | `ai_rl.h/cpp`, `poker_net.h`, `converter.h/cpp` | `ai_evolved.h/cpp`, `genetic_trainer.h/cpp` |
