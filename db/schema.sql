-- PostgreSQL schema for poker GA training history
-- Usage: createdb poker_ga && psql poker_ga < db/schema.sql

CREATE TABLE runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMP DEFAULT NOW(),
    population_size INT,
    num_generations INT,
    hands_per_session INT,
    buy_in          INT,
    big_blind       INT,
    small_blind     INT,
    genome_size     INT,
    survival_rate   FLOAT
);

CREATE TABLE opponents (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    label VARCHAR(100)
);

CREATE TABLE generations (
    id              SERIAL PRIMARY KEY,
    run_id          INT REFERENCES runs(id),
    generation_num  INT,
    best_fitness    FLOAT,
    avg_fitness     FLOAT,
    worst_fitness   FLOAT,
    num_elites      INT,
    num_survivors   INT,
    mutation_rate   FLOAT,
    mutation_strength FLOAT,
    gen_time_secs   FLOAT,
    UNIQUE(run_id, generation_num)
);

CREATE TABLE individuals (
    id              SERIAL PRIMARY KEY,
    generation_id   INT REFERENCES generations(id),
    fitness         FLOAT
);

CREATE TABLE individual_opponent_stats (
    id              SERIAL PRIMARY KEY,
    individual_id   INT REFERENCES individuals(id),
    opponent_id     INT REFERENCES opponents(id),
    normalized_earnings FLOAT,
    raw_earnings    FLOAT,
    mbb_per_hand    FLOAT,
    sessions_won    INT,
    sessions_lost   INT
);

CREATE TABLE games (
    id              SERIAL PRIMARY KEY,
    individual_id   INT REFERENCES individuals(id),
    opponent_id     INT REFERENCES opponents(id),
    seat_position   INT,
    final_stack_evolved  INT,
    final_stack_opponent INT,
    earnings        FLOAT,
    num_hands       INT
);

CREATE TABLE hands (
    id              SERIAL PRIMARY KEY,
    game_id         INT REFERENCES games(id),
    hand_num        INT,
    dealer_player   VARCHAR(20),
    evolved_card1   VARCHAR(3),
    evolved_card2   VARCHAR(3),
    opponent_card1  VARCHAR(3),
    opponent_card2  VARCHAR(3),
    flop1           VARCHAR(3),
    flop2           VARCHAR(3),
    flop3           VARCHAR(3),
    turn_card       VARCHAR(3),
    river_card      VARCHAR(3),
    pot_total       INT,
    winner_player   VARCHAR(20),
    win_amount      INT,
    went_to_showdown BOOLEAN DEFAULT FALSE,
    final_round     VARCHAR(10),
    evolved_stack_before  INT,
    opponent_stack_before INT
);

CREATE TABLE actions (
    id              SERIAL PRIMARY KEY,
    hand_id         INT REFERENCES hands(id),
    action_order    INT,
    round           VARCHAR(10),
    player          VARCHAR(20),
    action_type     VARCHAR(20),
    amount          INT DEFAULT 0
);

-- Indexes for common queries
CREATE INDEX idx_generations_run ON generations(run_id, generation_num);
CREATE INDEX idx_games_individual ON games(individual_id);
CREATE INDEX idx_games_opponent ON games(opponent_id);
CREATE INDEX idx_hands_game ON hands(game_id);
CREATE INDEX idx_hands_winner ON hands(winner_player);
CREATE INDEX idx_hands_showdown ON hands(went_to_showdown);
CREATE INDEX idx_actions_hand ON actions(hand_id);
CREATE INDEX idx_actions_type ON actions(action_type);
CREATE INDEX idx_actions_round_type ON actions(round, action_type);

-- Seed opponent data
INSERT INTO opponents (name, label) VALUES
    ('CheckFold', 'Scared Limper'),
    ('Call', 'Calling Machine'),
    ('Raise', 'Hothead Maniac'),
    ('Smart', 'Candid Statistician');
