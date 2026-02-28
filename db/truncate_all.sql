-- Truncate all tables in the poker_ga database
-- Usage: psql poker_ga < db/truncate_all.sql

-- TRUNCATE in dependency order (children first) with CASCADE for safety
TRUNCATE TABLE
    actions,
    hands,
    games,
    individual_opponent_stats,
    individuals,
    generations,
    runs,
    opponents
RESTART IDENTITY;

-- Re-seed opponent data
INSERT INTO opponents (name, label) VALUES
    ('CheckFold', 'Scared Limper'),
    ('Call', 'Calling Machine'),
    ('Raise', 'Hothead Maniac'),
    ('Smart', 'Candid Statistician');
