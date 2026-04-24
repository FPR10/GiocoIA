import os
import sys
import json
import random
import time
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ZolaGameS import ZolaGame
import GiocoIA.vecchiPlayer.playerExampleNostroMC_old as BASE_PLAYER   # ← PLAYER DA ALLENARE

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

GAMES_PER_OPPONENT = 30     # ↑ più stabile per MC
MAX_ITERATIONS     = 30
MAX_MOVES          = 300
MIN_WIN_RATE       = 0.55

PERTURB_N          = 3      # ↑ esplora più dimensioni
PERTURB_RANGE      = 2.5    # ↓ meno rumore

WORKERS            = max(1, (os.cpu_count() or 2) - 1)

OPPONENT_MODULES = [
    "playerExampleNostro",
    "playerExampleAlpha",
    "playerExampleNostro3",
    "playerExampleNostroMC"
]

# ── PESI PER RuleWeights ──────────────────────────────────────

BASE_WEIGHTS = {
    "w1_capture_src": 1.0,
    "w2_capture_tgt": 1.0,
    "w3_aggressive_own_sector": 1.0,
    "w4_evade_external_enemy": 1.0,
    "w5_piece_eval": 0.7,
    "w6_mobility_eval": 0.3,
}

# ═══════════════════════════════════════════════════════════════
# STRATEGY BUILDER (CORE FIX)
# ═══════════════════════════════════════════════════════════════

def make_strategy(weights_dict):
    module_name = "playerExampleNostroMC"

    if module_name in sys.modules:
        del sys.modules[module_name]

    mod = importlib.import_module(module_name)

    # CREA OGGETTO PESI CORRETTO
    weights_obj = mod.RuleWeights(**weights_dict)

    def strategy(game, state):
        return mod.playerStrategy(game, state, timeout=1, weights=weights_obj)

    return strategy


# ═══════════════════════════════════════════════════════════════
# SIMULAZIONE
# ═══════════════════════════════════════════════════════════════

def simulate_game(weights, opponent_name, cand_is_red, seed):
    random.seed(seed)

    opponent_mod = importlib.import_module(opponent_name)

    def opponent_strategy(game, state):
        return opponent_mod.playerStrategy(game, state, timeout=1)

    strategy_cand = make_strategy(weights)

    game = ZolaGame(size=8, first_player="Red")
    state = game.initial

    for _ in range(MAX_MOVES):
        if game.is_terminal(state):
            break

        moves = game.actions(state)
        if not moves:
            state = game.pass_turn(state)
            continue

        if state.to_move == "Red":
            move = strategy_cand(game, state) if cand_is_red else opponent_strategy(game, state)
        else:
            move = opponent_strategy(game, state) if cand_is_red else strategy_cand(game, state)

        if move not in moves:
            move = random.choice(moves)

        state = game.result(state, move)

    winner = game.winner(state)

    if winner is None:
        r = state.count("Red")
        b = state.count("Blue")
        if r > b:
            return "Red"
        elif b > r:
            return "Blue"
        return "Draw"

    return winner


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_vs_opponent(weights, opponent, n_games):
    print(f"\n▶ Contro {opponent}")

    points = 0
    tasks = []

    for i in range(n_games):
        cand_is_red = (i % 2 == 0)
        tasks.append((weights, opponent, cand_is_red, i))

    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futures = [exe.submit(simulate_game, *t) for t in tasks]

        for i, f in enumerate(as_completed(futures), 1):
            winner = f.result()

            cand_is_red = tasks[i-1][2]

            if winner == "Draw":
                points += 0.5
            elif winner == "Red" and cand_is_red:
                points += 1
            elif winner == "Blue" and not cand_is_red:
                points += 1

            if i % 5 == 0 or i == n_games:
                print(f"   progresso: {i}/{n_games}")

    wr = points / n_games
    print(f"   winrate: {wr:.3f}")
    return wr


def evaluate_candidate(weights):
    total = 0
    for opp in OPPONENT_MODULES:
        total += evaluate_vs_opponent(weights, opp, GAMES_PER_OPPONENT)
    return total / len(OPPONENT_MODULES)


# ═══════════════════════════════════════════════════════════════
# PERTURBAZIONE
# ═══════════════════════════════════════════════════════════════

def perturb(weights):
    new = weights.copy()
    keys = random.sample(list(weights.keys()), PERTURB_N)

    for k in keys:
        delta = random.uniform(-PERTURB_RANGE, PERTURB_RANGE)
        new[k] = max(0.01, new[k] + delta)

    return new


# ═══════════════════════════════════════════════════════════════
# HILL CLIMBING
# ═══════════════════════════════════════════════════════════════

def hill_climb():
    current = BASE_WEIGHTS.copy()
    best = current.copy()
    best_score = 0.5

    for it in range(1, MAX_ITERATIONS + 1):
        print("\n" + "="*50)
        print(f"ITERAZIONE {it}/{MAX_ITERATIONS}")

        candidate = perturb(current)

        print("Pesi modificati:")
        for k in candidate:
            if candidate[k] != current[k]:
                print(f"  {k}: {current[k]:.3f} → {candidate[k]:.3f}")

        t0 = time.time()
        score = evaluate_candidate(candidate)
        dt = time.time() - t0

        print(f"\nRisultato: {score:.3f}  (tempo {dt:.1f}s)")

        if score >= MIN_WIN_RATE:
            print("✔ ACCETTATO")
            current = candidate

            if score > best_score:
                best = candidate.copy()
                best_score = score
                print("★ NUOVO BEST")
        else:
            print("✘ RIFIUTATO")

    print("\n" + "="*50)
    print("FINE OTTIMIZZAZIONE")
    print(f"Best score: {best_score:.3f}")

    print("\nPesi finali:")
    for k, v in best.items():
        print(f"{k} = {v:.3f}")

    with open("best_weights.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nSalvato in best_weights.json")


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    hill_climb()

