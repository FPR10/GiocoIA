"""
tuneWeights.py
==============
Ottimizzazione automatica dei pesi euristici di playerExampleNostro.py
tramite hill-climbing stocastico con self-play headless.
"""

import sys
import os
import json
import math
import random
import time
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── compatibile sia con script (.py) che con notebook (Jupyter / VS Code) ──
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.insert(0, BASE_DIR)

from ZolaGameS import ZolaGame
import playerExampleNostroIbrido2 as _P


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GAMES_PER_EVAL  = 20
MAX_ITERATIONS  = 40
SEARCH_DEPTH    = 2
MAX_MOVES       = 400
MIN_WIN_RATE    = 0.52
PERTURB_N       = 2
PERTURB_RANGE   = 8
WORKERS         = max(1, (os.cpu_count() or 2) - 1)

BASE_WEIGHTS = {
    "_W_PIECES":         50,
    "_W_MOBILITY":        2,
    "_W_CAPTURE_COUNT":   5,
    "_W_CAPTURE_OUTER":   4,
    "_W_CAPTURE_INNER":   6,
    "_W_MOVE_OUTER":      3,
    "_W_OUTER_PRESSURE":  4,
    "_W_CORNER_SETUP":    5,
}

# ═══════════════════════════════════════════════════════════════════════════════


def _apply_weights(module, weights: dict):
    for k, v in weights.items():
        setattr(module, k, v)


def _make_strategy(weights: dict):
    import playerExampleNostroIbrido2 as mod
    _apply_weights(mod, weights)

    def strategy(game, state, timeout=60):
        legal_moves = game.actions(state)
        if not legal_moves:
            return None

        import math as _math

        def _ab(state, depth, alpha, beta, maximizing):
            lm = game.actions(state)
            if depth == 0 or game.is_terminal(state) or not lm:
                return mod.evaluate_state(game, state, root_player), None

            ordered = mod.order_moves(game, lm)
            best_moves = []

            if maximizing:
                value = -_math.inf
                for mv in ordered:
                    child = game.result(state, mv)
                    cv, _ = _ab(child, depth - 1, alpha, beta, False)
                    if cv > value:
                        value = cv
                        best_moves = [mv]
                    elif cv == value:
                        best_moves.append(mv)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            else:
                value = _math.inf
                for mv in ordered:
                    child = game.result(state, mv)
                    cv, _ = _ab(child, depth - 1, alpha, beta, True)
                    if cv < value:
                        value = cv
                        best_moves = [mv]
                    elif cv == value:
                        best_moves.append(mv)
                    beta = min(beta, value)
                    if alpha >= beta:
                        break

            return value, (random.choice(best_moves) if best_moves else None)

        root_player = state.to_move
        _, best = _ab(state, SEARCH_DEPTH, -_math.inf, _math.inf, True)
        return best if best is not None else random.choice(legal_moves)

    return strategy


def simulate_game(weights_red: dict, weights_blue: dict, seed: int = None) -> str:
    if seed is not None:
        random.seed(seed)

    game = ZolaGame(size=8, first_player="Red")
    state = game.initial
    strategy_red  = _make_strategy(weights_red)
    strategy_blue = _make_strategy(weights_blue)

    for _ in range(MAX_MOVES):
        if game.is_terminal(state):
            break

        legal = game.actions(state)
        if not legal:
            state = game.pass_turn(state)
            continue

        if state.to_move == "Red":
            move = strategy_red(game, state)
        else:
            move = strategy_blue(game, state)

        if move is None or move not in legal:
            move = random.choice(legal)

        state = game.result(state, move)

    winner = game.winner(state)
    if winner is None:
        red_c  = state.count("Red")
        blue_c = state.count("Blue")
        if red_c > blue_c:
            return "Red"
        if blue_c > red_c:
            return "Blue"
        return "Draw"
    return winner


# ── Worker top-level: riceve BASE_DIR esplicitamente per garantire il path ──

def _run_single_game(args):
    """Wrapper eseguito nei processi worker. Gestisce il sys.path autonomamente."""
    w_cand, w_curr, cand_is_red, seed, base_dir = args

    # ogni worker deve aggiungere il path da solo (i processi spawn non lo ereditano)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    try:
        if cand_is_red:
            winner = simulate_game(w_cand, w_curr, seed)
            return 1 if winner == "Red" else (0.5 if winner == "Draw" else 0)
        else:
            winner = simulate_game(w_curr, w_cand, seed)
            return 1 if winner == "Blue" else (0.5 if winner == "Draw" else 0)
    except Exception as e:
        import traceback
        raise RuntimeError(
            f"Errore nella partita (seed={seed}, cand_is_red={cand_is_red}):\n"
            + traceback.format_exc()
        ) from e


def evaluate_candidate(weights_cand: dict, weights_curr: dict, n_games: int) -> float:
    half = n_games // 2
    tasks = []
    for i in range(half):
        tasks.append((weights_cand, weights_curr, True,  i * 2,     BASE_DIR))
    for i in range(n_games - half):
        tasks.append((weights_cand, weights_curr, False, i * 2 + 1, BASE_DIR))

    points = 0.0
    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futures = {exe.submit(_run_single_game, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                points += fut.result()
            except Exception as exc:
                print(f"  [WARN] partita fallita: {exc}")
                points += 0.5

    return points / n_games


def perturb(weights: dict, n: int = PERTURB_N, rng: int = PERTURB_RANGE) -> dict:
    candidate = weights.copy()
    keys = random.sample(list(weights.keys()), k=min(n, len(weights)))
    for k in keys:
        delta = random.randint(-rng, rng)
        candidate[k] = max(1, candidate[k] + delta)
    return candidate


def hill_climb():
    print("=" * 64)
    print("  Ottimizzazione pesi -> Hill Climbing con self-play headless")
    print("=" * 64)
    print(f"  Partite/valutazione : {GAMES_PER_EVAL}")
    print(f"  Iterazioni max      : {MAX_ITERATIONS}")
    print(f"  Profondita alpha-beta  : {SEARCH_DEPTH}")
    print(f"  Worker paralleli    : {WORKERS}")
    print(f"  Min win-rate        : {MIN_WIN_RATE:.0%}")
    print(f"  BASE_DIR            : {BASE_DIR}")
    print("=" * 64)

    # ── test rapido: verifica che i worker riescano ad importare i moduli ──
    print("  Verifica import nei worker... ", end="", flush=True)
    try:
        test_task = (BASE_WEIGHTS.copy(), BASE_WEIGHTS.copy(), True, 0, BASE_DIR)
        with ProcessPoolExecutor(max_workers=1) as exe:
            result = exe.submit(_run_single_game, test_task).result(timeout=60)
        print(f"OK (risultato test: {result})")
    except Exception as e:
        print(f"\n  [ERRORE] Il worker non riesce ad avviarsi:\n  {e}")
        print("  Controlla che ZolaGameS e playerExampleNostroIbrido2 siano in:")
        print(f"  {BASE_DIR}")
        return None
    print("=" * 64)

    current = BASE_WEIGHTS.copy()
    best    = current.copy()
    best_wr = 0.5
    history = []

    for it in range(1, MAX_ITERATIONS + 1):
        t0 = time.perf_counter()
        candidate = perturb(current)

        changed = {k: (current[k], candidate[k])
                   for k in candidate if candidate[k] != current[k]}
        changed_str = "  ".join(f"{k}: {v[0]}→{v[1]}" for k, v in changed.items())

        wr = evaluate_candidate(candidate, current, GAMES_PER_EVAL)
        elapsed = time.perf_counter() - t0

        accepted = wr >= MIN_WIN_RATE
        tag = "✓ accettato" if accepted else "✗ rifiutato"
        print(f"[{it:3d}/{MAX_ITERATIONS}]  wr={wr:.3f}  {tag}  ({elapsed:.1f}s)")
        if changed_str:
            print(f"         modifiche: {changed_str}")

        if accepted:
            current = candidate
            if wr > best_wr:
                best    = candidate.copy()
                best_wr = wr
                print(f"  ★ Nuovo miglior set (wr={best_wr:.3f})")

        history.append({"iter": it, "win_rate": round(wr, 4),
                         "accepted": accepted, "weights": candidate.copy()})

    print("\n" + "=" * 64)
    print("  OTTIMIZZAZIONE COMPLETATA")
    print("=" * 64)
    print(f"  Miglior win-rate registrato: {best_wr:.3f}")
    print("\n  Pesi ottimali:")
    for k, v in best.items():
        orig = BASE_WEIGHTS[k]
        diff = v - orig
        sign = f"+{diff}" if diff > 0 else str(diff)
        print(f"    {k:<22} = {v:>4}   (base {orig:>3}, {sign})")

    out_path = os.path.join(BASE_DIR, "best_weights.json")
    with open(out_path, "w") as f:
        json.dump({"best_weights": best, "best_win_rate": best_wr,
                   "history": history}, f, indent=2)
    print(f"\n  Risultati salvati in: {out_path}")
    print("\n  Copia questi valori in playerExampleNostro.py:")
    print("  " + "-" * 50)
    for k, v in best.items():
        print(f"  {k:<22} = {v}")
    print("  " + "-" * 50)

    return best


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    hill_climb()