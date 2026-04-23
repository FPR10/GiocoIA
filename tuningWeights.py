"""
tuneWeights.py
==============
Ottimizzazione automatica dei pesi euristici di playerExampleNostro.py
tramite hill-climbing stocastico con self-play headless.

Come funziona
─────────────
1. Si parte dai pesi di default definiti in BASE_WEIGHTS.
2. Ad ogni iterazione si genera un candidato perturbando casualmente
   uno o più pesi.
3. Si fanno girare GAMES_PER_EVAL partite headless (senza GUI):
      candidato (Red)  vs  corrente (Blue)   -> metà partite
      candidato (Blue) vs  corrente (Red)     -> metà partite
   Alternare i colori elimina il vantaggio del primo giocatore.
4. Se il candidato vince più del 50 % delle partite, diventa il nuovo
   corrente (con un piccolo margine MIN_WIN_RATE per resistere al rumore).
5. Al termine il miglior set di pesi trovato viene stampato e salvato
   in best_weights.json.

Parametri configurabili (sezione CONFIG)
─────────────────────────────────────────
GAMES_PER_EVAL   partite per confronto (più alto = più accurato, più lento)
MAX_ITERATIONS   -> iterazioni totali dell'hill climbing
SEARCH_DEPTH     -> profondità alpha-beta fissa usata nelle partite headless
                   (valore basso = più veloce; usa 2 o 3)
MAX_MOVES        -> tetto al numero di mosse per partita (anti-loop)
MIN_WIN_RATE     -> win-rate minimo del candidato per essere accettato
PERTURB_N        -> quanti pesi perturbare contemporaneamente
PERTURB_RANGE    -> ampiezza massima della perturbazione (±)
WORKERS          -> partite in parallelo (usa os.cpu_count() per il massimo)

Uso
───
    python tuneWeights.py

Output
──────
    Stampa su console il progresso e scrive best_weights.json con i pesi
    ottimali da copiare in playerExampleNostro.py.
"""

import sys
import os
import json
import math
import random
import time
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── assicuriamoci di trovare ZolaGameS e playerExampleNostro nella stessa dir ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ZolaGameS import ZolaGame          # motore di gioco
import playerExampleNostro as _P        # strategia da ottimizzare


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  ──  modifica questi valori per bilanciare velocità/accuratezza
# ═══════════════════════════════════════════════════════════════════════════════

GAMES_PER_EVAL  = 20    # partite per valutare un candidato (min consigliato: 20)
MAX_ITERATIONS  = 40   # iterazioni hill-climbing totali
SEARCH_DEPTH    = 2    # profondità alpha-beta nelle partite headless
MAX_MOVES       = 400   # mosse massime per partita prima di dichiarare pari
MIN_WIN_RATE    = 0.52  # soglia minima per accettare il candidato
PERTURB_N       = 2     # quanti pesi perturbare per iterazione
PERTURB_RANGE   = 8     # perturbazione massima ± per ogni peso
WORKERS         = max(1, (os.cpu_count() or 2) - 1)  # processi paralleli

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


# ──────────────────────────────────────────────────────────────────────────────
# Simulatore headless
# ──────────────────────────────────────────────────────────────────────────────

def _apply_weights(module, weights: dict):
    """Imposta i pesi globali nel modulo strategia."""
    for k, v in weights.items():
        setattr(module, k, v)


def _make_strategy(weights: dict):
    """Ritorna una funzione strategy che usa i pesi dati (closure pulita)."""

    # importiamo di nuovo il modulo in modo isolato per evitare conflitti
    # tra processi paralleli: ogni chiamata usa una copia locale dei pesi
    import importlib
    import playerExampleNostro as mod

    _apply_weights(mod, weights)

    def strategy(game, state, timeout=60):
        """Alpha-beta a profondità fissa (niente timeout nel headless)."""
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
    """Gioca una partita headless. Ritorna 'Red', 'Blue' o 'Draw'."""
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
        # partita finita per MAX_MOVES: vince chi ha più pedine
        red_c  = state.count("Red")
        blue_c = state.count("Blue")
        if red_c > blue_c:
            return "Red"
        if blue_c > red_c:
            return "Blue"
        return "Draw"
    return winner


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper per ProcessPoolExecutor (deve essere pickle-able → funzione top-level)
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_game(args):
    w_cand, w_curr, cand_is_red, seed = args
    if cand_is_red:
        winner = simulate_game(w_cand, w_curr, seed)
        return 1 if winner == "Red" else (0.5 if winner == "Draw" else 0)
    else:
        winner = simulate_game(w_curr, w_cand, seed)
        return 1 if winner == "Blue" else (0.5 if winner == "Draw" else 0)


# ──────────────────────────────────────────────────────────────────────────────
# Valutazione: win-rate del candidato vs corrente
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_candidate(weights_cand: dict, weights_curr: dict, n_games: int) -> float:
    """
    Fa giocare n_games partite alternando i colori.
    Ritorna il win-rate del candidato in [0, 1].
    """
    half = n_games // 2
    tasks = []
    for i in range(half):
        tasks.append((weights_cand, weights_curr, True,  i * 2))      # cand = Red
    for i in range(n_games - half):
        tasks.append((weights_cand, weights_curr, False, i * 2 + 1))  # cand = Blue

    points = 0.0
    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futures = {exe.submit(_run_single_game, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                points += fut.result()
            except Exception as exc:
                print(f"  [WARN] partita fallita: {exc}")
                points += 0.5   # pareggio per sicurezza

    return points / n_games


# ──────────────────────────────────────────────────────────────────────────────
# Perturbazione dei pesi
# ──────────────────────────────────────────────────────────────────────────────

def perturb(weights: dict, n: int = PERTURB_N, rng: int = PERTURB_RANGE) -> dict:
    """Genera un candidato perturbando n pesi scelti a caso."""
    candidate = weights.copy()
    keys = random.sample(list(weights.keys()), k=min(n, len(weights)))
    for k in keys:
        delta = random.randint(-rng, rng)
        candidate[k] = max(1, candidate[k] + delta)   # pesi >= 1
    return candidate


# ──────────────────────────────────────────────────────────────────────────────
# Hill climbing principale
# ──────────────────────────────────────────────────────────────────────────────

def hill_climb():
    print("=" * 64)
    print("  Ottimizzazione pesi -> Hill Climbing con self-play headless")
    print("=" * 64)
    print(f"  Partite/valutazione : {GAMES_PER_EVAL}")
    print(f"  Iterazioni max      : {MAX_ITERATIONS}")
    print(f"  Profondita alpha-beta  : {SEARCH_DEPTH}")
    print(f"  Worker paralleli    : {WORKERS}")
    print(f"  Min win-rate        : {MIN_WIN_RATE:.0%}")
    print("=" * 64)

    current = BASE_WEIGHTS.copy()
    best    = current.copy()
    best_wr = 0.5    # win-rate rispetto a se stesso = 0.5 per definizione

    history = []     # (iterazione, win_rate, pesi)

    for it in range(1, MAX_ITERATIONS + 1):
        t0 = time.perf_counter()
        candidate = perturb(current)

        # mostra solo i pesi cambiati
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

    # ── risultati finali ─────────────────────────────────────────────────────
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

    # ── salvataggio ─────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_weights.json")
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


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Necessario su Windows per il multiprocessing con spawn
    from multiprocessing import freeze_support
    freeze_support()

    hill_climb()