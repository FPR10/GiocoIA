"""
tuneWeights_v2.py
=================
Ottimizzazione automatica dei pesi euristici tramite hill-climbing
stocastico con self-play headless contro un POOL di avversari.

Come funziona
─────────────
1. Si parte dai pesi di default definiti in BASE_WEIGHTS (coerenti con
   il player da ottimizzare).
2. Ad ogni iterazione si genera un candidato perturbando casualmente
   uno o più pesi.
3. Il candidato viene valutato contro OGNI avversario del pool:
      candidato (Red)  vs  avversario (Blue)   -> metà partite
      candidato (Blue) vs  avversario (Red)     -> metà partite
   Il win-rate finale è la media pesata sui pool_weights.
4. Se il win-rate supera MIN_WIN_RATE, il candidato diventa il nuovo
   "current" (base per la prossima perturbazione).
5. Al termine il miglior set di pesi viene stampato e salvato in
   best_weights.json.

Configurazione del pool (sezione POOL)
───────────────────────────────────────
Ogni entry è un dizionario con:
  "name"     -> nome descrittivo dell'avversario
  "module"   -> stringa del modulo Python da importare
  "weight"   -> importanza relativa dell'avversario nel calcolo del win-rate
                (es. 2.0 = vale doppio rispetto a un avversario con peso 1.0)

Parametri configurabili (sezione CONFIG)
─────────────────────────────────────────
GAMES_PER_EVAL   partite per ogni coppia candidato-avversario
MAX_ITERATIONS   iterazioni totali dell'hill climbing
SEARCH_DEPTH     profondità alpha-beta fissa nelle partite headless
MAX_MOVES        tetto al numero di mosse per partita (anti-loop)
MIN_WIN_RATE     win-rate minimo del candidato per essere accettato
PERTURB_N        quanti pesi perturbare contemporaneamente
PERTURB_RANGE    ampiezza massima della perturbazione (±)
WORKERS          processi paralleli

Uso
───
    python tuneWeights_v2.py

Output
──────
    Stampa su console il progresso e scrive best_weights.json.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    # Fallback minimale se tqdm non è installato
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        class _DummyTqdm:
            def __init__(self): pass
            def update(self, n=1): pass
            def set_postfix_str(self, s): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _DummyTqdm()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ZolaGameS import ZolaGame

# ═══════════════════════════════════════════════════════════════════════════════
# POOL DI AVVERSARI
# ═══════════════════════════════════════════════════════════════════════════════
# Modifica qui per aggiungere/rimuovere avversari.
# "module" deve essere il percorso importabile del modulo (es. "playerExampleRandom").
# "weight" determina quanto pesa questo avversario nel win-rate finale:
#   - avversari più forti → peso maggiore (orientano l'ottimizzazione verso la forza)
#   - avversari più deboli → peso minore (servono per non "dimenticare" le basi)

OPPONENT_POOL = [
    {
        "name":   "Random",
        "module": "playerProfessore.playerExampleRandom",
        "weight": 0.5,   # avversario debole: peso basso
    },
    {
        "name":   "AlphaBetaProfessore",
        "module": "playerProfessore.playerExampleAlpha",
        "weight": 1,   # avversario medio
    },
    {
        "name":   "Nostro v1",
        "module": "vecchiPlayerEuristici.playerExampleNostro_v1",
        "weight": 1.5,  
    },
    {
        "name":   "Nostro v2",
        "module": "vecchiPlayerEuristici.playerExampleNostro_v2",
        "weight": 1.5,  
    },
    {
        "name":   "Nostro EU",
        "module": "vecchiPlayerEuristici.playerExampleNostroEU",
        "weight": 1.5,   
    },
    {
        "name":   "Montecarlo",
        "module": "vecchiPlayerMonteCarlo.playerExampleNostroMC",
        "weight": 1.5,  
    },
    {
        "name":   "Regola standard 2",
        "module": "playerExampleNostroRegolaStandard2",
        "weight": 2,  
    },
]
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GAMES_PER_EVAL  = 20    # partite per ogni coppia (candidato vs avversario)
MAX_ITERATIONS  = 40    # iterazioni hill-climbing totali
SEARCH_DEPTH    = 2     # profondità alpha-beta nelle partite headless
MAX_MOVES       = 400   # mosse massime per partita prima di dichiarare pari
MIN_WIN_RATE    = 0.52  # soglia minima per accettare il candidato
PERTURB_N       = 2     # quanti pesi perturbare per iterazione
PERTURB_RANGE   = 8     # perturbazione massima ± per ogni peso
WORKERS         = max(1, (os.cpu_count() or 2) - 1)

# ── Pesi base del player da ottimizzare (documento 6) ────────────────────────
# Devono corrispondere esattamente ai nomi delle variabili nel modulo player.
BASE_WEIGHTS = {
    "_W_PIECES":            80,
    "_W_MOBILITY":           2,
    "_W_CAPTURE_COUNT":     10,
    "_W_CAPTURE_OUTER":      5,
    "_W_MOVE_OUTER":         3,
    "_W_THREAT_PRESSURE":    1,
    "_W_CAPTURE_DANGEROUS":  2,
    "_W_CORNER_SETUP":       4,
}

# Modulo del player da ottimizzare (cambia questo se il file si chiama diversamente)
PLAYER_MODULE = "playerExampleNostroIbrido"   # <- adatta al nome del tuo file

# ═══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Helpers interni
# ──────────────────────────────────────────────────────────────────────────────

def _apply_weights(module, weights: dict):
    """Imposta i pesi globali nel modulo strategia."""
    for k, v in weights.items():
        setattr(module, k, v)


def _make_candidate_strategy(weights: dict):
    """
    Ritorna una funzione strategy che usa i pesi dati.
    Usa alpha-beta a profondità fissa (SEARCH_DEPTH) per velocità nel tuning.
    """
    import importlib
    mod = importlib.import_module(PLAYER_MODULE)
    _apply_weights(mod, weights)

    def strategy(game, state, timeout=60):
        legal_moves = game.actions(state)
        if not legal_moves:
            return None

        root_player = state.to_move

        def _ab(state, depth, alpha, beta, maximizing):
            lm = game.actions(state)
            if depth == 0 or game.is_terminal(state):
                return mod.evaluate_state(game, state, root_player), None
            if not lm:
                # Passa il turno se non ci sono mosse
                passed = game.pass_turn(state)
                return _ab(passed, depth - 1, alpha, beta, not maximizing)

            ordered = mod.order_moves(game, lm)
            best_moves = []

            if maximizing:
                value = -math.inf
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
                value = math.inf
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

            return value, (best_moves[0] if best_moves else None)

        _, best = _ab(state, SEARCH_DEPTH, -math.inf, math.inf, True)
        return best if best is not None else random.choice(legal_moves)

    return strategy


def _make_opponent_strategy(module_name: str):
    """
    Carica la strategia di un avversario del pool.
    Usa direttamente playerStrategy del modulo importato.
    """
    import importlib
    mod = importlib.import_module(module_name)
    return mod.playerStrategy


# ──────────────────────────────────────────────────────────────────────────────
# Simulazione headless di una singola partita
# ──────────────────────────────────────────────────────────────────────────────

def simulate_game(
    strategy_red,
    strategy_blue,
    seed: int = None,
) -> str:
    """Gioca una partita headless. Ritorna 'Red', 'Blue' o 'Draw'."""
    if seed is not None:
        random.seed(seed)

    game  = ZolaGame(size=8, first_player="Red")
    state = game.initial

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


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper pickle-able per ProcessPoolExecutor
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_game(args):
    """
    args = (weights_cand, opponent_module, cand_is_red, seed, game_index, total_games)

    Costruisce le due strategie nel processo figlio (sicuro con spawn/fork).
    Ritorna (punti, info_stringa) con dettagli sulla partita giocata.
    """
    weights_cand, opponent_module, cand_is_red, seed, game_index, total_games = args

    cand_strategy = _make_candidate_strategy(weights_cand)
    opp_strategy  = _make_opponent_strategy(opponent_module)

    opp_short = opponent_module.split(".")[-1]   # solo il nome del file, senza package

    if cand_is_red:
        red_label  = f"Candidato"
        blue_label = opp_short
        winner = simulate_game(cand_strategy, opp_strategy, seed)
        pt = 1.0 if winner == "Red" else (0.5 if winner == "Draw" else 0.0)
        result_label = "WIN" if winner == "Red" else ("DRAW" if winner == "Draw" else "LOSS")
    else:
        red_label  = opp_short
        blue_label = f"Candidato"
        winner = simulate_game(opp_strategy, cand_strategy, seed)
        pt = 1.0 if winner == "Blue" else (0.5 if winner == "Draw" else 0.0)
        result_label = "WIN" if winner == "Blue" else ("DRAW" if winner == "Draw" else "LOSS")

    info = (
        f"[{game_index:>3}/{total_games}] "
        f"Red: {red_label:<28} vs  Blue: {blue_label:<28}  "
        f"→ vince {winner:<5}  ({result_label})"
    )
    return pt, info


# ──────────────────────────────────────────────────────────────────────────────
# Valutazione del candidato contro l'intero pool
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_candidate_vs_pool(weights_cand: dict, n_games: int, it: int) -> tuple[float, dict]:
    """
    Valuta il candidato contro tutti gli avversari del pool.

    Ritorna:
      (win_rate_pesato_globale, {nome_avversario: win_rate})
    """
    all_tasks = []
    task_meta = []   # (opponent_name, opponent_weight)

    total_games = len(OPPONENT_POOL) * n_games
    game_index  = 0

    for opp in OPPONENT_POOL:
        half = n_games // 2
        for i in range(half):
            game_index += 1
            all_tasks.append((weights_cand, opp["module"], True,  i * 2, game_index, total_games))
            task_meta.append((opp["name"], opp["weight"]))
        for i in range(n_games - half):
            game_index += 1
            all_tasks.append((weights_cand, opp["module"], False, i * 2 + 1, game_index, total_games))
            task_meta.append((opp["name"], opp["weight"]))

    points_by_opp = {opp["name"]: 0.0 for opp in OPPONENT_POOL}
    games_by_opp  = {opp["name"]: 0   for opp in OPPONENT_POOL}
    games_done    = 0

    print(f"\n  Iterazione {it}/{MAX_ITERATIONS} – {total_games} partite in corso "
          f"({len(OPPONENT_POOL)} avversari × {n_games} partite)")

    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futures = {
            exe.submit(_run_single_game, task): (task, meta)
            for task, meta in zip(all_tasks, task_meta)
        }

        with tqdm(total=total_games, unit="partita", ncols=80,
                  bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                  ) as pbar:

            for fut in as_completed(futures):
                _, (opp_name, _) = futures[fut]
                try:
                    pt, info = fut.result()
                    print(f"    {info}")
                except Exception as exc:
                    pt   = 0.5
                    print(f"  [WARN] partita fallita ({opp_name}): {exc}")

                points_by_opp[opp_name] += pt
                games_by_opp[opp_name]  += 1
                games_done += 1

                # Aggiorna la barra con il win-rate parziale dell'avversario corrente
                wr_now = points_by_opp[opp_name] / games_by_opp[opp_name]
                pbar.set_postfix_str(f"vs {opp_name}: wr={wr_now:.2f}")
                pbar.update(1)

    # Win-rate per avversario
    wr_by_opp = {}
    for opp in OPPONENT_POOL:
        name = opp["name"]
        g    = games_by_opp[name]
        wr_by_opp[name] = points_by_opp[name] / g if g > 0 else 0.5

    # Win-rate globale pesato
    total_weight = sum(opp["weight"] for opp in OPPONENT_POOL)
    weighted_wr  = sum(
        wr_by_opp[opp["name"]] * opp["weight"]
        for opp in OPPONENT_POOL
    ) / total_weight

    return weighted_wr, wr_by_opp


# ──────────────────────────────────────────────────────────────────────────────
# Perturbazione dei pesi
# ──────────────────────────────────────────────────────────────────────────────

def perturb(weights: dict, n: int = PERTURB_N, rng: int = PERTURB_RANGE) -> dict:
    """Genera un candidato perturbando n pesi scelti a caso."""
    candidate = weights.copy()
    keys = random.sample(list(weights.keys()), k=min(n, len(weights)))
    for k in keys:
        delta = random.randint(-rng, rng)
        candidate[k] = max(1, candidate[k] + delta)
    return candidate


# ──────────────────────────────────────────────────────────────────────────────
# Hill climbing principale
# ──────────────────────────────────────────────────────────────────────────────

def hill_climb():
    print("=" * 68)
    print("  Ottimizzazione pesi – Hill Climbing con pool di avversari")
    print("=" * 68)
    print(f"  Player da ottimizzare : {PLAYER_MODULE}")
    print(f"  Avversari nel pool    : {len(OPPONENT_POOL)}")
    for opp in OPPONENT_POOL:
        print(f"    • {opp['name']:<20} (peso {opp['weight']})")
    print(f"  Partite/avversario    : {GAMES_PER_EVAL}")
    print(f"  Iterazioni max        : {MAX_ITERATIONS}")
    print(f"  Profondità alpha-beta : {SEARCH_DEPTH}")
    print(f"  Worker paralleli      : {WORKERS}")
    print(f"  Min win-rate (pesato) : {MIN_WIN_RATE:.0%}")
    print("=" * 68)

    current  = BASE_WEIGHTS.copy()
    best     = current.copy()
    best_wr  = 0.0

    history  = []

    iter_bar = tqdm(range(1, MAX_ITERATIONS + 1), unit="iter", ncols=80,
                    bar_format="  Iterazioni: {l_bar}{bar}| {n_fmt}/{total_fmt}")

    for it in iter_bar:
        t0        = time.perf_counter()
        candidate = perturb(current)

        changed     = {k: (current[k], candidate[k])
                       for k in candidate if candidate[k] != current[k]}
        changed_str = "  ".join(f"{k}: {v[0]}→{v[1]}" for k, v in changed.items())

        wr, wr_detail = evaluate_candidate_vs_pool(candidate, GAMES_PER_EVAL, it)
        elapsed       = time.perf_counter() - t0

        accepted = wr >= MIN_WIN_RATE
        tag      = "✓ accettato" if accepted else "✗ rifiutato"

        print(f"\n[{it:3d}/{MAX_ITERATIONS}]  wr_pesato={wr:.3f}  {tag}  ({elapsed:.1f}s)")
        if changed_str:
            print(f"  modifiche: {changed_str}")
        for opp in OPPONENT_POOL:
            name = opp["name"]
            print(f"  vs {name:<22} wr={wr_detail[name]:.3f}")

        if accepted:
            current = candidate
            if wr > best_wr:
                best    = candidate.copy()
                best_wr = wr
                print(f"  ★ Nuovo miglior set (wr_pesato={best_wr:.3f})")

        iter_bar.set_postfix_str(f"best_wr={best_wr:.3f}")

        history.append({
            "iter":       it,
            "win_rate":   round(wr, 4),
            "wr_detail":  {k: round(v, 4) for k, v in wr_detail.items()},
            "accepted":   accepted,
            "weights":    candidate.copy(),
        })

    # ── risultati finali ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  OTTIMIZZAZIONE COMPLETATA")
    print("=" * 68)
    print(f"  Miglior win-rate pesato registrato: {best_wr:.3f}")
    print("\n  Pesi ottimali:")
    for k, v in best.items():
        orig = BASE_WEIGHTS[k]
        diff = v - orig
        sign = f"+{diff}" if diff >= 0 else str(diff)
        print(f"    {k:<26} = {v:>4}   (base {orig:>3}, {sign})")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_weights.json")
    with open(out_path, "w") as f:
        json.dump({
            "best_weights":   best,
            "best_win_rate":  best_wr,
            "base_weights":   BASE_WEIGHTS,
            "pool":           [{"name": o["name"], "weight": o["weight"]}
                                for o in OPPONENT_POOL],
            "history":        history,
        }, f, indent=2)

    print(f"\n  Risultati salvati in: {out_path}")
    print("\n  Copia questi valori nel tuo player:")
    print("  " + "-" * 52)
    for k, v in best.items():
        print(f"  {k:<26} = {v}")
    print("  " + "-" * 52)

    return best


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    hill_climb()