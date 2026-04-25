"""
tuneWeights_v2.py
=================
Ottimizzazione automatica dei pesi euristici tramite hill-climbing
stocastico con self-play headless contro un POOL di avversari.

Come funziona
─────────────
1. Si parte dai pesi BASE_WEIGHTS.
2. Ad ogni iterazione si genera un candidato perturbando casualmente
   uno o più pesi.
3. Il candidato affronta OGNI avversario del pool in sequenza:
     - GAMES_PER_EVAL partite sequenziali (metà come Red, metà come Blue)
     - Per ogni partita viene mostrata una barra mossa-per-mossa
4. Dopo ogni blocco avversario vengono stampati i pesi correnti.
5. Se il win-rate pesato supera MIN_WIN_RATE, il candidato diventa la
   nuova base per la prossima perturbazione.
6. Al termine il miglior set di pesi viene salvato in best_weights.json.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import math
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ZolaGameS import ZolaGame

# ═══════════════════════════════════════════════════════════════════════════════
# POOL DI AVVERSARI
# ═══════════════════════════════════════════════════════════════════════════════

OPPONENT_POOL = [
    {
        "name":   "Random",
        "module": "playerProfessore.playerExampleRandom",
        "weight": 0.5,
    },
    {
        "name":   "AlphaBetaProfessore",
        "module": "playerProfessore.playerExampleAlpha",
        "weight": 1.0,
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
        "weight": 2.0,
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GAMES_PER_EVAL  = 10     # partite per ogni coppia candidato-avversario
MAX_ITERATIONS  = 40     # iterazioni totali hill-climbing
SEARCH_DEPTH    = 2      # profondità alpha-beta nelle partite headless
MAX_MOVES       = 400    # mosse massime per partita (anti-loop)
MIN_WIN_RATE    = 0.52   # soglia minima per accettare il candidato
PERTURB_N       = 2      # quanti pesi perturbare per iterazione
PERTURB_RANGE   = 8      # perturbazione massima ± per ogni peso

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

PLAYER_MODULE = "playerExampleNostroIbrido"

# ═══════════════════════════════════════════════════════════════════════════════
# COSTANTI OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

_W         = 90    # larghezza righe separatori
_BAR_MOVES = 25    # larghezza barra mosse in-game
_SYM       = {"WIN": "✓ WIN ", "DRAW": "~ DRAW", "LOSS": "✗ LOSS"}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def _bar(done: int, total: int, width: int) -> str:
    filled = int(width * done / total) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _sep(char: str = "─") -> str:
    return char * _W


def _print_weights(weights: dict, label: str = "Pesi candidato"):
    """Stampa i pesi in una riga compatta, abbreviando il prefisso _W_."""
    parts = "  ".join(
        f"{k.replace('_W_', '').replace('_', ' ')}: {v}"
        for k, v in weights.items()
    )
    print(f"  [{label}]  {parts}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIE
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_weights(module, weights: dict):
    for k, v in weights.items():
        setattr(module, k, v)


def _make_candidate_strategy(weights: dict):
    """Costruisce la strategy del candidato con alpha-beta a profondità fissa."""
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
                passed = game.pass_turn(state)
                return _ab(passed, depth - 1, alpha, beta, not maximizing)

            ordered    = mod.order_moves(game, lm)
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


def _load_opponent_strategy(module_name: str):
    import importlib
    mod = importlib.import_module(module_name)
    return mod.playerStrategy


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULAZIONE SINGOLA PARTITA  (sequenziale, con barra mossa-per-mossa)
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_game_live(
    strategy_red,
    strategy_blue,
    prefix: str,
    seed: int = None,
) -> tuple[str, int, float]:
    """
    Gioca una partita aggiornando una barra \r mossa per mossa.
    'prefix' è il testo fisso a sinistra della barra (es. "Partita  3/10 [Candidato=Rosso]").

    Ritorna (vincitore, numero_mosse, secondi).
    """
    if seed is not None:
        random.seed(seed)

    game  = ZolaGame(size=8, first_player="Red")
    state = game.initial
    moves = 0
    t0    = time.perf_counter()

    # Riga iniziale — senza \n, sarà sovrascritta ad ogni mossa
    print(f"\r  {prefix}  [{_bar(0, MAX_MOVES, _BAR_MOVES)}]   0 mosse",
          end="", flush=True)

    for _ in range(MAX_MOVES):
        if game.is_terminal(state):
            break

        legal = game.actions(state)
        if not legal:
            state = game.pass_turn(state)
            continue

        move = (strategy_red if state.to_move == "Red" else strategy_blue)(game, state)
        if move is None or move not in legal:
            move = random.choice(legal)

        state  = game.result(state, move)
        moves += 1

        print(
            f"\r  {prefix}  [{_bar(moves, MAX_MOVES, _BAR_MOVES)}] {moves:>3} mosse",
            end="", flush=True,
        )

    elapsed = time.perf_counter() - t0

    winner = game.winner(state)
    if winner is None:
        rc, bc = state.count("Red"), state.count("Blue")
        winner = "Red" if rc > bc else ("Blue" if bc > rc else "Draw")

    return winner, moves, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# VALUTAZIONE CANDIDATO VS UN SINGOLO AVVERSARIO
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_vs_one(
    weights_cand: dict,
    opp: dict,
    n_games: int,
    opp_idx: int,
    total_opp: int,
) -> float:
    """
    Gioca n_games partite sequenziali contro un avversario.
    Ogni partita mostra una barra live, poi viene sovrascritta con la riga finale.
    Ritorna il win-rate del candidato (0..1).
    """
    opp_name      = opp["name"]
    cand_strategy = _make_candidate_strategy(weights_cand)
    opp_strategy  = _load_opponent_strategy(opp["module"])

    # ── Intestazione blocco ──────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print(f"  Candidato  vs  {opp_name}  "
          f"[{opp_idx}/{total_opp}]  peso {opp['weight']}  —  {n_games} partite")
    print(_sep("─"))

    half      = n_games // 2
    total_pts = 0.0

    # Metà partite: candidato = Red; restanti: candidato = Blue
    games_plan = (
        [(True,  i * 2)     for i in range(half)] +
        [(False, i * 2 + 1) for i in range(n_games - half)]
    )

    for game_num, (cand_is_red, seed) in enumerate(games_plan, start=1):
        role    = "Rosso" if cand_is_red else "Blu  "
        prefix  = f"Partita {game_num:>2}/{n_games}  [Candidato={role}]"

        red_s, blue_s = (
            (cand_strategy, opp_strategy) if cand_is_red
            else (opp_strategy, cand_strategy)
        )

        # Esegui partita con barra live
        winner, n_moves, elapsed = _simulate_game_live(red_s, blue_s, prefix, seed)

        # Determina risultato dal punto di vista del candidato
        if cand_is_red:
            pt     = 1.0 if winner == "Red"  else (0.5 if winner == "Draw" else 0.0)
            result = "WIN"  if winner == "Red"  else ("DRAW" if winner == "Draw" else "LOSS")
        else:
            pt     = 1.0 if winner == "Blue" else (0.5 if winner == "Draw" else 0.0)
            result = "WIN"  if winner == "Blue" else ("DRAW" if winner == "Draw" else "LOSS")

        total_pts += pt
        wr_now     = total_pts / game_num
        sym        = _SYM[result]

        # Sovrascrive la barra con la riga definitiva (aggiunge \n)
        move_bar = _bar(n_moves, MAX_MOVES, _BAR_MOVES)
        print(
            f"\r  Partita {game_num:>2}/{n_games}  [Candidato={role}]  "
            f"[{move_bar}] {n_moves:>3} mosse  "
            f"{elapsed:>5.1f}s  "
            f"{sym}   wr: {wr_now:.2f}"
        )

    # ── Riepilogo blocco ─────────────────────────────────────────────────────
    wr = total_pts / n_games
    print(_sep("─"))
    print(f"  Risultato vs {opp_name:<24}  "
          f"punti: {total_pts:.1f}/{n_games}   win-rate: {wr:.3f}")
    _print_weights(weights_cand)
    print(_sep("─"))

    return wr


# ═══════════════════════════════════════════════════════════════════════════════
# VALUTAZIONE CANDIDATO VS INTERO POOL
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_candidate_vs_pool(
    weights_cand: dict,
    n_games: int,
    it: int,
) -> tuple[float, dict]:
    """
    Valuta il candidato contro tutti gli avversari del pool in sequenza.
    Ritorna (win_rate_pesato_globale, {nome_avversario: win_rate}).
    """
    total_opp = len(OPPONENT_POOL)
    wr_by_opp = {}

    for idx, opp in enumerate(OPPONENT_POOL, start=1):
        wr = _evaluate_vs_one(weights_cand, opp, n_games, idx, total_opp)
        wr_by_opp[opp["name"]] = wr

    total_weight = sum(o["weight"] for o in OPPONENT_POOL)
    weighted_wr  = sum(
        wr_by_opp[o["name"]] * o["weight"] for o in OPPONENT_POOL
    ) / total_weight

    return weighted_wr, wr_by_opp


# ═══════════════════════════════════════════════════════════════════════════════
# PERTURBAZIONE DEI PESI
# ═══════════════════════════════════════════════════════════════════════════════

def perturb(weights: dict) -> dict:
    candidate = weights.copy()
    keys = random.sample(list(weights.keys()), k=min(PERTURB_N, len(weights)))
    for k in keys:
        candidate[k] = max(1, candidate[k] + random.randint(-PERTURB_RANGE, PERTURB_RANGE))
    return candidate


# ═══════════════════════════════════════════════════════════════════════════════
# HILL CLIMBING PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

def hill_climb():
    # ── Banner iniziale ──────────────────────────────────────────────────────
    print(_sep("═"))
    print("  Ottimizzazione pesi – Hill Climbing con pool di avversari")
    print(_sep("═"))
    print(f"  Player        : {PLAYER_MODULE}")
    print(f"  Avversari     : {len(OPPONENT_POOL)}")
    for opp in OPPONENT_POOL:
        print(f"    • {opp['name']:<28}  peso {opp['weight']}")
    print(f"  Partite/opp   : {GAMES_PER_EVAL}")
    print(f"  Iterazioni    : {MAX_ITERATIONS}")
    print(f"  Depth AB      : {SEARCH_DEPTH}")
    print(f"  Min win-rate  : {MIN_WIN_RATE:.0%}")
    print(_sep("═"))

    current = BASE_WEIGHTS.copy()
    best    = current.copy()
    best_wr = 0.0
    history = []

    for it in range(1, MAX_ITERATIONS + 1):
        t0        = time.perf_counter()
        candidate = perturb(current)

        changed = {k: (current[k], candidate[k])
                   for k in candidate if candidate[k] != current[k]}

        # ── Intestazione iterazione ──────────────────────────────────────────
        print()
        print(_sep("═"))
        print(f"  ITERAZIONE {it}/{MAX_ITERATIONS}")
        if changed:
            for k, (old, new) in changed.items():
                arrow = "▲" if new > old else "▼"
                print(f"    {arrow}  {k:<28}  {old:>4}  →  {new:>4}")
        else:
            print("    (nessuna modifica ai pesi)")
        print(_sep("═"))

        # ── Esecuzione contro il pool ────────────────────────────────────────
        wr, wr_detail = evaluate_candidate_vs_pool(candidate, GAMES_PER_EVAL, it)
        elapsed       = time.perf_counter() - t0
        accepted      = wr >= MIN_WIN_RATE
        tag           = "✓ ACCETTATO" if accepted else "✗ rifiutato"

        # ── Riepilogo iterazione ─────────────────────────────────────────────
        print()
        print(_sep("═"))
        print(f"  FINE ITERAZIONE {it}/{MAX_ITERATIONS}  │  "
              f"wr pesato = {wr:.3f}  │  {tag}  │  {elapsed:.0f}s")
        print(_sep("─"))
        print(f"  {'Avversario':<28}  {'WR':>6}  {'Peso':>5}")
        print(f"  {'─'*28}  {'─'*6}  {'─'*5}")
        for opp in OPPONENT_POOL:
            name = opp["name"]
            bar  = "●" * round(wr_detail[name] * 10)
            print(f"  {name:<28}  {wr_detail[name]:>6.3f}  {opp['weight']:>5.1f}  {bar}")
        print(f"  {'─'*28}  {'─'*6}  {'─'*5}")
        print(f"  {'WR PESATO TOTALE':<28}  {wr:>6.3f}")
        print(_sep("─"))

        if accepted:
            current = candidate
            if wr > best_wr:
                best    = candidate.copy()
                best_wr = wr
                print(f"  ★  Nuovo miglior set di pesi  (wr = {best_wr:.3f})")

        print(f"  Best globale finora: {best_wr:.3f}")
        print(_sep("═"))

        history.append({
            "iter":      it,
            "win_rate":  round(wr, 4),
            "wr_detail": {k: round(v, 4) for k, v in wr_detail.items()},
            "accepted":  accepted,
            "weights":   candidate.copy(),
        })

    # ── Risultati finali ─────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print("  OTTIMIZZAZIONE COMPLETATA")
    print(_sep("═"))
    print(f"  Miglior win-rate pesato: {best_wr:.3f}")
    print()
    print(f"  {'Peso':<28}  {'Ottimale':>8}  {'Base':>5}  {'Delta':>6}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*5}  {'─'*6}")
    for k, v in best.items():
        orig = BASE_WEIGHTS[k]
        diff = v - orig
        sign = f"+{diff}" if diff >= 0 else str(diff)
        print(f"  {k:<28}  {v:>8}  {orig:>5}  {sign:>6}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_weights.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_weights":  best,
            "best_win_rate": best_wr,
            "base_weights":  BASE_WEIGHTS,
            "pool":          [{"name": o["name"], "weight": o["weight"]}
                               for o in OPPONENT_POOL],
            "history":       history,
        }, f, indent=2)

    print()
    print(f"  Salvato in: {out_path}")
    print()
    print("  Copia nel tuo player:")
    print("  " + _sep("-"))
    for k, v in best.items():
        print(f"  {k:<28} = {v}")
    print("  " + _sep("-"))

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    hill_climb()