import math
import random
import time

# ═══════════════════════════════════════════════════════════════════════════════
#  Zola – Strategia Alpha-Beta + Monte Carlo pesato + Iterative Deepening
#
#  QUATTRO REGOLE FONDAMENTALI
#  ───────────────────────────
#  R1 · SETTORE DI CASA
#        Preferire mosse che partono dal proprio lato della scacchiera.
#        (Blue: quadranti alto-sx e basso-dx; Red: alto-dx e basso-sx)
#
#  R2 · CATTURA PIÙ ESTERNA CON PEZZO PIÙ ESTERNO
#        Tra le catture disponibili:
#          1. preferire quella con ring_dest più basso (bordo = ring 0)
#          2. a parità di ring_dest, preferire src con ring più basso
#          3. a ulteriore parità, preferire dest nel settore avversario
#        Non-catturanti fortemente penalizzate se esistono catture valide.
#
#  R3 · PEDINA PIÙ ESTERNA CATTURA PRIMA / MUOVERSI VERSO L'ESTERNO
#        - La pedina con ring più basso (più esterna) ha priorità di cattura.
#        - Per le non-catturanti: preferire destinazioni con ring più basso
#          (più verso il bordo); evitare di muovere pedine interne se ce ne
#          sono di esterne inattive.
#
#  R4 · RETRODIFESA
#        Se una pedina avversaria è a ring più basso (più esterna) di una
#        nostra pedina, quest'ultima deve spostarsi verso l'esterno
#        (mossa non-catturante verso ring inferiore) per aggirarla.
#
#  METRICA DI DISTANZA
#  ───────────────────
#  Si usa il ring-level:  ring(r,c) = min(r, c, size-1-r, size-1-c)
#  ring 0 = bordo esterno (massima priorità), ring 3 = centro (8×8).
#  BASSO ring → PIÙ ESTERNO → PREFERITO.
#  Nota: nel gioco i livelli euclidei sono usati per le regole di movimento
#  (mossa legale), ma la strategia ragiona in ring-level.
#
#  ARCHITETTURA
#  ────────────
#  • Le foglie dell'albero alpha-beta sono valutate con K playout MC pesati.
#  • Nessuna euristica numerica: solo risultati dei playout (+1/-1/0).
#  • I playout usano campionamento pesato dalle R1–R4.
#  • Iterative deepening: la profondità cresce finché il tempo lo permette;
#    i rollout per foglia si dimezzano ad ogni livello per rispettare i 3 s.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Temporizzazione ────────────────────────────────────────────────────────────
_TIME_MARGIN      = 0.12   # secondi di margine prima del timeout

# ── Monte Carlo ────────────────────────────────────────────────────────────────
_MC_ROLLOUTS_BASE = 8      # playout per foglia a depth=1 dell'alpha-beta
_MC_ROLLOUT_DEPTH = 35     # profondità massima di ogni playout

# ── Pesi campionamento (moltiplicatori) ────────────────────────────────────────

# R1 – settore di casa
_W_HOME           = 2.5    # boost se la pedina parte dal proprio lato

# R2/R3 – catture
_W_CAP_BASE       = 6.0    # boost base per ogni cattura
_W_CAP_RING_BONUS = 2.0    # ulteriore boost per ogni ring di vantaggio sulla cattura
                            # migliore (dest ring = min tra le catture disponibili)
_W_CAP_SRC_BONUS  = 1.5    # ulteriore boost per ogni ring di vantaggio del pezzo
                            # attaccante (src ring = min tra gli attaccanti disponibili)
_W_CAP_OPP_SECTOR = 1.8    # boost se dest è nel settore avversario (tie-break R2.3)
_W_NONCAP_PEN     = 0.15   # penalità non-catturanti quando esistono catture

# R3 – non-catturanti (quando non ci sono catture)
_W_MOVE_RING      = 2.2    # boost per ogni ring di vantaggio della destinazione
                            # (ring più basso = più esterno = meglio)
_W_MOVE_OUTER_SRC = 1.6    # boost se src ha ring minore di qualsiasi pedina
                            # nostra che non si sta muovendo (pedine esterne prima)

# R4 – retrodifesa
_W_RETRO          = 3.0    # boost alla mossa non-catturante verso l'esterno
                            # quando esiste un avversario a ring inferiore (più esterno)
                            # rispetto alla pedina che si muove


# ═══════════════════════════════════════════════════════════════════════════════
#  Sezione A – Geometria: ring-level e settori
# ═══════════════════════════════════════════════════════════════════════════════

def _ring(size, r, c):
    """Ring-level: 0 = bordo esterno, size//2-1 = centro.

    Corrisponde ai "cerchi concentrici" 8×8, 7×7, 6×6 … visti dall'esterno.
    Più basso il ring, più la cella è verso il bordo (= preferita).
    """
    return min(r, c, size - 1 - r, size - 1 - c)


def _is_in_home(size, player, r, c):
    """True se (r,c) è nel settore di casa di player.

    Blue ha pedine iniziali sulle celle con (r+c) pari → angoli (0,0) e (s-1,s-1)
      → settori: quadrante alto-sx  ∪  quadrante basso-dx
    Red ha pedine iniziali sulle celle con (r+c) dispari → angoli (0,s-1) e (s-1,0)
      → settori: quadrante alto-dx  ∪  quadrante basso-sx
    """
    mid = size // 2
    if player == "Blue":
        return (r < mid and c < mid) or (r >= mid and c >= mid)
    else:
        return (r < mid and c >= mid) or (r >= mid and c < mid)


def _is_in_opp_sector(size, player, r, c):
    opp = "Blue" if player == "Red" else "Red"
    return _is_in_home(size, opp, r, c)


# ═══════════════════════════════════════════════════════════════════════════════
#  Sezione B – Calcolo dei pesi per il campionamento MC
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_weights(game, state, moves, player):
    """Assegna un peso a ogni mossa secondo R1–R4.

    Restituisce una lista parallela a `moves` di float > 0.
    """
    if not moves:
        return []

    size     = state.size
    opponent = "Blue" if player == "Red" else "Red"

    # Pre-calcola ring-level per non ripeterlo
    def rng(r, c):
        return _ring(size, r, c)

    # ── Separa catture e non-catturanti ──────────────────────────────────────
    caps    = [m for m in moves if     m[2]]
    noncaps = [m for m in moves if not m[2]]

    # ── Statistiche sulle catture disponibili ────────────────────────────────
    # Ring minimo (= più esterno) tra le destinazioni delle catture
    min_cap_dest_ring = min((rng(*m[1]) for m in caps), default=None)
    # Ring minimo tra i sorgenti delle catture
    min_cap_src_ring  = min((rng(*m[0]) for m in caps), default=None)

    # ── Statistiche sulle nostre pedine in campo (per R3 "pedine esterne prima")
    all_our_rings = [
        rng(r, c)
        for r in range(size)
        for c in range(size)
        if state.board[r][c] == player
    ]
    min_our_ring = min(all_our_rings) if all_our_rings else 0  # ring più esterno

    # ── Pedine avversarie più esterne di alcune nostre (per R4 retrodifesa) ──
    opp_rings = [
        rng(r, c)
        for r in range(size)
        for c in range(size)
        if state.board[r][c] == opponent
    ]
    min_opp_ring = min(opp_rings) if opp_rings else size

    weights = []

    for m in moves:
        fr, fc  = m[0]
        tr, tc  = m[1]
        is_cap  = m[2]
        src_ring = rng(fr, fc)
        dst_ring = rng(tr, tc)
        w = 1.0

        # ── R1: settore di casa ───────────────────────────────────────────────
        if _is_in_home(size, player, fr, fc):
            w *= _W_HOME

        if is_cap:
            # ── R2 + R3: catture ─────────────────────────────────────────────

            # Boost base cattura
            w *= _W_CAP_BASE

            # R2.1: più la destinazione è esterna (ring basso), meglio.
            # Bonus per ogni ring di vantaggio rispetto alla cattura peggiore
            # disponibile; la cattura con ring_dest minimo riceve il massimo.
            ring_dest_advantage = (dst_ring - min_cap_dest_ring)
            # ring_dest_advantage = 0 per la cattura più esterna (migliore)
            # Trasformiamo: meno ring_dest, più boost.
            # Usiamo: boost = BASE ^ (max_possible - advantage) → inversione
            # Più semplicemente: penalità proporzionale allo svantaggio.
            if ring_dest_advantage > 0:
                w *= (1.0 / (_W_CAP_RING_BONUS ** ring_dest_advantage))
            else:
                w *= _W_CAP_RING_BONUS  # questa è la cattura più esterna

            # R2.2 / R3: pedina più esterna cattura per prima.
            src_ring_advantage = (src_ring - min_cap_src_ring)
            if src_ring_advantage > 0:
                w *= (1.0 / (_W_CAP_SRC_BONUS ** src_ring_advantage))
            else:
                w *= _W_CAP_SRC_BONUS  # questa è la pedina più esterna

            # R2.3: tie-break settore avversario
            if _is_in_opp_sector(size, player, tr, tc):
                w *= _W_CAP_OPP_SECTOR

        else:
            # ── Non-catturante ────────────────────────────────────────────────

            # Se esistono catture: forte penalità alle non-catturanti
            if caps:
                w *= _W_NONCAP_PEN
            else:
                # ── R4: retrodifesa ──────────────────────────────────────────
                # Se un avversario è più esterno (ring inferiore) di questa
                # pedina, è urgente muoversi verso l'esterno per aggirarlo.
                # Una mossa non-catturante porta verso livelli euclidei più
                # alti (= più esterni nel sistema del gioco), ma il ring
                # della destinazione è necessariamente ≥ src (le non-catturanti
                # del gioco vanno verso livello euclideo maggiore, che può
                # corrispondere a ring uguale o inferiore a seconda della cella).
                # Premiamo le non-catturanti che riducono il ring (= verso bordo).
                if min_opp_ring < src_ring:
                    # C'è un avversario più esterno: retrodifesa attiva
                    if dst_ring < src_ring:
                        # La mossa va verso il bordo: ottima
                        w *= _W_RETRO
                    elif dst_ring == src_ring:
                        # La mossa mantiene il ring: accettabile
                        w *= _W_RETRO * 0.5

                # ── R3: preferire destinazioni più esterne (ring più basso) ──
                ring_improvement = src_ring - dst_ring  # positivo = più esterno
                if ring_improvement > 0:
                    w *= _W_MOVE_RING ** ring_improvement
                elif ring_improvement == 0:
                    w *= 1.0   # neutro
                else:
                    w *= 0.5   # verso l'interno: penalità leggera

                # ── R3: pedine esterne si muovono per prime ───────────────────
                # Se questa pedina è tra le più esterne, boost aggiuntivo.
                if src_ring == min_our_ring:
                    w *= _W_MOVE_OUTER_SRC

        weights.append(w)

    # Fallback: se tutti i pesi collassano (non dovrebbe accadere)
    total = sum(weights)
    if total <= 0.0:
        return [1.0] * len(moves)
    return weights


def _weighted_choice(moves, weights):
    """Campionamento categorico pesato."""
    total = sum(weights)
    r     = random.random() * total
    cum   = 0.0
    for m, w in zip(moves, weights):
        cum += w
        if r <= cum:
            return m
    return moves[-1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Sezione C – Playout Monte Carlo puro
# ═══════════════════════════════════════════════════════════════════════════════

def _rollout(game, state, root_player):
    """Singolo playout MC pesato fino a terminazione o profondità massima.

    Restituisce:
      +1.0  se root_player vince
      -1.0  se perde
      ±0.5  stima leggera dal conteggio pedine se si raggiunge _MC_ROLLOUT_DEPTH
    """
    cur   = state
    depth = 0

    while depth < _MC_ROLLOUT_DEPTH:
        winner = game.winner(cur)
        if winner is not None:
            return 1.0 if winner == root_player else -1.0

        moves = game.actions(cur)
        if not moves:
            cur   = game.pass_turn(cur)
            depth += 1
            continue

        w    = _compute_weights(game, cur, moves, cur.to_move)
        move = _weighted_choice(moves, w)
        cur  = game.result(cur, move)
        depth += 1

    # Stima dal conteggio pedine residue
    rp = cur.count(root_player)
    op = cur.count(game.opponent(root_player))
    if rp > op:
        return 0.5
    if op > rp:
        return -0.5
    return 0.0


def _mc_eval(game, state, root_player, n):
    """Media di n playout pesati, valore in [-1.0, +1.0]."""
    if n <= 0:
        return 0.0
    return sum(_rollout(game, state, root_player) for _ in range(n)) / n


# ═══════════════════════════════════════════════════════════════════════════════
#  Sezione D – Alpha-Beta con foglie MC pure
# ═══════════════════════════════════════════════════════════════════════════════

class _Timeout(Exception):
    pass


def _order_moves(game, moves):
    """Move ordering per efficacia del pruning alpha-beta.

    Ordine: catture verso ring più basso (più esterno) → altre catture → non-catturanti.
    All'interno delle catture, prima quelle con src a ring più basso.
    """
    size = game.size

    def key(m):
        fr, fc = m[0]
        tr, tc = m[1]
        is_cap = m[2]
        if is_cap:
            return (0, _ring(size, tr, tc), _ring(size, fr, fc))
        else:
            return (1, _ring(size, tr, tc), _ring(size, fr, fc))

    return sorted(moves, key=key)


def _alphabeta(
    game, state, depth, alpha, beta,
    maximizing, root_player, deadline, n_rollouts,
):
    """Alpha-beta minimax con foglie valutate via MC puro.

    Lancia _Timeout se viene superato `deadline`.
    """
    if time.perf_counter() >= deadline:
        raise _Timeout()

    # ── Terminale ────────────────────────────────────────────────────────────
    winner = game.winner(state)
    if winner is not None:
        return (1.0 if winner == root_player else -1.0), None

    legal = game.actions(state)
    if not legal:
        # Nessuna mossa legale: turno saltato (non è terminale)
        passed = game.pass_turn(state)
        return _alphabeta(
            game, passed, depth, alpha, beta,
            maximizing, root_player, deadline, n_rollouts,
        )

    # ── Foglia: valutazione MC pura ──────────────────────────────────────────
    if depth == 0:
        return _mc_eval(game, state, root_player, n_rollouts), None

    ordered    = _order_moves(game, legal)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            if time.perf_counter() >= deadline:
                raise _Timeout()
            child   = game.result(state, move)
            cval, _ = _alphabeta(
                game, child, depth - 1, alpha, beta,
                False, root_player, deadline, n_rollouts,
            )
            if cval > value:
                value      = cval
                best_moves = [move]
            elif cval == value:
                best_moves.append(move)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for move in ordered:
            if time.perf_counter() >= deadline:
                raise _Timeout()
            child   = game.result(state, move)
            cval, _ = _alphabeta(
                game, child, depth - 1, alpha, beta,
                True, root_player, deadline, n_rollouts,
            )
            if cval < value:
                value      = cval
                best_moves = [move]
            elif cval == value:
                best_moves.append(move)
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, (random.choice(best_moves) if best_moves else None)


# ═══════════════════════════════════════════════════════════════════════════════
#  Sezione E – playerStrategy
# ═══════════════════════════════════════════════════════════════════════════════

def playerStrategy(game, state, timeout=3):
    """Strategia Alpha-Beta + Monte Carlo pesato con Iterative Deepening.

    Regole applicate nei playout MC e nel move ordering dell'alpha-beta:
      R1 – Preferire mosse dal settore di casa del giocatore.
      R2 – Tra le catture: prima quella con destinazione ring più basso
           (più esterna), effettuata dal pezzo con ring più basso (più
           esterno); tie-break: settore avversario.
      R3 – Senza catture: muoversi verso ring più basso (verso il bordo);
           pedine più esterne hanno priorità di movimento.
      R4 – Retrodifesa: se un avversario è a ring più basso di una nostra
           pedina, quella pedina si muove verso l'esterno per aggirarlo.

    La profondità alpha-beta cresce da 1 iterativamente. Il numero di
    rollout per foglia si dimezza a ogni livello di profondità aggiuntivo
    (depth=1: 8 rollout, depth=2: 4, depth=3: 2, depth≥4: 1) per restare
    nei 3 secondi disponibili. Fallback sicuro: se anche depth=1 va in
    timeout, la mossa è scelta con campionamento pesato direttamente dalla
    radice.
    """
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    # Fallback pesato immediato (garanzia contro timeout totale)
    w0        = _compute_weights(game, state, legal_moves, state.to_move)
    best_move = _weighted_choice(legal_moves, w0)

    depth = 1
    while time.perf_counter() < deadline:
        # Rollout per foglia: si dimezza ad ogni livello di profondità
        n_rollouts = max(1, _MC_ROLLOUTS_BASE >> max(0, depth - 1))

        try:
            _, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, state.to_move,
                deadline, n_rollouts,
            )
            if move is not None:
                best_move = move
            depth += 1
        except _Timeout:
            break

    return best_move