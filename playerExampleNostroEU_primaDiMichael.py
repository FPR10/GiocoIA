import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Iterative-deepening alpha-beta e timeout
#
# Struttura:
#   • evaluate_state   – funzione euristica multi-feature
#   • order_moves      – ordinamento delle mosse (catture esterne prima)
#   • alphabeta        – alpha-beta con move ordering
#   • playerStrategy   – iterative deepening con guardia al tempo
#
# Principi strategici incorporati:
#   P1. Preferire catture verso l'esterno (livello destinazione alto)
#   P2. Nelle mosse non-catturanti preferire destinazioni più esterne
#   P3. Pressione nelle zone esterne dove ci sono pedine avversarie esterne
#   P4. Catturare la pedina avversaria più interna (livello basso) = più pericolosa
#   P5. Setup angolo: bonus se una mossa non-catturante apre una cattura al turno dopo
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15   # secondi di margine di sicurezza prima del timeout


# ─────────────────────────────────────────────────────────────────────────────
# Costanti di peso dell'euristica  (modifica qui per bilanciare)
# ─────────────────────────────────────────────────────────────────────────────
_W_PIECES          = 50   # differenza pedine residue
_W_MOBILITY        =  2   # differenza mosse legali disponibili
_W_CAPTURE_COUNT   =  5   # differenza numero catture disponibili
_W_CAPTURE_OUTER   =  6   # P1: bonus catture verso l'esterno (nostro)
_W_CAPTURE_INNER   =  0   # P4: bonus per catturare pedine avversarie più interne
_W_MOVE_OUTER      =  4   # P2: bonus mosse non-catturanti verso l'esterno
_W_OUTER_PRESSURE  =  4   # P3: nostre pedine esterne vicino a zone con avversari esterni
_W_CORNER_SETUP    =  8   # P5: setup in due mosse verso angolo+cattura


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """Livello massimo presente nella scacchiera (angoli)."""
    s = game.size - 1
    return game.distance_levels[0][0]   # angolo = massima distanza


def _capture_outer_bonus(game, captures):
    """P1 – somma dei livelli di destinazione delle catture disponibili.

    Catture verso celle più esterne (livello alto) sono premiate.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_inner_target_bonus(game, captures):
    """P4 – premia le catture di pedine avversarie molto interne (livello basso).

    Una pedina avversaria interna è più pericolosa: catturarla vale di più.
    Usiamo (max_level - livello_destinazione) come contributo: più la pedina
    catturata è interna, più alto il bonus.
    """
    ml = _max_level(game)
    return sum(ml - _level(game, tr, tc) for (_, (tr, tc), flag) in captures if flag)


def _move_outer_bonus(game, non_captures):
    """P2 – somma dei livelli di destinazione delle mosse non-catturanti.

    Mosse che portano più all'esterno (livello alto) sono premiate.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in non_captures)


def _outer_pressure_bonus(game, state, player):
    """P3 – bonus per nostre pedine esterne vicino a zone con pedine avversarie esterne.

    Per ogni nostra pedina con livello >= soglia, contiamo le pedine avversarie
    adiacenti (nelle 8 direzioni) con livello >= soglia. Più sono, più siamo
    in pressione nella zona esterna.
    """
    opponent = game.opponent(player)
    threshold = _max_level(game) - 2   # zona "esterna": ultimi 2 livelli
    bonus = 0
    for r in range(state.size):
        for c in range(state.size):
            if state.board[r][c] != player:
                continue
            if _level(game, r, c) < threshold:
                continue
            for dr, dc in game.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if game.in_bounds(nr, nc) and state.board[nr][nc] == opponent:
                    if _level(game, nr, nc) >= threshold:
                        bonus += 1
    return bonus


def _corner_setup_bonus(game, state, player, non_captures):
    """P5 – premia mosse non-catturanti che, dopo lo spostamento, abilitano
    almeno una cattura verso una cella molto esterna (angolo o periferia).

    Simuliamo la mossa e contiamo le catture verso zone esterne che diventano
    disponibili. Viene calcolato solo se è il turno del giocatore corrente.
    """
    if state.to_move != player:
        return 0
    ml = _max_level(game)
    threshold = ml - 1   # solo angoli e celle adiacenti agli angoli
    bonus = 0
    for move in non_captures:
        child = game.result(state, move)
        new_captures = [m for m in game._actions_for_player(child, player) if m[2]]
        bonus += sum(1 for (_, (tr, tc), _) in new_captures
                     if _level(game, tr, tc) >= threshold)
    return bonus


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica principale
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_state(game, state, root_player):
    """Valutazione euristica dal punto di vista di root_player.

    Feature combinate:
      • differenza pedine residue                          (P base)
      • differenza mobilità totale                         (P base)
      • differenza catture disponibili                     (P base)
      • catture esterne proprie vs avversarie              (P1)
      • catture di pedine interne avversarie               (P4)
      • mosse non-catturanti verso esterno                 (P2)
      • pressione esterna (nostre pedine vicino avversari) (P3)
      • setup angolo: mosse che aprono catture in periferia(P5)
    """
    winner = game.winner(state)
    if winner == root_player:
        return 100_000
    if winner is not None:
        return -100_000

    opponent = game.opponent(root_player)

    # ── conteggi base ────────────────────────────────────────────────────────
    root_pieces = state.count(root_player)
    opp_pieces  = state.count(opponent)

    root_moves    = game._actions_for_player(state, root_player)
    opp_moves     = game._actions_for_player(state, opponent)

    root_caps     = [m for m in root_moves if m[2]]
    opp_caps      = [m for m in opp_moves  if m[2]]
    root_noncaps  = [m for m in root_moves if not m[2]]
    opp_noncaps   = [m for m in opp_moves  if not m[2]]

    # ── feature strategiche ──────────────────────────────────────────────────
    # P1: catture verso esterno
    cap_outer = _capture_outer_bonus(game, root_caps) - _capture_outer_bonus(game, opp_caps)

    # P4: catturare pedine più interne (più pericolose)
    cap_inner_target = (
        _capture_inner_target_bonus(game, root_caps)
        - _capture_inner_target_bonus(game, opp_caps)
    )

    # P2: mosse non-catturanti verso esterno
    move_outer = (
        _move_outer_bonus(game, root_noncaps)
        - _move_outer_bonus(game, opp_noncaps)
    )

    # P3: pressione nelle zone esterne
    outer_pressure = (
        _outer_pressure_bonus(game, state, root_player)
        - _outer_pressure_bonus(game, state, opponent)
    )

    # P5: setup angolo (solo per noi, costoso da calcolare per entrambi)
    corner_setup = _corner_setup_bonus(game, state, root_player, root_noncaps)

    score = (
        _W_PIECES         * (root_pieces - opp_pieces)
      + _W_MOBILITY       * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT  * (len(root_caps)  - len(opp_caps))
      + _W_CAPTURE_OUTER  * cap_outer
      + _W_CAPTURE_INNER  * cap_inner_target
      + _W_MOVE_OUTER     * move_outer
      + _W_OUTER_PRESSURE * outer_pressure
      + _W_CORNER_SETUP   * corner_setup
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento delle mosse (move ordering)
# ─────────────────────────────────────────────────────────────────────────────

def order_moves(game, moves):
    """Ordina le mosse per massimizzare l'efficacia del pruning alpha-beta.

    Priorità:
      1. Catture verso l'esterno (livello destinazione alto)  – P1
      2. Catture verso l'interno (cattura pedine pericolose)  – P4
      3. Mosse non-catturanti verso l'esterno                 – P2
      4. Resto (shuffle per varietà)
    """
    def move_priority(m):
        (fr, fc), (tr, tc), is_cap = m
        dest_level = _level(game, tr, tc)
        src_level  = _level(game, fr, fc)
        if is_cap:
            # cattura esterna: destinazione lontana dal centro → priorità alta
            return (0, -dest_level)
        else:
            # mossa non-catturante: più esterna è la destinazione, meglio
            return (1, -dest_level)

    caps     = sorted([m for m in moves if m[2]],     key=move_priority)
    noncaps  = sorted([m for m in moves if not m[2]], key=move_priority)
    return caps + noncaps


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con time check
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    """Eccezione interna lanciata quando il tempo è scaduto."""


def _alphabeta(game, state, depth, alpha, beta, maximizing, root_player, deadline):
    """Alpha-beta ricorsivo.

    Lancia _Timeout se si supera 'deadline' (timestamp assoluto).
    """
    if time.perf_counter() >= deadline:
        raise _Timeout()

    legal_moves = game.actions(state)

    if depth == 0 or game.is_terminal(state) or not legal_moves:
        return evaluate_state(game, state, root_player), None

    ordered = order_moves(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(game, child, depth - 1, alpha, beta, False, root_player, deadline)

            if child_val > value:
                value = child_val
                best_moves = [move]
            elif child_val == value:
                best_moves.append(move)

            alpha = max(alpha, value)
            if alpha >= beta:
                break

    else:
        value = math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(game, child, depth - 1, alpha, beta, True, root_player, deadline)

            if child_val < value:
                value = child_val
                best_moves = [move]
            elif child_val == value:
                best_moves.append(move)

            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, (random.choice(best_moves) if best_moves else None)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point: iterative deepening con guardia al tempo
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """Strategia con iterative-deepening alpha-beta e timeout sicuro.

    Esegue ricerche a profondità crescente (1, 2, 3, …).  Quando sta per
    scadere il tempo concesso (timeout - _TIME_MARGIN), restituisce la
    migliore mossa trovata fino a quel momento.  Se nessuna ricerca completa
    è mai terminata, ricade su una mossa casuale (non dovrebbe accadere con
    timeout ≥ 1 s, ma è la rete di sicurezza richiesta).
    """
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move  = random.choice(legal_moves)   # fallback sicuro
    best_value = -math.inf

    depth = 1
    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, state.to_move,
                deadline,
            )
            if move is not None:
                best_move  = move
                best_value = value
            depth += 1
        except _Timeout:
            # Tempo scaduto durante questa iterazione: usa il risultato precedente
            break

    return best_move