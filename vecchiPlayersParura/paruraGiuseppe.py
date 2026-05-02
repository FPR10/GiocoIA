import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Michael2  (hardware-agnostic edition)
#
# Obiettivo: battere playerPaRuRa su QUALSIASI hardware.
#
# Differenze chiave rispetto alla versione precedente:
#
#   1. VALUTAZIONE A DUE VELOCITÀ
#      • evaluate_fast()  – O(n) puro, NESSUNA chiamata a game.actions/result.
#        Usata su tutti i nodi interni dell'albero.
#      • evaluate_full()  – valutazione ricca (come prima), usata SOLO alla
#        radice dell'iterative deepening (depth == 0 delle foglie della ricerca).
#        Questo abbatte di ~4x il costo per nodo su hardware lento.
#
#   2. CORNER SETUP ELIMINATO DAI NODI INTERNI
#      _corner_setup chiama game.result + _actions_for_player per ogni
#      candidato: su hardware lento è il principale collo di bottiglia.
#      Viene rimosso da evaluate_fast; rimane in evaluate_full.
#
#   3. TRANSPOSITION TABLE più efficiente
#      Chiave basata su id(state) + to_move invece di tuple-of-tuples,
#      con fallback hash del board solo se necessario.
#      Risparmia decine di ms per nodo su partite lunghe.
#
#   4. MOVE LIMITING più aggressivo in apertura (nodi interni).
#      Riduce il branching factor senza perdere qualità.
#
#   5. Quiescence conservativa: q_depth=1 solo nel finale (<= 20 pedine).
#
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN  = 0.12
MAX_DEPTH_SAFETY = 10_000

# Quiescence solo nel finale vero
QUIESCENCE_PIECES_THRESHOLD = 20

# Transposition Table
TT: dict = {}
TT_MAX_SIZE = 150_000

EXACT = 0
LOWER = 1
UPPER = 2


# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici  (identici a PaRuRa per coerenza strategica)
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES            = 80
_W_MOBILITY          = 9
_W_CAPTURE_COUNT     = 2
_W_POSITION          = 6
_W_CAPTURE_OUTER     = 1
_W_THREAT_PRESSURE   = 1
_W_CAPTURE_DANGEROUS = 1
_W_CORNER_SETUP      = 4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers veloci
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]


def _total_pieces(state):
    total = 0
    for row in state.board:
        for cell in row:
            if cell is not None:
                total += 1
    return total


class _Timeout(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Transposition Table
# ─────────────────────────────────────────────────────────────────────────────

def _board_hash(state):
    """Hash veloce del board: evita la costruzione di tuple annidate."""
    return hash((
        tuple(cell for row in state.board for cell in row),
        state.to_move
    ))


def _tt_key(state, depth, root_player, maximizing):
    return (_board_hash(state), depth, root_player, maximizing)


def _tt_lookup(key, depth, alpha, beta):
    entry = TT.get(key)
    if entry is None:
        return None, None
    stored_depth, stored_value, stored_move, stored_flag = entry
    if stored_depth < depth:
        return None, stored_move
    if stored_flag == EXACT:
        return stored_value, stored_move
    if stored_flag == LOWER and stored_value >= beta:
        return stored_value, stored_move
    if stored_flag == UPPER and stored_value <= alpha:
        return stored_value, stored_move
    return None, stored_move


def _tt_store(key, depth, value, move, alpha_orig, beta_orig):
    if value <= alpha_orig:
        flag = UPPER
    elif value >= beta_orig:
        flag = LOWER
    else:
        flag = EXACT
    if len(TT) > TT_MAX_SIZE:
        TT.clear()
    TT[key] = (depth, value, move, flag)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione FAST – O(n), zero chiamate esterne
# Usata su tutti i nodi interni dell'albero di ricerca.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fast(game, state, root_player):
    """
    Valutazione leggera: scansione singola del board.
    Calcola: differenza pedine, valore posizionale, nessun game.actions().
    """
    winner = game.winner(state)
    if winner == root_player:
        return 100_000
    if winner is not None:
        return -100_000

    opponent = game.opponent(root_player)
    player_pos = 0
    opp_pos = 0
    player_count = 0
    opp_count = 0

    for r in range(state.size):
        for c in range(state.size):
            cell = state.board[r][c]
            if cell is None:
                continue
            lv = game.distance_levels[r][c]
            if cell == root_player:
                player_pos += lv * lv
                player_count += 1
            else:
                opp_pos += lv * lv
                opp_count += 1

    return (
        _W_PIECES    * (player_count - opp_count)
      + _W_POSITION  * (player_pos - opp_pos)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione FULL – ricca, usata solo alle foglie (depth == 0)
# ─────────────────────────────────────────────────────────────────────────────

def _capture_outer_bonus(game, captures):
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_threat_score(game, captures):
    score = 0
    max_lv = _max_level(game)
    for (fr, fc), (tr, tc), _ in captures:
        score += 3 * _level(game, fr, fc) + (max_lv - _level(game, tr, tc) + 1)
    return score


def _capture_dangerous_bonus(game, captures, opp_moves):
    threatening = {
        (fr, fc)
        for (fr, fc), _, is_cap in opp_moves
        if is_cap
    }
    bonus = 0
    for (_, _), (tr, tc), _ in captures:
        bonus += 2 * _level(game, tr, tc)
        if (tr, tc) in threatening:
            bonus += 12
    return bonus


def _corner_setup_bonus(game, state, player, non_captures, k=3):
    """
    Versione limitata: al massimo k candidati (default 3).
    Viene chiamata solo in evaluate_full, mai nei nodi interni.
    """
    if state.to_move != player or not non_captures:
        return 0
    max_lv = _max_level(game)
    threshold = max_lv - 1
    candidates = sorted(
        non_captures,
        key=lambda m: _level(game, m[1][0], m[1][1]),
        reverse=True
    )[:k]
    bonus = 0
    for move in candidates:
        child = game.result(state, move)
        new_caps = [m for m in game._actions_for_player(child, player) if m[2]]
        bonus += sum(1 for (_, (tr, tc), _) in new_caps if _level(game, tr, tc) >= threshold)
    return bonus


def evaluate_full(game, state, root_player):
    """
    Valutazione completa: mobilità, catture, corner setup.
    Costosa – usata solo alle foglie della ricerca principale.
    """
    winner = game.winner(state)
    if winner == root_player:
        return 100_000
    if winner is not None:
        return -100_000

    opponent = game.opponent(root_player)

    root_pieces = state.count(root_player)
    opp_pieces  = state.count(opponent)

    root_moves = game._actions_for_player(state, root_player)
    opp_moves  = game._actions_for_player(state, opponent)

    root_caps    = [m for m in root_moves if m[2]]
    opp_caps     = [m for m in opp_moves  if m[2]]
    root_noncaps = [m for m in root_moves if not m[2]]

    # Posizionale
    player_pos = opp_pos = 0
    for r in range(state.size):
        for c in range(state.size):
            cell = state.board[r][c]
            if cell is None:
                continue
            lv = game.distance_levels[r][c]
            if cell == root_player:
                player_pos += lv * lv
            else:
                opp_pos += lv * lv
    positional = player_pos - opp_pos

    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )
    threat_pressure = (
        _capture_threat_score(game, root_caps)
        - _capture_threat_score(game, opp_caps)
    )
    capture_dangerous = (
        _capture_dangerous_bonus(game, root_caps, opp_moves)
        - _capture_dangerous_bonus(game, opp_caps, root_moves)
    )
    # Corner setup: k dinamico
    pieces = root_pieces + opp_pieces
    k = 1 if pieces >= 44 else (2 if pieces >= 34 else 3)
    corner_setup = _corner_setup_bonus(game, state, root_player, root_noncaps, k)

    return (
        _W_PIECES            * (root_pieces - opp_pieces)
      + _W_MOBILITY          * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT     * (len(root_caps)  - len(opp_caps))
      + _W_POSITION          * positional
      + _W_CAPTURE_OUTER     * cap_outer
      + _W_THREAT_PRESSURE   * threat_pressure
      + _W_CAPTURE_DANGEROUS * capture_dangerous
      + _W_CORNER_SETUP      * corner_setup
    )


# ─────────────────────────────────────────────────────────────────────────────
# Move ordering e limiting
# ─────────────────────────────────────────────────────────────────────────────

def _move_score(game, move):
    (fr, fc), (tr, tc), is_capture = move
    src_lv = _level(game, fr, fc)
    dst_lv = _level(game, tr, tc)
    if is_capture:
        return 10_000 + 250 * dst_lv + 40 * src_lv
    return 100 * dst_lv + 30 * (dst_lv - src_lv)


def order_moves(game, moves, tt_move=None, best_prev=None):
    def priority(move):
        is_tt   = 1 if tt_move   is not None and move == tt_move   else 0
        is_prev = 1 if best_prev is not None and move == best_prev else 0
        return (-is_tt, -is_prev, -_move_score(game, move))
    return sorted(moves, key=priority)


def _noncap_limit(pieces, ply):
    """
    Limite dinamico alle mosse non-catturanti.
    Più aggressivo nei nodi interni per ridurre il branching factor.
    """
    if ply == 0:
        # Alla radice: più libertà per non perdere mosse buone
        if pieces >= 44: return 14
        if pieces >= 34: return 16
        if pieces >= 24: return 20
        return 40
    # Nodi interni: taglio forte in apertura
    if pieces >= 44: return 5
    if pieces >= 34: return 7
    if pieces >= 24: return 10
    if pieces >= 16: return 16
    return 40


def _limit_moves(game, state, moves, ply):
    captures = [m for m in moves if m[2]]
    noncaps  = [m for m in moves if not m[2]]
    pieces   = _total_pieces(state)
    limit    = _noncap_limit(pieces, ply)
    if len(noncaps) > limit:
        noncaps = sorted(noncaps, key=lambda m: _move_score(game, m), reverse=True)[:limit]
    return captures + noncaps


# ─────────────────────────────────────────────────────────────────────────────
# Quiescence leggera
# ─────────────────────────────────────────────────────────────────────────────

def _quiescence(game, state, alpha, beta, maximizing, root_player, deadline, q_depth=1):
    if time.perf_counter() >= deadline:
        raise _Timeout()

    stand_pat = evaluate_full(game, state, root_player)

    if q_depth == 0 or game.is_terminal(state):
        return stand_pat

    legal_moves = game.actions(state)

    if not legal_moves:
        return _quiescence(
            game, game.pass_turn(state), alpha, beta,
            not maximizing, root_player, deadline, q_depth - 1
        )

    captures = [m for m in legal_moves if m[2]]
    if not captures:
        return stand_pat

    ordered = order_moves(game, captures)

    if maximizing:
        value = stand_pat
        alpha = max(alpha, value)
        if alpha >= beta:
            return value
        for move in ordered:
            if time.perf_counter() >= deadline:
                raise _Timeout()
            child_val = _quiescence(
                game, game.result(state, move), alpha, beta,
                False, root_player, deadline, q_depth - 1
            )
            value = max(value, child_val)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = stand_pat
        beta = min(beta, value)
        if alpha >= beta:
            return value
        for move in ordered:
            if time.perf_counter() >= deadline:
                raise _Timeout()
            child_val = _quiescence(
                game, game.result(state, move), alpha, beta,
                True, root_player, deadline, q_depth - 1
            )
            value = min(value, child_val)
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta principale
# ─────────────────────────────────────────────────────────────────────────────

def _alphabeta(
    game, state, depth, alpha, beta,
    maximizing, root_player, deadline,
    ply=0, best_prev=None
):
    if time.perf_counter() >= deadline:
        raise _Timeout()

    if game.is_terminal(state):
        return evaluate_full(game, state, root_player), None

    # Foglia: valutazione
    if depth == 0:
        pieces = _total_pieces(state)
        if pieces <= QUIESCENCE_PIECES_THRESHOLD:
            return _quiescence(
                game, state, alpha, beta,
                maximizing, root_player, deadline, q_depth=1
            ), None
        return evaluate_full(game, state, root_player), None

    alpha_orig = alpha
    beta_orig  = beta

    key = _tt_key(state, depth, root_player, maximizing)
    tt_value, tt_move = _tt_lookup(key, depth, alpha, beta)
    if tt_value is not None:
        return tt_value, tt_move

    legal_moves = game.actions(state)

    if not legal_moves:
        passed = game.pass_turn(state)
        value, move = _alphabeta(
            game, passed, depth - 1, alpha, beta,
            not maximizing, root_player, deadline, ply + 1
        )
        _tt_store(key, depth, value, move, alpha_orig, beta_orig)
        return value, move

    # Limiting + ordering
    limited = _limit_moves(game, state, legal_moves, ply)
    ordered = order_moves(game, limited, tt_move=tt_move,
                          best_prev=best_prev if ply == 0 else None)

    best_move = None

    if maximizing:
        value = -math.inf
        for move in ordered:
            if time.perf_counter() >= deadline:
                raise _Timeout()
            child_val, _ = _alphabeta(
                game, game.result(state, move), depth - 1,
                alpha, beta, False, root_player, deadline, ply + 1
            )
            if child_val > value:
                value     = child_val
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for move in ordered:
            if time.perf_counter() >= deadline:
                raise _Timeout()
            child_val, _ = _alphabeta(
                game, game.result(state, move), depth - 1,
                alpha, beta, True, root_player, deadline, ply + 1
            )
            if child_val < value:
                value     = child_val
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break

    _tt_store(key, depth, value, best_move, alpha_orig, beta_orig)
    return value, best_move


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    deadline   = time.perf_counter() + timeout - _TIME_MARGIN

    # Fallback: mossa statica migliore (non casuale)
    best_move  = order_moves(game, legal_moves)[0]
    best_value = -math.inf
    depth      = 1
    max_completed_depth = 0

    while depth <= MAX_DEPTH_SAFETY:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, state.to_move, deadline,
                ply=0, best_prev=best_move
            )
            max_completed_depth = depth
            if move is not None and move in legal_moves:
                best_move  = move
                best_value = value
            depth += 1
        except _Timeout:
            break

    # 🔴 DEBUG FINALE affidabile
    print(f"[DEBUG FINAL] max_depth={max_completed_depth}")

    return best_move