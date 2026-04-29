import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - B Plus Adaptive Depth
#
# Obiettivo:
#   mantenere la strategia forte del giocatore B Plus,
#   ma permettere profondità maggiore già in apertura.
#
# Filosofia:
#   • NON saltare depth 1,2,3: iterative deepening normale;
#   • rendere l'apertura più leggera per arrivare a depth 4/5;
#   • non tagliare aggressivamente la mossa alla radice;
#   • limitare di più i nodi profondi in apertura;
#   • quiescence disattivata in apertura, attiva nel medio/finale;
#   • corner setup dinamico: leggero in apertura, più ricco dopo;
#   • Transposition Table con flag EXACT / LOWER / UPPER.
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15
DEBUG_DEPTH = True

MAX_DEPTH_SAFETY = 10_000

# Quiescence solo quando la partita è abbastanza avanzata.
QUIESCENCE_PIECES_THRESHOLD = 24

# Transposition Table
TT = {}
TT_MAX_SIZE = 200_000

EXACT = "EXACT"
LOWER = "LOWER"
UPPER = "UPPER"


# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
# Manteniamo lo stile del giocatore B originale.
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES             = 80
_W_MOBILITY           = 9
_W_CAPTURE_COUNT      = 2

_W_POSITION           = 6
_W_CAPTURE_OUTER      = 1
_W_THREAT_PRESSURE    = 1
_W_CAPTURE_DANGEROUS  = 1
_W_CORNER_SETUP       = 4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _state_key(state):
    return (
        tuple(tuple(row) for row in state.board),
        state.to_move
    )


def _tt_key(state, depth, root_player, maximizing):
    return (
        _state_key(state),
        depth,
        root_player,
        maximizing
    )


class _Timeout(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Euristiche
# ─────────────────────────────────────────────────────────────────────────────

def _positional_value(game, state, player):
    opponent = game.opponent(player)

    player_val = 0
    opp_val = 0

    for r in range(state.size):
        for c in range(state.size):
            cell = state.board[r][c]

            if cell is None:
                continue

            lv = _level(game, r, c)

            if cell == player:
                player_val += lv * lv
            else:
                opp_val += lv * lv

    return player_val - opp_val


def _capture_outer_bonus(game, captures):
    return sum(
        _level(game, tr, tc)
        for (_, (tr, tc), _) in captures
    )


def _capture_threat_score_from_caps(game, captures):
    score = 0
    max_level = _max_level(game)

    for move in captures:
        (fr, fc), (tr, tc), _ = move

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)

        score += 3 * src_level + (max_level - dst_level + 1)

    return score


def _capture_dangerous_piece_bonus_from_moves(game, captures, opponent_moves):
    threatening_pieces = {
        (fr, fc)
        for (fr, fc), _, is_cap in opponent_moves
        if is_cap
    }

    bonus = 0

    for (_, _), (tr, tc), _ in captures:
        target_level = _level(game, tr, tc)

        bonus += 2 * target_level

        if (tr, tc) in threatening_pieces:
            bonus += 12

    return bonus


def _corner_setup_candidate_count(state):
    """
    Corner setup dinamico.

    In apertura era troppo costoso calcolare 4/8 candidate a ogni foglia.
    Però eliminarlo del tutto rende il player più debole.
    Qui lo teniamo, ma molto leggero all'inizio.
    """
    pieces = _total_pieces(state)

    if pieces >= 44:
        return 1
    if pieces >= 34:
        return 2
    if pieces >= 24:
        return 3
    return 4


def _corner_setup_bonus_limited(game, state, player, non_captures):
    if state.to_move != player or not non_captures:
        return 0

    max_level = _max_level(game)
    threshold = max_level - 1

    k = _corner_setup_candidate_count(state)

    candidates = sorted(
        non_captures,
        key=lambda m: _level(game, m[1][0], m[1][1]),
        reverse=True
    )[:k]

    bonus = 0

    for move in candidates:
        child = game.result(state, move)

        new_caps = [
            m for m in game._actions_for_player(child, player)
            if m[2]
        ]

        bonus += sum(
            1
            for (_, (tr, tc), _) in new_caps
            if _level(game, tr, tc) >= threshold
        )

    return bonus


def evaluate_state(game, state, root_player):
    winner = game.winner(state)

    if winner == root_player:
        return 100_000

    if winner is not None:
        return -100_000

    opponent = game.opponent(root_player)

    root_pieces = state.count(root_player)
    opp_pieces = state.count(opponent)

    root_moves = game._actions_for_player(state, root_player)
    opp_moves = game._actions_for_player(state, opponent)

    root_caps = [m for m in root_moves if m[2]]
    opp_caps = [m for m in opp_moves if m[2]]

    root_noncaps = [m for m in root_moves if not m[2]]

    positional = _positional_value(game, state, root_player)

    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )

    threat_pressure = (
        _capture_threat_score_from_caps(game, root_caps)
        - _capture_threat_score_from_caps(game, opp_caps)
    )

    capture_dangerous = (
        _capture_dangerous_piece_bonus_from_moves(game, root_caps, opp_moves)
        - _capture_dangerous_piece_bonus_from_moves(game, opp_caps, root_moves)
    )

    corner_setup = _corner_setup_bonus_limited(
        game,
        state,
        root_player,
        root_noncaps
    )

    score = (
        _W_PIECES               * (root_pieces - opp_pieces)
      + _W_MOBILITY             * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT        * (len(root_caps) - len(opp_caps))
      + _W_POSITION             * positional
      + _W_CAPTURE_OUTER        * cap_outer
      + _W_THREAT_PRESSURE      * threat_pressure
      + _W_CAPTURE_DANGEROUS    * capture_dangerous
      + _W_CORNER_SETUP         * corner_setup
    )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Move limiting dinamico
# ─────────────────────────────────────────────────────────────────────────────

def _dynamic_noncap_limit(state, ply):
    """
    Limite dinamico sulle mosse non catturanti.

    Importante:
      - alla radice non tagliamo troppo, perché lì scegliamo davvero la mossa;
      - nei nodi profondi tagliamo di più per arrivare a profondità 4/5;
      - nel finale lasciamo molte più mosse.
    """
    pieces = _total_pieces(state)

    # Radice: lascia più libertà.
    if ply == 0:
        if pieces >= 44:
            return 16
        if pieces >= 34:
            return 18
        if pieces >= 24:
            return 22
        return 40

    # Nodi interni: più selettivi in apertura.
    if pieces >= 44:
        return 8
    if pieces >= 34:
        return 10
    if pieces >= 24:
        return 14
    if pieces >= 16:
        return 20
    return 40


def _move_static_score(game, move):
    """
    Score statico per ordinare le mosse.

    Mantiene lo stile del player originale:
      - catture prima;
      - destinazione più esterna;
      - sorgente esterna;
      - delta verso esterno per le non-catture.
    """
    (fr, fc), (tr, tc), is_capture = move

    src_level = _level(game, fr, fc)
    dst_level = _level(game, tr, tc)
    delta = dst_level - src_level

    if is_capture:
        return (
            10_000
            + 250 * dst_level
            + 40 * src_level
        )

    return (
        100 * dst_level
        + 30 * delta
    )


def _limit_moves_soft(game, state, moves, ply):
    """
    Tiene sempre tutte le catture.
    Limita solo le non-catture, e in modo diverso tra radice/nodi interni.
    """
    captures = [m for m in moves if m[2]]
    noncaps = [m for m in moves if not m[2]]

    max_noncaptures = _dynamic_noncap_limit(state, ply)

    if len(noncaps) <= max_noncaptures:
        return moves

    noncaps = sorted(
        noncaps,
        key=lambda m: _move_static_score(game, m),
        reverse=True
    )[:max_noncaptures]

    return captures + noncaps


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento mosse
# ─────────────────────────────────────────────────────────────────────────────

def order_moves(game, moves, best_prev=None, tt_move=None):
    def move_priority(move):
        is_tt = 1 if tt_move is not None and move == tt_move else 0
        is_prev = 1 if best_prev is not None and move == best_prev else 0

        return (
            -is_tt,
            -is_prev,
            -_move_static_score(game, move)
        )

    return sorted(moves, key=move_priority)


# ─────────────────────────────────────────────────────────────────────────────
# Quiescence leggera
# ─────────────────────────────────────────────────────────────────────────────

def _quiescence(game, state, alpha, beta, maximizing, root_player, deadline, q_depth=1):
    if time.perf_counter() >= deadline:
        raise _Timeout()

    stand_pat = evaluate_state(game, state, root_player)

    if q_depth == 0 or game.is_terminal(state):
        return stand_pat

    legal_moves = game.actions(state)

    if not legal_moves:
        passed_state = game.pass_turn(state)

        return _quiescence(
            game,
            passed_state,
            alpha,
            beta,
            not maximizing,
            root_player,
            deadline,
            q_depth - 1
        )

    capture_moves = [m for m in legal_moves if m[2]]

    if not capture_moves:
        return stand_pat

    ordered_captures = order_moves(game, capture_moves)

    if maximizing:
        value = stand_pat
        alpha = max(alpha, value)

        if alpha >= beta:
            return value

        for move in ordered_captures:
            if time.perf_counter() >= deadline:
                raise _Timeout()

            child = game.result(state, move)

            child_value = _quiescence(
                game,
                child,
                alpha,
                beta,
                False,
                root_player,
                deadline,
                q_depth - 1
            )

            value = max(value, child_value)
            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    else:
        value = stand_pat
        beta = min(beta, value)

        if alpha >= beta:
            return value

        for move in ordered_captures:
            if time.perf_counter() >= deadline:
                raise _Timeout()

            child = game.result(state, move)

            child_value = _quiescence(
                game,
                child,
                alpha,
                beta,
                True,
                root_player,
                deadline,
                q_depth - 1
            )

            value = min(value, child_value)
            beta = min(beta, value)

            if alpha >= beta:
                break

        return value


# ─────────────────────────────────────────────────────────────────────────────
# Transposition Table con flag
# ─────────────────────────────────────────────────────────────────────────────

def _tt_lookup(key, depth, alpha, beta):
    entry = TT.get(key)

    if entry is None:
        return None, None

    stored_depth = entry["depth"]
    stored_value = entry["value"]
    stored_move = entry["move"]
    stored_flag = entry["flag"]

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

    TT[key] = {
        "depth": depth,
        "value": value,
        "move": move,
        "flag": flag
    }


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta
# ─────────────────────────────────────────────────────────────────────────────

def _alphabeta(
    game,
    state,
    depth,
    alpha,
    beta,
    maximizing,
    root_player,
    deadline,
    ply=0,
    best_prev=None
):
    if time.perf_counter() >= deadline:
        raise _Timeout()

    if game.is_terminal(state):
        return evaluate_state(game, state, root_player), None

    if depth == 0:
        # In apertura la quiescence rallenta e spesso non serve.
        # Nel medio/finale aiuta a non fermarsi prima di una cattura.
        if _total_pieces(state) <= QUIESCENCE_PIECES_THRESHOLD:
            return _quiescence(
                game,
                state,
                alpha,
                beta,
                maximizing,
                root_player,
                deadline,
                q_depth=1
            ), None

        return evaluate_state(game, state, root_player), None

    alpha_orig = alpha
    beta_orig = beta

    key = _tt_key(state, depth, root_player, maximizing)

    tt_value, tt_move = _tt_lookup(key, depth, alpha, beta)

    if tt_value is not None:
        return tt_value, tt_move

    legal_moves = game.actions(state)

    if not legal_moves:
        passed_state = game.pass_turn(state)

        value, move = _alphabeta(
            game,
            passed_state,
            depth - 1,
            alpha,
            beta,
            not maximizing,
            root_player,
            deadline,
            ply + 1,
            best_prev=None
        )

        _tt_store(key, depth, value, move, alpha_orig, beta_orig)
        return value, move

    legal_moves = _limit_moves_soft(game, state, legal_moves, ply)

    ordered_moves = order_moves(
        game,
        legal_moves,
        best_prev=best_prev if ply == 0 else None,
        tt_move=tt_move
    )

    best_move = None

    if maximizing:
        value = -math.inf

        for move in ordered_moves:
            if time.perf_counter() >= deadline:
                raise _Timeout()

            child = game.result(state, move)

            child_value, _ = _alphabeta(
                game,
                child,
                depth - 1,
                alpha,
                beta,
                False,
                root_player,
                deadline,
                ply + 1,
                best_prev=None
            )

            if child_value > value:
                value = child_value
                best_move = move

            alpha = max(alpha, value)

            if alpha >= beta:
                break

    else:
        value = math.inf

        for move in ordered_moves:
            if time.perf_counter() >= deadline:
                raise _Timeout()

            child = game.result(state, move)

            child_value, _ = _alphabeta(
                game,
                child,
                depth - 1,
                alpha,
                beta,
                True,
                root_player,
                deadline,
                ply + 1,
                best_prev=None
            )

            if child_value < value:
                value = child_value
                best_move = move

            beta = min(beta, value)

            if alpha >= beta:
                break

    _tt_store(key, depth, value, best_move, alpha_orig, beta_orig)

    return value, best_move


# ─────────────────────────────────────────────────────────────────────────────
# Fallback non casuale
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_move(game, state, legal_moves):
    ordered = order_moves(game, legal_moves)
    return ordered[0] if ordered else random.choice(legal_moves)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    legal_moves = game.actions(state)

    if not legal_moves:
        return None

    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move = _fallback_move(game, state, legal_moves)
    best_value = -math.inf

    reached_depth = 0
    depth = 1

    while depth <= MAX_DEPTH_SAFETY:
        if time.perf_counter() >= deadline:
            break

        try:
            value, move = _alphabeta(
                game,
                state,
                depth,
                -math.inf,
                math.inf,
                True,
                state.to_move,
                deadline,
                ply=0,
                best_prev=best_move
            )

            if move is not None and move in legal_moves:
                best_move = move
                best_value = value

            reached_depth = depth
            depth += 1

        except _Timeout:
            break

    if DEBUG_DEPTH:
        print(
            f"[B Plus Adaptive Depth] {state.to_move} → profondità raggiunta: "
            f"{reached_depth} | valore: {best_value:.1f} | mossa scelta: {best_move}"
        )

    return best_move