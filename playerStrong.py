import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Strong Version
#
# Migliorie principali:
#   • Transposition Table
#   • Iterative deepening con riuso best move precedente
#   • Killer moves
#   • History heuristic
#   • Quiescence search sulle catture
#   • Gestione corretta del passaggio turno
#
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
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
# Tabelle globali
# ─────────────────────────────────────────────────────────────────────────────

TT = {}
KILLER_MOVES = {}
HISTORY = {}

TT_MAX_SIZE = 200_000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]


def _state_key(state):
    """
    Crea una chiave hashabile per lo stato.

    Non uso state.hash() perché nel vostro codice Board non sembra avere
    un metodo hash. Quindi trasformo la board in tuple di tuple.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Euristiche
# ─────────────────────────────────────────────────────────────────────────────

def _positional_value(game, state, player):
    """
    Valore posizionale assoluto: somma livello^2 delle nostre pedine
    meno livello^2 delle pedine avversarie.
    """
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
    """
    R1: premia catture verso celle più esterne.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_threat_score_from_caps(game, captures):
    """
    R2: qualità delle catture disponibili.

    Premia:
      - catture fatte da pedine esterne;
      - catture verso pedine più interne.
    """
    score = 0
    max_level = _max_level(game)

    for move in captures:
        (fr, fc), (tr, tc), _ = move

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)

        score += 3 * src_level + (max_level - dst_level + 1)

    return score


def _capture_dangerous_piece_bonus_from_moves(game, captures, opponent_moves):
    """
    R3: premia catture di pedine avversarie pericolose.

    Una pedina è pericolosa se ha almeno una cattura disponibile.
    """
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


def _corner_setup_bonus_limited(game, state, player, non_captures):
    """
    R4: setup verso angoli/periferia.

    Analizza le 8 mosse non catturanti più orientate verso l'esterno
    e premia quelle che aprono catture verso celle di livello alto.
    """
    if state.to_move != player or not non_captures:
        return 0

    max_level = _max_level(game)
    threshold = max_level - 1

    candidates = sorted(
        non_captures,
        key=lambda m: _level(game, m[1][0], m[1][1]),
        reverse=True
    )[:8]

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
    """
    Valutazione dello stato dal punto di vista di root_player.
    """
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
        _W_PIECES              * (root_pieces - opp_pieces)
      + _W_MOBILITY            * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT       * (len(root_caps) - len(opp_caps))
      + _W_POSITION            * positional
      + _W_CAPTURE_OUTER       * cap_outer
      + _W_THREAT_PRESSURE     * threat_pressure
      + _W_CAPTURE_DANGEROUS   * capture_dangerous
      + _W_CORNER_SETUP        * corner_setup
    )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Move ordering
# ─────────────────────────────────────────────────────────────────────────────

def _history_score(move):
    return HISTORY.get(move, 0)


def _killer_score(move, ply):
    killers = KILLER_MOVES.get(ply, [])

    if move in killers:
        return 1

    return 0


def order_moves(game, moves, tt_move=None, best_prev=None, ply=0):
    """
    Ordinamento mosse avanzato.

    Priorità:
      1. best move dalla transposition table;
      2. best move trovata alla profondità precedente;
      3. killer moves;
      4. history heuristic;
      5. catture;
      6. livello assoluto di destinazione;
      7. delta di livello.
    """
    moves = list(moves)

    def move_priority(move):
        (fr, fc), (tr, tc), is_capture = move

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)
        delta = dst_level - src_level

        is_tt = 1 if tt_move is not None and move == tt_move else 0
        is_prev = 1 if best_prev is not None and move == best_prev else 0
        killer = _killer_score(move, ply)
        hist = _history_score(move)

        if is_capture:
            return (
                -is_tt,
                -is_prev,
                -killer,
                -hist,
                0,
                -dst_level,
                -src_level
            )

        return (
            -is_tt,
            -is_prev,
            -killer,
            -hist,
            1,
            -dst_level,
            -delta
        )

    return sorted(moves, key=move_priority)


def _register_killer(move, ply):
    """
    Salva una killer move per il livello ply.
    """
    if move is None:
        return

    killers = KILLER_MOVES.setdefault(ply, [])

    if move in killers:
        return

    killers.insert(0, move)

    if len(killers) > 2:
        killers.pop()


def _register_history(move, depth):
    """
    Aumenta il punteggio history per le mosse che causano cutoff.
    """
    if move is None:
        return

    HISTORY[move] = HISTORY.get(move, 0) + depth * depth


# ─────────────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Quiescence search
# ─────────────────────────────────────────────────────────────────────────────

def _quiescence(game, state, alpha, beta, maximizing, root_player, deadline, q_depth=3):
    """
    Ricerca tattica sulle sole catture.

    Serve a ridurre l'horizon effect:
    invece di valutare una posizione instabile a depth 0,
    continua per qualche ply solo sulle catture.
    """
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

    ordered_captures = order_moves(
        game,
        capture_moves,
        tt_move=None,
        best_prev=None,
        ply=0
    )

    if maximizing:
        value = stand_pat
        alpha = max(alpha, value)

        if alpha >= beta:
            return value

        for move in ordered_captures:
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
# Alpha-beta con transposition table
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
        return _quiescence(
            game,
            state,
            alpha,
            beta,
            maximizing,
            root_player,
            deadline,
            q_depth=3
        ), None

    alpha_orig = alpha
    beta_orig = beta

    key = _tt_key(state, depth, root_player, maximizing)

    tt_move = None

    entry = TT.get(key)

    if entry is not None:
        stored_depth, stored_value, stored_move, flag = entry

        if stored_depth >= depth:
            tt_move = stored_move

            if flag == "EXACT":
                return stored_value, stored_move

            if flag == "LOWER":
                alpha = max(alpha, stored_value)

            elif flag == "UPPER":
                beta = min(beta, stored_value)

            if alpha >= beta:
                return stored_value, stored_move

    legal_moves = game.actions(state)

    if not legal_moves:
        passed_state = game.pass_turn(state)

        return _alphabeta(
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

    ordered_moves = order_moves(
        game,
        legal_moves,
        tt_move=tt_move,
        best_prev=best_prev,
        ply=ply
    )

    best_move = None

    if maximizing:
        value = -math.inf

        for move in ordered_moves:
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
                _register_killer(move, ply)
                _register_history(move, depth)
                break

    else:
        value = math.inf

        for move in ordered_moves:
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
                _register_killer(move, ply)
                _register_history(move, depth)
                break

    # Flag per transposition table
    if value <= alpha_orig:
        flag = "UPPER"
    elif value >= beta_orig:
        flag = "LOWER"
    else:
        flag = "EXACT"

    if len(TT) > TT_MAX_SIZE:
        TT.clear()

    TT[key] = (depth, value, best_move, flag)

    return value, best_move


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """
    Strategia principale con iterative deepening.

    Restituisce una mossa legale nel formato prodotto da game.actions(state).
    """
    legal_moves = game.actions(state)

    if not legal_moves:
        return None

    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move = random.choice(legal_moves)
    best_value = -math.inf

    depth = 1

    while True:
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

            if move is not None:
                best_move = move
                best_value = value

            depth += 1

        except _Timeout:
            break

    return best_move