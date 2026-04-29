import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - B Plus
#
# Obiettivo:
#   migliorare il giocatore B originale senza distruggere la sua strategia.
#
# Mantiene:
#   • stessi pesi del giocatore B;
#   • stesso stile posizionale verso esterno/angoli;
#   • stesso setup verso angoli/periferia.
#
# Aggiunge:
#   • Transposition Table;
#   • riuso della best move precedente;
#   • corner_setup meno costoso: 4 candidati invece di 8;
#   • soft move limit solo quando ci sono troppe non-catture;
#   • quiescence search leggera sulle catture.
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.18

DEBUG_DEPTH = True


# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici: lasciati uguali al giocatore B originale
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
# Transposition Table
# ─────────────────────────────────────────────────────────────────────────────

TT = {}
TT_MAX_SIZE = 150_000

# Soft limit: non taglia quasi mai all'inizio, ma evita esplosioni.
_MAX_NONCAPTURES = 18


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """Livello massimo della scacchiera. Su 8x8 è 9, su 6x6 è 6."""
    return game.distance_levels[0][0]


def _state_key(state):
    """
    Chiave hashabile dello stato.

    Include:
      - disposizione della board;
      - giocatore di turno.
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

    Mantiene il punto forte del giocatore B:
    una pedina su livello 9 vale molto più di una su livello 8.
    """
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
    """
    R1: premia catture verso celle più esterne.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_threat_score_from_caps(game, captures):
    """
    R2: pressione tattica tramite qualità delle catture disponibili.

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

    Differenza rispetto al giocatore B:
    prima valutava 8 candidate non-catturanti;
    qui ne valuta 4.

    Così mantiene il comportamento strategico,
    ma l'euristica costa molto meno.
    """
    if state.to_move != player or not non_captures:
        return 0

    max_level = _max_level(game)
    threshold = max_level - 1

    candidates = sorted(
        non_captures,
        key=lambda m: _level(game, m[1][0], m[1][1]),
        reverse=True
    )[:4]

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
    Valuta lo stato dal punto di vista di root_player.
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
# Soft move limiting
# ─────────────────────────────────────────────────────────────────────────────

def _limit_moves_soft(game, moves, max_noncaptures=_MAX_NONCAPTURES):
    """
    Limitazione morbida del branching.

    Tiene sempre tutte le catture.
    Taglia le non-catture solo se sono troppe.

    Questo evita il problema della versione Deep:
    non elimina aggressivamente mosse strategiche.
    """
    captures = [m for m in moves if m[2]]
    noncaps = [m for m in moves if not m[2]]

    if len(noncaps) <= max_noncaptures:
        return moves

    noncaps = sorted(
        noncaps,
        key=lambda m: (
            _level(game, m[1][0], m[1][1]),
            _level(game, m[1][0], m[1][1]) - _level(game, m[0][0], m[0][1])
        ),
        reverse=True
    )[:max_noncaptures]

    return captures + noncaps


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento mosse
# ─────────────────────────────────────────────────────────────────────────────

def order_moves(game, moves, best_prev=None, tt_move=None):
    """
    Ordinamento mosse.

    Mantiene l'ordinamento del giocatore B:
      - catture prima;
      - destinazione con livello assoluto alto;
      - delta come criterio secondario.

    Aggiunge:
      - mossa dalla TT;
      - best move della profondità precedente.
    """
    def move_priority(move):
        (fr, fc), (tr, tc), is_capture = move

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)
        delta = dst_level - src_level

        is_tt = 1 if tt_move is not None and move == tt_move else 0
        is_prev = 1 if best_prev is not None and move == best_prev else 0

        if is_capture:
            return (
                -is_tt,
                -is_prev,
                0,
                -dst_level,
                -src_level
            )

        return (
            -is_tt,
            -is_prev,
            1,
            -dst_level,
            -delta
        )

    return sorted(moves, key=move_priority)


# ─────────────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Quiescence leggera
# ─────────────────────────────────────────────────────────────────────────────

def _quiescence(game, state, alpha, beta, maximizing, root_player, deadline, q_depth=1):
    """
    Quiescence search leggera.

    Non vogliamo renderla pesante.
    Serve solo a evitare di fermarsi esattamente prima di una cattura immediata.
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
        best_prev=None,
        tt_move=None
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
# Alpha-beta + TT
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
            q_depth=1
        ), None

    key = _tt_key(state, depth, root_player, maximizing)

    tt_move = None
    entry = TT.get(key)

    if entry is not None:
        stored_value, stored_move = entry
        tt_move = stored_move
        return stored_value, stored_move

    legal_moves = game.actions(state)

    if not legal_moves:
        passed_state = game.pass_turn(state)

        result = _alphabeta(
            game,
            passed_state,
            depth - 1,
            alpha,
            beta,
            not maximizing,
            root_player,
            deadline,
            best_prev=None
        )

        if len(TT) > TT_MAX_SIZE:
            TT.clear()

        TT[key] = result
        return result

    legal_moves = _limit_moves_soft(game, legal_moves)

    ordered_moves = order_moves(
        game,
        legal_moves,
        best_prev=best_prev,
        tt_move=tt_move
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
                best_prev=None
            )

            if child_value < value:
                value = child_value
                best_move = move

            beta = min(beta, value)

            if alpha >= beta:
                break

    result = (value, best_move)

    if len(TT) > TT_MAX_SIZE:
        TT.clear()

    TT[key] = result

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """
    Strategia principale con iterative deepening.

    Questa versione prova a migliorare il giocatore B:
      - senza cambiare troppo la sua euristica;
      - aggiungendo ottimizzazioni leggere.
    """
    legal_moves = game.actions(state)

    if not legal_moves:
        return None

    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move = random.choice(legal_moves)
    best_value = -math.inf

    depth = 1
    reached_depth = 0

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
            f"[B Plus] {state.to_move} → profondità raggiunta: "
            f"{reached_depth} | mossa scelta: {best_move}"
        )

    return best_move