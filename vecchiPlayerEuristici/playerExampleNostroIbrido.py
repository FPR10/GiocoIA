import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Versione ibrida
#
# Obiettivo:
#   combinare la solidità della vecchia EU con alcune migliorie tattiche.
#
# Manteniamo:
#   • iterative deepening
#   • alpha-beta pruning
#   • timeout sicuro
#   • P5 verso angoli/periferia
#
# Aggiungiamo:
#   • gestione corretta del passaggio turno
#   • move ordering migliore
#   • P3 tattica leggera
#   • P4 per catturare pedine pericolose
#
# Evitiamo:
#   • quiescence sempre attiva
#   • P5 troppo pesante senza limite
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES             = 80    # differenza pedine residue
_W_MOBILITY           = 2     # differenza mosse legali disponibili
_W_CAPTURE_COUNT      = 10    # differenza numero catture disponibili

_W_CAPTURE_OUTER      = 5     # P1: catture verso livelli esterni
_W_MOVE_OUTER         = 3     # P2: mosse non catturanti verso esterno

_W_THREAT_PRESSURE    = 1     # P3: qualità delle catture disponibili
_W_CAPTURE_DANGEROUS  = 2     # P4: cattura pedine pericolose

_W_CORNER_SETUP       = 4     # P5: setup verso angoli/periferia


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """
    Livello massimo della scacchiera.
    Su 6x6 è 6, su 8x8 è 9.
    Gli angoli hanno il livello massimo.
    """
    return game.distance_levels[0][0]


def _capture_outer_bonus(game, captures):
    """
    P1: premia catture verso celle più esterne.

    Attenzione: in Zola una cattura non può aumentare il livello.
    Però catturare su livelli alti può mantenere la pedina in zona forte.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _move_outer_bonus(game, non_captures):
    """
    P2: premia mosse non catturanti verso l'esterno.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in non_captures)


def _capture_threat_score_from_caps(game, captures):
    """
    P3 leggera: pressione tattica tramite catture reali già disponibili.

    Non richiama game._actions_for_player: usa direttamente root_caps / opp_caps
    già calcolate in evaluate_state.

    Premia:
      • catture fatte da pedine esterne;
      • catture verso pedine più interne.
    """
    score = 0
    max_level = _max_level(game)

    for move in captures:
        (fr, fc), (tr, tc), is_capture = move

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)

        score += 3 * src_level + (max_level - dst_level + 1)

    return score


def _capture_dangerous_piece_bonus_from_moves(game, captures, opponent_moves):
    """
    P4 leggera: premia catture di pedine avversarie pericolose.

    Una pedina avversaria è considerata pericolosa se:
      • è esterna;
      • ha almeno una cattura disponibile.

    Usa opponent_moves già calcolate, quindi è più veloce.
    """
    threatening_pieces = set()

    for move in opponent_moves:
        (fr, fc), (tr, tc), is_capture = move
        if is_capture:
            threatening_pieces.add((fr, fc))

    bonus = 0

    for move in captures:
        (fr, fc), (tr, tc), is_capture = move

        target_level = _level(game, tr, tc)

        # Pedina catturata esterna: spesso più pericolosa.
        bonus += 2 * target_level

        # Se quella pedina aveva una cattura disponibile, eliminarla è molto utile.
        if (tr, tc) in threatening_pieces:
            bonus += 12

    return bonus


def _corner_setup_bonus_limited(game, state, player, non_captures):
    """
    P5 ibrida: setup verso angoli/periferia, ma limitata per non rallentare troppo.

    Idea presa dalla vecchia versione:
      una mossa non catturante è buona se dopo apre catture verso celle molto esterne.

    Miglioria:
      non analizziamo tutte le non-catture, ma solo le più promettenti,
      cioè quelle che aumentano di più il livello.
    """
    if state.to_move != player:
        return 0

    if not non_captures:
        return 0

    max_level = _max_level(game)
    threshold = max_level - 1

    # Valutiamo solo le 8 mosse non catturanti più orientate verso l'esterno.
    candidate_moves = sorted(
        non_captures,
        key=lambda m: _level(game, m[1][0], m[1][1]) - _level(game, m[0][0], m[0][1]),
        reverse=True
    )[:8]

    bonus = 0

    for move in candidate_moves:
        child = game.result(state, move)

        new_captures = [
            m for m in game._actions_for_player(child, player)
            if m[2]
        ]

        # Bonus se dopo la mossa si aprono catture verso livelli molto esterni.
        # Su 8x8: livello 8 o 9.
        # Su 6x6: livello 5 o 6.
        bonus += sum(
            1
            for (_, (tr, tc), _) in new_captures
            if _level(game, tr, tc) >= threshold
        )

    return bonus


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica
# ─────────────────────────────────────────────────────────────────────────────

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
    opp_noncaps = [m for m in opp_moves if not m[2]]

    # P1: catture verso livelli esterni
    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )

    # P2: mosse non catturanti verso esterno
    move_outer = (
        _move_outer_bonus(game, root_noncaps)
        - _move_outer_bonus(game, opp_noncaps)
    )

    # P3: pressione tattica tramite qualità delle catture disponibili
    threat_pressure = (
        _capture_threat_score_from_caps(game, root_caps)
        - _capture_threat_score_from_caps(game, opp_caps)
    )

    # P4: cattura pedine pericolose
    capture_dangerous = (
        _capture_dangerous_piece_bonus_from_moves(game, root_caps, opp_moves)
        - _capture_dangerous_piece_bonus_from_moves(game, opp_caps, root_moves)
    )

    # P5: setup verso angoli/periferia, limitato
    corner_setup = _corner_setup_bonus_limited(
        game,
        state,
        root_player,
        root_noncaps
    )

    score = (
        _W_PIECES             * (root_pieces - opp_pieces)
      + _W_MOBILITY           * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT      * (len(root_caps) - len(opp_caps))
      + _W_CAPTURE_OUTER      * cap_outer
      + _W_MOVE_OUTER         * move_outer
      + _W_THREAT_PRESSURE    * threat_pressure
      + _W_CAPTURE_DANGEROUS  * capture_dangerous
      + _W_CORNER_SETUP       * corner_setup
    )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento mosse
# ─────────────────────────────────────────────────────────────────────────────

def order_moves(game, moves):
    """
    Ordina le mosse per migliorare il pruning alpha-beta.

    Priorità:
      1. catture;
      2. catture fatte da pedine esterne;
      3. catture verso destinazioni esterne;
      4. mosse non catturanti che aumentano di più il livello.
    """
    def move_priority(move):
        (fr, fc), (tr, tc), is_capture = move

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)

        if is_capture:
            return (0, -src_level, -dst_level)

        delta = dst_level - src_level
        return (1, -delta, -dst_level)

    return sorted(moves, key=move_priority)


# ─────────────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta
# ─────────────────────────────────────────────────────────────────────────────

def _alphabeta(game, state, depth, alpha, beta, maximizing, root_player, deadline):
    if time.perf_counter() >= deadline:
        raise _Timeout()

    if game.is_terminal(state):
        return evaluate_state(game, state, root_player), None

    if depth == 0:
        return evaluate_state(game, state, root_player), None

    legal_moves = game.actions(state)

    # Regola corretta di Zola:
    # se il giocatore corrente non ha mosse, passa il turno.
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
        )

    ordered_moves = order_moves(game, legal_moves)
    best_moves = []

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
            )

            if child_value > value:
                value = child_value
                best_moves = [move]
            elif child_value == value:
                best_moves.append(move)

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
            )

            if child_value < value:
                value = child_value
                best_moves = [move]
            elif child_value == value:
                best_moves.append(move)

            beta = min(beta, value)

            if alpha >= beta:
                break

    # Deterministico: più stabile in torneo.
    return value, best_moves[0] if best_moves else None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point richiesto dalla competizione
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """
    Strategia principale.

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
            )

            if move is not None:
                best_move = move
                best_value = value

            depth += 1

        except _Timeout:
            break

    return best_move