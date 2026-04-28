import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Strategia con iterative deepening e alpha-beta pruning
#
# La valutazione posizionale usa il livello ASSOLUTO di destinazione,
# pesato esponenzialmente: livello 9 vale molto più di livello 8,
# indipendentemente da dove la pedina si trovava prima.
# L'ordinamento delle mosse usa prima il livello assoluto di destinazione
# (decrescente), poi il delta come criterio secondario.
#
# Caratteristiche principali:
#   • iterative deepening
#   • alpha-beta pruning con timeout sicuro
#   • gestione corretta del passaggio turno
#   • R1: valutazione tattica (qualità catture)
#   • R2: cattura pedine pericolose
#
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES             = 80    # differenza pedine residue
_W_MOBILITY           = 9     # differenza mosse legali disponibili
_W_CAPTURE_COUNT      = 2    # differenza numero catture disponibili

_W_POSITION           = 6     # valore posizionale assoluto (livello esponenziale)
_W_CAPTURE_OUTER      = 1     # R1: catture verso livelli esterni
_W_THREAT_PRESSURE    = 1     # R2: qualità delle catture disponibili
_W_CAPTURE_DANGEROUS  = 1     # R3: cattura pedine pericolose
_W_CORNER_SETUP       = 4     # R4: setup verso angoli/periferia


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """Livello massimo della scacchiera. Su 8x8 è 9, su 6x6 è 6."""
    return game.distance_levels[0][0]


def _positional_value(game, state, player):
    """
    Valore posizionale assoluto: somma esponenziale dei livelli occupati.

    Usare livello^2 invece di livello lineare garantisce che raggiungere
    un angolo (livello 9) valga molto più che stare su livello 8,
    anche se il delta è identico.

    Il vantaggio posizionale è la differenza tra i valori dei due giocatori.
    """
    opponent = game.opponent(player)
    player_val = 0
    opp_val    = 0

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
    """R1: premia catture verso celle più esterne (livello assoluto)."""
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_threat_score_from_caps(game, captures):
    """
    R2: pressione tattica tramite qualità delle catture disponibili.
    Premia catture fatte da pedine esterne e verso pedine più interne.
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
    for (fr, fc), (tr, tc), _ in captures:
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

    # Ordiniamo per livello ASSOLUTO di destinazione
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


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_state(game, state, root_player):
    """Valuta lo stato dal punto di vista di root_player."""
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

    # Valore posizionale assoluto (livello² sommato su tutte le pedine)
    positional = _positional_value(game, state, root_player)

    # R1: catture verso livelli esterni
    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )

    # R2: pressione tattica tramite qualità delle catture
    threat_pressure = (
        _capture_threat_score_from_caps(game, root_caps)
        - _capture_threat_score_from_caps(game, opp_caps)
    )

    # R3: cattura pedine pericolose
    capture_dangerous = (
        _capture_dangerous_piece_bonus_from_moves(game, root_caps, opp_moves)
        - _capture_dangerous_piece_bonus_from_moves(game, opp_caps, root_moves)
    )

    # R4: setup verso angoli/periferia
    corner_setup = _corner_setup_bonus_limited(
        game, state, root_player, root_noncaps
    )

    score = (
        _W_PIECES           * (root_pieces - opp_pieces)
      + _W_MOBILITY         * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT    * (len(root_caps)  - len(opp_caps))
      + _W_POSITION         * positional
      + _W_CAPTURE_OUTER    * cap_outer
      + _W_THREAT_PRESSURE  * threat_pressure
      + _W_CAPTURE_DANGEROUS * capture_dangerous
      + _W_CORNER_SETUP     * corner_setup
    )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento mosse
# ─────────────────────────────────────────────────────────────────────────────

def order_moves(game, moves):
    """
    Ordina le mosse per migliorare il pruning alpha-beta.

    Priorità:
      1. catture (is_capture=True) prima delle non-catture;
      2. tra le catture: prima quelle verso destinazioni di livello più alto;
      3. tra le non-catture: prima quelle verso destinazioni di livello più alto
         (livello assoluto come criterio primario, delta come secondario).

    Questo garantisce che una pedina preferisca spostarsi verso l'angolo
    a livello 9 piuttosto che su una cella a livello 8, indipendentemente
    dal delta.
    """
    def move_priority(move):
        (fr, fc), (tr, tc), is_capture = move
        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)
        delta     = dst_level - src_level

        if is_capture:
            # Le catture vengono prima; tra esse, privilegia le destinazioni
            # di livello assoluto più alto (pedine nemiche più periferiche).
            return (0, -dst_level, -src_level)

        # Mosse non catturanti: livello assoluto di destinazione come
        # criterio PRIMARIO, delta come criterio secondario.
        return (1, -dst_level, -delta)

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

    # Regola corretta di Zola: nessuna mossa → passa il turno.
    if not legal_moves:
        passed_state = game.pass_turn(state)
        return _alphabeta(
            game, passed_state, depth - 1,
            alpha, beta, not maximizing,
            root_player, deadline,
        )

    ordered_moves = order_moves(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered_moves:
            child = game.result(state, move)
            child_value, _ = _alphabeta(
                game, child, depth - 1,
                alpha, beta, False,
                root_player, deadline,
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
                game, child, depth - 1,
                alpha, beta, True,
                root_player, deadline,
            )
            if child_value < value:
                value = child_value
                best_moves = [move]
            elif child_value == value:
                best_moves.append(move)
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, best_moves[0] if best_moves else None


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

    deadline  = time.perf_counter() + timeout - _TIME_MARGIN
    best_move = random.choice(legal_moves)
    depth     = 1

    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, state.to_move, deadline,
            )
            if move is not None:
                best_move = move
            depth += 1
        except _Timeout:
            break

    return best_move