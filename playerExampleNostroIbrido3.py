import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Versione ibrida v3
#
# Fix rispetto a v2:
#
#   v2 usava _positional_value con livello² sommato su tutte le pedine,
#   producendo valori nell'ordine dei migliaia che soffocavano i segnali
#   di cattura (_W_CAPTURE_COUNT, _W_CAPTURE_DANGEROUS diventavano irrilevanti).
#   Inoltre era scomparsa la logica "mi sposto verso l'esterno perché
#   l'avversario è più esterno di me" (era _W_MOVE_OUTER nella v1).
#
#   Soluzione:
#   1. _positional_value ora calcola solo il VANTAGGIO RELATIVO di livello
#      medio tra i due giocatori (scala piccola, comparabile agli altri pesi).
#   2. Reintrodotto _W_MOVE_OUTER: premia mosse non catturanti verso celle
#      di livello assoluto alto — questo è il segnale di reattività posizionale.
#   3. Pesi ribilanciati in modo che catture e posizione abbiano scala simile.
#   4. Mantenuto il fix di order_moves (dst_level assoluto come criterio
#      primario per le non-catture).
#
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES             = 80    # differenza pedine residue
_W_MOBILITY           = 2     # differenza mosse legali disponibili
_W_CAPTURE_COUNT      = 12    # differenza numero catture disponibili (alzato)

_W_POSITION           = 3     # vantaggio relativo di livello medio (scala piccola)
_W_CAPTURE_OUTER      = 5     # P1: catture verso livelli esterni
_W_MOVE_OUTER         = 3     # P2: mosse non catturanti verso esterno (reattività)
_W_THREAT_PRESSURE    = 1     # P3: qualità delle catture disponibili
_W_CAPTURE_DANGEROUS  = 4     # P4: cattura pedine pericolose (alzato)
_W_CORNER_SETUP       = 4     # P5: setup verso angoli/periferia


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """Livello massimo della scacchiera. Su 8x8 è 9, su 6x6 è 6."""
    return game.distance_levels[0][0]


def _positional_advantage(game, state, player):
    """
    Vantaggio posizionale RELATIVO: differenza tra il livello medio
    delle pedine di player e quello dell'avversario, scalata x10.

    Scala tipica: [-50, +50] su 8x8 — comparabile agli altri termini
    senza schiacciarli. Cattura esattamente "l'avversario è più esterno
    di me" senza confondere numero di pedine e qualità di posizione.
    """
    opponent = game.opponent(player)
    player_sum, player_cnt = 0, 0
    opp_sum,    opp_cnt    = 0, 0

    for r in range(state.size):
        for c in range(state.size):
            cell = state.board[r][c]
            if cell is None:
                continue
            lv = _level(game, r, c)
            if cell == player:
                player_sum += lv
                player_cnt += 1
            else:
                opp_sum += lv
                opp_cnt += 1

    player_avg = player_sum / player_cnt if player_cnt else 0
    opp_avg    = opp_sum    / opp_cnt    if opp_cnt    else 0

    return int((player_avg - opp_avg) * 10)


def _move_outer_bonus(game, non_captures):
    """
    P2: premia mosse non catturanti verso celle di livello assoluto alto.

    Reattività posizionale: "mi sposto verso l'esterno perché l'avversario
    occupa già celle periferiche". Usa livello ASSOLUTO di destinazione.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in non_captures)


def _capture_outer_bonus(game, captures):
    """P1: premia catture verso celle più esterne (livello assoluto)."""
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_threat_score_from_caps(game, captures):
    """
    P3: pressione tattica tramite qualità delle catture disponibili.
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
    P4: premia catture di pedine avversarie pericolose.
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
    P5: setup verso angoli/periferia.
    Analizza le 8 mosse non catturanti più orientate verso l'esterno
    e premia quelle che aprono catture verso celle di livello alto.
    """
    if state.to_move != player or not non_captures:
        return 0

    max_level = _max_level(game)
    threshold = max_level - 1

    # Ordiniamo per livello ASSOLUTO di destinazione (fix del bug)
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
    opp_noncaps  = [m for m in opp_moves  if not m[2]]

    # Vantaggio posizionale relativo (livello medio, scala x10)
    positional = _positional_advantage(game, state, root_player)

    # P1: catture verso livelli esterni
    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )

    # P2: mosse non catturanti verso esterno (reattività posizionale)
    move_outer = (
        _move_outer_bonus(game, root_noncaps)
        - _move_outer_bonus(game, opp_noncaps)
    )

    # P3: pressione tattica tramite qualità delle catture
    threat_pressure = (
        _capture_threat_score_from_caps(game, root_caps)
        - _capture_threat_score_from_caps(game, opp_caps)
    )

    # P4: cattura pedine pericolose
    capture_dangerous = (
        _capture_dangerous_piece_bonus_from_moves(game, root_caps, opp_moves)
        - _capture_dangerous_piece_bonus_from_moves(game, opp_caps, root_moves)
    )

    # P5: setup verso angoli/periferia
    corner_setup = _corner_setup_bonus_limited(
        game, state, root_player, root_noncaps
    )

    score = (
        _W_PIECES            * (root_pieces - opp_pieces)
      + _W_MOBILITY          * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT     * (len(root_caps)  - len(opp_caps))
      + _W_POSITION          * positional
      + _W_CAPTURE_OUTER     * cap_outer
      + _W_MOVE_OUTER        * move_outer
      + _W_THREAT_PRESSURE   * threat_pressure
      + _W_CAPTURE_DANGEROUS * capture_dangerous
      + _W_CORNER_SETUP      * corner_setup
    )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento mosse  ← FIX PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def order_moves(game, moves):
    """
    Ordina le mosse per migliorare il pruning alpha-beta.

    Priorità:
      1. catture (is_capture=True) prima delle non-catture;
      2. tra le catture: prima quelle verso destinazioni di livello più alto;
      3. tra le non-catture: prima quelle verso destinazioni di livello più alto
         (FIX: era ordinato per delta = dst-src, ora usiamo dst assoluto).
         Il delta è usato solo come criterio secondario.

    Questo risolve il bug per cui una pedina a livello 6 preferiva spostarsi
    sulla cella liberata a livello 8 (delta=2) invece che sull'angolo a
    livello 9 (delta=3): entrambe avevano delta positivo, ma il livello
    assoluto dell'angolo è maggiore e ora vince come criterio primario.
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
# Entry point richiesto dalla competizione
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