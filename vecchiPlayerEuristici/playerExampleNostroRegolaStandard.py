import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Versione ibrida con comportamento standard agli angoli
#
# Manteniamo:
#   • iterative deepening
#   • alpha-beta pruning
#   • timeout sicuro
#   • tutte le euristiche P1-P5
#
# Aggiungiamo:
#   • CASO A: pedina nostra nell'angolo → sequenza di catture garantita
#   • CASO B: pedina avversaria nell'angolo → contro-manovra diagonale
#   • priorità Caso B > Caso A
#   • una volta avviata la sequenza, va completata prima di tornare all'euristica
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Stato persistente della sequenza angolo (condiviso tra le chiamate)
# ─────────────────────────────────────────────────────────────────────────────

_corner_sequence_state = {
    "active": False,       # stiamo eseguendo una sequenza?
    "corner": None,        # (r, c) angolo coinvolto
    "case": None,          # "A" o "B"
    "step": 0,             # passo corrente nella sequenza
    "extra": None,         # dati accessori (es. posizione pezzo diagonale nel caso B)
}


def _reset_sequence():
    _corner_sequence_state["active"] = False
    _corner_sequence_state["corner"] = None
    _corner_sequence_state["case"] = None
    _corner_sequence_state["step"] = 0
    _corner_sequence_state["extra"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Geometria degli angoli
# ─────────────────────────────────────────────────────────────────────────────

def _get_corners(size):
    """Restituisce le 4 celle angolo della scacchiera."""
    n = size - 1
    return [(0, 0), (0, n), (n, 0), (n, n)]


def _adjacent_to_corner(corner, size):
    """
    Restituisce le 2 celle ortogonalmente adiacenti all'angolo
    (non la diagonale, ma le due "fianco").
    In Zola agli angoli si muove solo ortogonalmente verso l'interno.
    """
    r, c = corner
    n = size - 1
    cells = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr <= n and 0 <= nc <= n:
            cells.append((nr, nc))
    return cells


def _diagonal_of_corner(corner, size):
    """
    Cella diagonalmente adiacente all'angolo (unica, sul bordo interno).
    """
    r, c = corner
    n = size - 1
    dr = 1 if r == 0 else -1
    dc = 1 if c == 0 else -1
    nr, nc = r + dr, c + dc
    if 0 <= nr <= n and 0 <= nc <= n:
        return (nr, nc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Rilevamento delle situazioni agli angoli
# ─────────────────────────────────────────────────────────────────────────────

def _detect_case_B(game, state, player):
    """
    CASO B: pedina avversaria nell'angolo.

    Condizione di attivazione:
      - L'angolo contiene una pedina avversaria.
      - L'avversario ha appena mosso dall'angolo verso una delle celle adiacenti
        (cioè last_move è una cattura dall'angolo verso una nostra pedina adiacente).
      - Noi abbiamo una pedina sulla diagonale dell'angolo, da cui possiamo catturare
        la pedina appena mossa.

    Restituisce lista di (corner, diagonal_pos, target_pos) oppure [].
    """
    opponent = game.opponent(player)
    size = state.size
    corners = _get_corners(size)
    results = []

    last = state.last_move
    if not last or last.get("type") != "move":
        return results

    last_from = last.get("from")
    last_to = last.get("to")
    last_is_capture = last.get("capture", False)

    if not last_is_capture:
        return results

    for corner in corners:
        # L'avversario deve essere ora nell'angolo
        if state.board[corner[0]][corner[1]] != opponent:
            continue

        adjacents = _adjacent_to_corner(corner, size)
        diagonal = _diagonal_of_corner(corner, size)

        if diagonal is None:
            continue

        # L'avversario ha catturato DA questo angolo (last_from == corner)
        if last_from != corner:
            continue

        # La pedina mossa è ora in last_to
        if last_to not in adjacents:
            continue

        # Noi dobbiamo avere una pedina sulla diagonale
        if state.board[diagonal[0]][diagonal[1]] != player:
            continue

        # La nostra pedina diagonale deve poter catturare last_to
        can_capture = False
        for move in game._actions_for_player(state, player):
            if move[0] == diagonal and move[1] == last_to and move[2]:
                can_capture = True
                break

        if can_capture:
            results.append({
                "corner": corner,
                "diagonal": diagonal,
                "target": last_to,
            })

    return results


def _detect_case_A(game, state, player):
    """
    CASO A: pedina nostra nell'angolo.

    Condizione di attivazione:
      - Noi abbiamo una pedina in un angolo.
      - In quell'angolo ci sono pedine avversarie nelle celle adiacenti.
      - La pedina nostra nell'angolo può catturare almeno una di quelle avversarie.

    Restituisce lista di dict con corner e catture disponibili.
    """
    size = state.size
    corners = _get_corners(size)
    results = []

    for corner in corners:
        if state.board[corner[0]][corner[1]] != player:
            continue

        adjacents = _adjacent_to_corner(corner, size)
        captures_from_corner = []

        for move in game._actions_for_player(state, player):
            if move[0] == corner and move[2] and move[1] in adjacents:
                captures_from_corner.append(move)

        if captures_from_corner:
            results.append({
                "corner": corner,
                "captures": captures_from_corner,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Esecuzione della sequenza Caso B
# ─────────────────────────────────────────────────────────────────────────────

def _execute_case_B(game, state, player, seq):
    """
    Gestisce la sequenza del CASO B in base al passo corrente.

    step 0 → (non raggiunto qui, si salta all'attivazione)
    step 1 → cattura la pedina appena mossa con la nostra pedina diagonale
    step 2 → ritira la nostra pedina nell'angolo (se non siamo stati mangiati)
    """
    cs = _corner_sequence_state
    step = cs["step"]
    corner = cs["corner"]
    extra = cs["extra"]  # {"diagonal": ..., "target": ..., "our_piece_after_capture": ...}

    legal_moves = game.actions(state)

    if step == 1:
        # Cattura la pedina target con la nostra diagonale
        diagonal = extra["diagonal"]
        target = extra["target"]

        capture_move = None
        for move in legal_moves:
            if move[0] == diagonal and move[1] == target and move[2]:
                capture_move = move
                break

        if capture_move is None:
            # Situazione fuori controllo: abbandona sequenza
            _reset_sequence()
            return None

        cs["step"] = 2
        # Dopo la cattura, la nostra pedina sarà in "target"
        cs["extra"] = {
            "diagonal": diagonal,
            "our_piece": target,
            "corner": corner,
        }
        return capture_move

    elif step == 2:
        # Ritira la nostra pedina nell'angolo
        our_piece = extra.get("our_piece")

        # Verifica che la nostra pedina sia ancora lì
        if our_piece is None or state.board[our_piece[0]][our_piece[1]] != player:
            # Siamo stati mangiati: sequenza finita
            _reset_sequence()
            return None

        # Cerca la mossa che porta our_piece → corner
        retreat_move = None
        for move in legal_moves:
            if move[0] == our_piece and move[1] == corner and not move[2]:
                retreat_move = move
                break

        if retreat_move is None:
            # Non possiamo rientrare nell'angolo (situazione imprevista)
            _reset_sequence()
            return None

        _reset_sequence()
        return retreat_move

    else:
        _reset_sequence()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Esecuzione della sequenza Caso A
# ─────────────────────────────────────────────────────────────────────────────

def _execute_case_A(game, state, player, seq):
    """
    Gestisce la sequenza del CASO A in base al passo corrente.

    step 1 → prima cattura dall'angolo (verso la pedina adiacente)
    step 2 → seconda cattura (verso la pedina rimanente)
            oppure, se siamo stati mangiati, sposta diagonale verso angolo
    step 3 → (solo se siamo stati mangiati al passo 1)
            cattura la pedina che aveva occupato l'angolo con la nostra diagonale
    """
    cs = _corner_sequence_state
    step = cs["step"]
    corner = cs["corner"]
    extra = cs["extra"]

    size = state.size
    legal_moves = game.actions(state)
    opponent = game.opponent(player)

    if step == 1:
        # Prima cattura dall'angolo
        adjacents = _adjacent_to_corner(corner, size)
        captures_from_corner = [
            m for m in legal_moves
            if m[0] == corner and m[2] and m[1] in adjacents
        ]

        if not captures_from_corner:
            _reset_sequence()
            return None

        chosen = captures_from_corner[0]
        cs["step"] = 2
        cs["extra"] = {
            "first_capture_target": chosen[1],
        }
        return chosen

    elif step == 2:
        # Verifico se la nostra pedina è ancora nell'angolo
        if state.board[corner[0]][corner[1]] == player:
            # Siamo ancora nell'angolo: seconda cattura verso la rimanente
            adjacents = _adjacent_to_corner(corner, size)
            captures_from_corner = [
                m for m in legal_moves
                if m[0] == corner and m[2] and m[1] in adjacents
            ]

            if captures_from_corner:
                chosen = captures_from_corner[0]
                _reset_sequence()
                return chosen
            else:
                _reset_sequence()
                return None

        else:
            # Siamo stati mangiati dall'angolo.
            # Dobbiamo spostare la pedina diagonale verso l'angolo.
            diagonal = _diagonal_of_corner(corner, size)
            if diagonal is None:
                _reset_sequence()
                return None

            if state.board[diagonal[0]][diagonal[1]] != player:
                # Non abbiamo la pedina diagonale
                _reset_sequence()
                return None

            # Cerchiamo una mossa non catturante dal diagonale all'angolo
            retreat_move = None
            for move in legal_moves:
                if move[0] == diagonal and move[1] == corner and not move[2]:
                    retreat_move = move
                    break

            if retreat_move is None:
                _reset_sequence()
                return None

            cs["step"] = 3
            cs["extra"] = {
                "diagonal_piece_moved_to": corner,
                "original_diagonal": diagonal,
            }
            return retreat_move

    elif step == 3:
        # La nostra pedina è nell'angolo (dopo il ritiro).
        # Ora catturiamo la pedina che aveva preso l'angolo.
        # Quella pedina dovrebbe essere nella posizione da cui aveva mangiato
        # (corner o nelle adiacenti all'angolo).
        adjacents = _adjacent_to_corner(corner, size)
        captures_from_corner = [
            m for m in legal_moves
            if m[0] == corner and m[2] and m[1] in adjacents
        ]

        if captures_from_corner:
            chosen = captures_from_corner[0]
            _reset_sequence()
            return chosen
        else:
            _reset_sequence()
            return None

    else:
        _reset_sequence()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Logica principale degli angoli
# ─────────────────────────────────────────────────────────────────────────────

def _corner_strategy_move(game, state, player):
    """
    Punto di ingresso per la logica degli angoli.

    Restituisce una mossa se siamo in una situazione angolare standard,
    altrimenti None.

    Priorità: Caso B > Caso A.
    Se più angoli soddisfano le condizioni, si sceglie casualmente tra equivalenti.
    """
    cs = _corner_sequence_state
    legal_moves = game.actions(state)

    # ── Sequenza già in corso ──────────────────────────────────────────────
    if cs["active"]:
        if cs["case"] == "B":
            move = _execute_case_B(game, state, player, cs)
        else:
            move = _execute_case_A(game, state, player, cs)

        if move is not None and move in legal_moves:
            return move
        # Sequenza interrotta: reset già fatto in _execute_*
        _reset_sequence()
        return None

    # ── Rilevamento nuovi casi ─────────────────────────────────────────────

    # Caso B ha priorità
    case_b_list = _detect_case_B(game, state, player)

    if case_b_list:
        # Scegli casualmente tra angoli Caso B equivalenti
        chosen = random.choice(case_b_list)
        cs["active"] = True
        cs["case"] = "B"
        cs["corner"] = chosen["corner"]
        cs["step"] = 1
        cs["extra"] = {
            "diagonal": chosen["diagonal"],
            "target": chosen["target"],
        }
        move = _execute_case_B(game, state, player, cs)
        if move is not None and move in legal_moves:
            return move
        _reset_sequence()
        return None

    # Caso A
    case_a_list = _detect_case_A(game, state, player)

    if case_a_list:
        chosen = random.choice(case_a_list)
        cs["active"] = True
        cs["case"] = "A"
        cs["corner"] = chosen["corner"]
        cs["step"] = 1
        cs["extra"] = {}
        move = _execute_case_A(game, state, player, cs)
        if move is not None and move in legal_moves:
            return move
        _reset_sequence()
        return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES             = 80
_W_MOBILITY           = 2
_W_CAPTURE_COUNT      = 10

_W_CAPTURE_OUTER      = 5
_W_MOVE_OUTER         = 3

_W_THREAT_PRESSURE    = 1
_W_CAPTURE_DANGEROUS  = 2

_W_CORNER_SETUP       = 4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]


def _capture_outer_bonus(game, captures):
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _move_outer_bonus(game, non_captures):
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in non_captures)


def _capture_threat_score_from_caps(game, captures):
    score = 0
    max_level = _max_level(game)
    for move in captures:
        (fr, fc), (tr, tc), is_capture = move
        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)
        score += 3 * src_level + (max_level - dst_level + 1)
    return score


def _capture_dangerous_piece_bonus_from_moves(game, captures, opponent_moves):
    threatening_pieces = set()
    for move in opponent_moves:
        (fr, fc), (tr, tc), is_capture = move
        if is_capture:
            threatening_pieces.add((fr, fc))

    bonus = 0
    for move in captures:
        (fr, fc), (tr, tc), is_capture = move
        target_level = _level(game, tr, tc)
        bonus += 2 * target_level
        if (tr, tc) in threatening_pieces:
            bonus += 12
    return bonus


def _corner_setup_bonus_limited(game, state, player, non_captures):
    if state.to_move != player:
        return 0
    if not non_captures:
        return 0

    max_level = _max_level(game)
    threshold = max_level - 1

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

    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )

    move_outer = (
        _move_outer_bonus(game, root_noncaps)
        - _move_outer_bonus(game, opp_noncaps)
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
        game, state, root_player, root_noncaps
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

    if not legal_moves:
        passed_state = game.pass_turn(state)
        return _alphabeta(
            game, passed_state, depth - 1, alpha, beta,
            not maximizing, root_player, deadline,
        )

    ordered_moves = order_moves(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered_moves:
            child = game.result(state, move)
            child_value, _ = _alphabeta(
                game, child, depth - 1, alpha, beta,
                False, root_player, deadline,
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
                game, child, depth - 1, alpha, beta,
                True, root_player, deadline,
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
    Strategia principale.

    1. Se siamo in una situazione angolare standard (Caso A o B), eseguiamo
       la mossa prevista dalla sequenza, indipendentemente dall'euristica.
    2. Altrimenti, usiamo iterative deepening con alpha-beta.
    """
    legal_moves = game.actions(state)

    if not legal_moves:
        return None

    player = state.to_move

    # ── Logica angoli (priorità assoluta) ─────────────────────────────────
    corner_move = _corner_strategy_move(game, state, player)
    if corner_move is not None:
        return corner_move

    # ── Iterative deepening con alpha-beta ────────────────────────────────
    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move = random.choice(legal_moves)
    depth = 1

    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, player, deadline,
            )
            if move is not None:
                best_move = move
            depth += 1
        except _Timeout:
            break

    return best_move