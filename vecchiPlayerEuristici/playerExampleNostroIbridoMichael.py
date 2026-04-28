import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - RossoVideo Strategy
#
# Obiettivo:
#   imitare il comportamento del giocatore rosso visto nel video:
#
#   • occupare/cercare cornice e angoli;
#   • non regalare pedine esterne;
#   • usare le pedine esterne come pressione tattica;
#   • accettare scambi solo se la ricattura è buona;
#   • seguire pattern angolari solo quando non siamo in svantaggio materiale;
#   • nel finale diventare più concreto sulle catture.
#
# Strategia:
#   • alpha-beta con iterative deepening;
#   • valutazione materiale + catture + posizione;
#   • DANGER: evita pedine catturabili;
#   • SAFE_OUTER: premia pedine esterne stabili;
#   • regole angolari leggere;
#   • endgame leggero.
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES              = 90
_W_MOBILITY            = 2
_W_CAPTURE_COUNT       = 12

_W_POSITION            = 5
_W_CAPTURE_OUTER       = 4
_W_THREAT_PRESSURE     = 1
_W_CAPTURE_DANGEROUS   = 2
_W_CORNER_SETUP        = 3

# Regola che ha reso forte il modello contro Ibrido2:
# conta quante pedine sono catturabili da noi e dall'avversario.
_W_DANGER              = 18

# Regola angolare standard, ma leggera.
_W_CORNER_STANDARD     = 2

# Nuova regola ispirata al rosso del video:
# premia pedine esterne/angolari stabili e non catturabili.
_W_SAFE_OUTER          = 6

# Endgame leggero: nel finale dà più importanza alle catture.
_W_ENDGAME_CAPTURE     = 5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers base
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """
    Livello massimo della scacchiera.
    Su 8x8 è 9, su 6x6 è 6.
    """
    return game.distance_levels[0][0]


def _corners(game):
    s = game.size - 1
    return [(0, 0), (0, s), (s, 0), (s, s)]


def _corner_inward_cells(game, corner):
    """
    Restituisce le tre celle interne vicine a un angolo:
      - verticale interna;
      - orizzontale interna;
      - diagonale interna.
    """
    r, c = corner
    dr = 1 if r == 0 else -1
    dc = 1 if c == 0 else -1

    cells = [
        (r + dr, c),
        (r, c + dc),
        (r + dr, c + dc),
    ]

    return [(x, y) for x, y in cells if game.in_bounds(x, y)]


def _corner_orthogonal_cells(game, corner):
    """
    Restituisce le due celle ortogonali vicine all'angolo.
    """
    r, c = corner
    dr = 1 if r == 0 else -1
    dc = 1 if c == 0 else -1

    cells = [
        (r + dr, c),
        (r, c + dc),
    ]

    return [(x, y) for x, y in cells if game.in_bounds(x, y)]


def _corner_diagonal_cell(game, corner):
    """
    Restituisce la cella diagonale interna rispetto all'angolo.
    """
    r, c = corner
    dr = 1 if r == 0 else -1
    dc = 1 if c == 0 else -1

    cell = (r + dr, c + dc)

    if game.in_bounds(cell[0], cell[1]):
        return cell

    return None


def _is_endgame(state):
    """
    Fase finale: poche pedine totali sulla scacchiera.
    In questa fase conviene essere più concreti sulle catture.
    """
    total = 0

    for row in state.board:
        for cell in row:
            if cell is not None:
                total += 1

    return total <= 12


# ─────────────────────────────────────────────────────────────────────────────
# Euristiche principali
# ─────────────────────────────────────────────────────────────────────────────

def _positional_value(game, state, player):
    """
    Valore posizionale assoluto.

    Somma livello^2 delle nostre pedine meno livello^2 delle pedine avversarie.

    Questo premia fortemente cornice e angoli, ma senza essere una regola
    obbligata.
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
    P1: premia catture verso celle più esterne.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_threat_score_from_caps(game, captures):
    """
    P3: pressione tattica tramite qualità delle catture disponibili.

    Premia:
      - catture fatte da pedine esterne;
      - catture verso bersagli più interni.
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

    Una pedina avversaria è più pericolosa se:
      - è esterna;
      - ha almeno una cattura disponibile.
    """
    threatening_pieces = {
        (fr, fc)
        for (fr, fc), _, is_cap in opponent_moves
        if is_cap
    }

    bonus = 0

    for (_, _), (tr, tc), _ in captures:
        target_level = _level(game, tr, tc)

        # Pedina esterna avversaria: spesso controlla linee importanti.
        bonus += 2 * target_level

        # Se aveva una cattura disponibile, eliminarla è molto utile.
        if (tr, tc) in threatening_pieces:
            bonus += 12

    return bonus


def _pieces_under_attack_from_moves(moves):
    """
    Restituisce l'insieme delle celle che possono essere catturate
    dalle mosse passate.

    Se passiamo le mosse dell'avversario, otteniamo le nostre pedine
    sotto attacco.
    """
    attacked = set()

    for move in moves:
        (_, _), (tr, tc), is_capture = move

        if is_capture:
            attacked.add((tr, tc))

    return attacked


def _safe_outer_value(game, state, player, opponent_moves):
    """
    Regola ispirata al rosso del video.

    Premia pedine nostre su cornice/angoli che NON sono immediatamente
    catturabili.

    Penalizza pedine esterne se sono esposte.

    Idea strategica:
      una pedina esterna è forte solo se è stabile.
      Se è esterna ma catturabile subito, può diventare un regalo.
    """
    attacked = _pieces_under_attack_from_moves(opponent_moves)

    max_level = _max_level(game)
    outer_threshold = max_level - 1

    value = 0

    for r in range(state.size):
        for c in range(state.size):
            if state.board[r][c] != player:
                continue

            lv = _level(game, r, c)

            if lv < outer_threshold:
                continue

            if (r, c) in attacked:
                # Pedina esterna ma catturabile: male.
                value -= lv
            else:
                # Pedina esterna stabile: molto buona.
                value += lv

    return value


def _corner_setup_bonus_limited(game, state, player, non_captures):
    """
    P5: setup verso angoli/periferia.

    Analizza solo le 8 mosse non catturanti più orientate verso l'esterno
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


# ─────────────────────────────────────────────────────────────────────────────
# Regola angolare standard, versione leggera
# ─────────────────────────────────────────────────────────────────────────────

def _corner_standard_rule_bonus(game, state, player, player_moves):
    """
    Regola angolare standard controllata.

    CASO 1:
      nostra pedina nell'angolo + nemici vicini.
      Bonus se possiamo catturare dalle celle angolari.

    CASO 2:
      pedina avversaria nell'angolo + nostre pedine vicine.
      Bonus maggiore, perché può nascere il pattern:
        - avversario cattura dall'angolo;
        - nostra pedina diagonale ricattura;
        - se possibile, rientriamo nell'angolo.

    È volutamente leggera: deve suggerire il pattern, non forzarlo.
    """
    opponent = game.opponent(player)
    bonus = 0

    capture_moves = [m for m in player_moves if m[2]]
    non_capture_moves = [m for m in player_moves if not m[2]]

    corners = _corners(game)

    # CASO 1: nostra pedina nell'angolo.
    for corner in corners:
        cr, cc = corner

        if state.board[cr][cc] != player:
            continue

        orthogonal_cells = _corner_orthogonal_cells(game, corner)

        nearby_enemy_count = sum(
            1
            for r, c in orthogonal_cells
            if state.board[r][c] == opponent
        )

        if nearby_enemy_count == 0:
            continue

        corner_captures = [
            m for m in capture_moves
            if m[0] == corner and m[1] in orthogonal_cells
        ]

        bonus += 12 * len(corner_captures)

        if nearby_enemy_count >= 2:
            bonus += 8

    # CASO 2: pedina avversaria nell'angolo.
    for corner in corners:
        cr, cc = corner

        if state.board[cr][cc] != opponent:
            continue

        orthogonal_cells = _corner_orthogonal_cells(game, corner)
        diagonal_cell = _corner_diagonal_cell(game, corner)

        own_orthogonal_count = sum(
            1
            for r, c in orthogonal_cells
            if state.board[r][c] == player
        )

        diagonal_guard = (
            diagonal_cell is not None
            and state.board[diagonal_cell[0]][diagonal_cell[1]] == player
        )

        if own_orthogonal_count > 0 and diagonal_guard:
            bonus += 20 + 8 * own_orthogonal_count

        # Ricattura dopo cattura avversaria dall'angolo.
        last = state.last_move

        if last and last.get("type") == "capture":
            if last.get("player") == opponent and last.get("from") == corner:
                captured_to = last.get("to")

                can_recapture = any(
                    m[2] and m[1] == captured_to
                    for m in capture_moves
                )

                if can_recapture:
                    bonus += 50

    # Step finale: riposizionarsi in un angolo.
    for move in non_capture_moves:
        (_, _), (tr, tc), _ = move

        if (tr, tc) in corners:
            bonus += 20

    # Stabilizza leggermente il possesso degli angoli.
    own_corners = sum(
        1 for r, c in corners
        if state.board[r][c] == player
    )

    opp_corners = sum(
        1 for r, c in corners
        if state.board[r][c] == opponent
    )

    bonus += 4 * (own_corners - opp_corners)

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
    opp_pieces = state.count(opponent)
    material_diff = root_pieces - opp_pieces

    root_moves = game._actions_for_player(state, root_player)
    opp_moves = game._actions_for_player(state, opponent)

    root_caps = [m for m in root_moves if m[2]]
    opp_caps = [m for m in opp_moves if m[2]]

    root_noncaps = [m for m in root_moves if not m[2]]

    # Posizione assoluta: angoli/cornice importanti.
    positional = _positional_value(game, state, root_player)

    # P1: catture verso livelli esterni.
    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )

    # P3: pressione tattica.
    threat_pressure = (
        _capture_threat_score_from_caps(game, root_caps)
        - _capture_threat_score_from_caps(game, opp_caps)
    )

    # P4: cattura pedine pericolose.
    capture_dangerous = (
        _capture_dangerous_piece_bonus_from_moves(game, root_caps, opp_moves)
        - _capture_dangerous_piece_bonus_from_moves(game, opp_caps, root_moves)
    )

    # P5: setup verso angoli/periferia.
    corner_setup = _corner_setup_bonus_limited(
        game,
        state,
        root_player,
        root_noncaps
    )

    # DANGER semplice:
    # se noi abbiamo pedine catturabili: male;
    # se l'avversario ha pedine catturabili: bene.
    root_under_attack = len(_pieces_under_attack_from_moves(opp_moves))
    opp_under_attack = len(_pieces_under_attack_from_moves(root_moves))

    danger = opp_under_attack - root_under_attack

    # SAFE_OUTER:
    # imita il rosso del video: pedine esterne sì, ma solo se stabili.
    safe_outer = (
        _safe_outer_value(game, state, root_player, opp_moves)
        - _safe_outer_value(game, state, opponent, root_moves)
    )

    # Regola angolare standard:
    # attiva solo se non siamo troppo sotto materiale.
    # Se siamo sotto di 2 o più pedine, meglio recuperare materiale.
    if material_diff >= -1:
        corner_standard = (
            _corner_standard_rule_bonus(game, state, root_player, root_moves)
            - _corner_standard_rule_bonus(game, state, opponent, opp_moves)
        )
    else:
        corner_standard = 0

    # Endgame leggero:
    # quando rimangono poche pedine, chiudere le catture diventa più importante.
    if _is_endgame(state):
        endgame_bonus = _W_ENDGAME_CAPTURE * (len(root_caps) - len(opp_caps))
    else:
        endgame_bonus = 0

    score = (
        _W_PIECES              * material_diff
      + _W_MOBILITY            * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT       * (len(root_caps) - len(opp_caps))
      + _W_POSITION            * positional
      + _W_CAPTURE_OUTER       * cap_outer
      + _W_THREAT_PRESSURE     * threat_pressure
      + _W_CAPTURE_DANGEROUS   * capture_dangerous
      + _W_CORNER_SETUP        * corner_setup
      + _W_DANGER              * danger
      + _W_SAFE_OUTER          * safe_outer
      + _W_CORNER_STANDARD     * corner_standard
      + endgame_bonus
    )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Move ordering con priorità angolare
# ─────────────────────────────────────────────────────────────────────────────

def _corner_move_bonus(game, state, move, player):
    """
    Bonus usato solo per ordinare le mosse.

    È volutamente forte perché non entra direttamente nello score:
    serve ad alpha-beta per guardare prima mosse angolari/ricatture.
    """
    if state is None or player is None:
        return 0

    opponent = game.opponent(player)
    (fr, fc), (tr, tc), is_capture = move

    corners = _corners(game)
    source = (fr, fc)
    target = (tr, tc)

    bonus = 0

    last = state.last_move

    # Ricattura dopo cattura avversaria partita da un angolo.
    if last and last.get("type") == "capture":
        if last.get("player") == opponent and last.get("from") in corners:
            if is_capture and target == last.get("to"):
                bonus += 200

    # Cattura da una nostra pedina nell'angolo.
    if is_capture and source in corners:
        bonus += 80

    # Ritorno/riposizionamento in angolo.
    if not is_capture and target in corners:
        bonus += 100

    # Cattura vicino all'angolo.
    for corner in corners:
        if target in _corner_inward_cells(game, corner):
            bonus += 25

    return bonus


def order_moves(game, moves, state=None, player=None):
    """
    Ordina le mosse per migliorare alpha-beta.

    Priorità:
      1. catture;
      2. mosse angolari/ricatture;
      3. livello assoluto di destinazione;
      4. delta di livello.
    """
    def move_priority(move):
        (fr, fc), (tr, tc), is_capture = move

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)
        delta = dst_level - src_level

        corner_bonus = _corner_move_bonus(game, state, move, player)

        if is_capture:
            return (0, -corner_bonus, -dst_level, -src_level)

        return (1, -corner_bonus, -dst_level, -delta)

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

    ordered_moves = order_moves(game, legal_moves, state, state.to_move)
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

    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move = random.choice(legal_moves)
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

            depth += 1

        except _Timeout:
            break

    return best_move