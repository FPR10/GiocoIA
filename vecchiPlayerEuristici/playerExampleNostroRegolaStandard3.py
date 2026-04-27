import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Versione ibrida con euristica angolare
#
# Manteniamo:
#   • iterative deepening
#   • alpha-beta pruning
#   • timeout sicuro
#   • P1–P5 dell'ibrido originale
#
# Aggiungiamo:
#   • P6: euristica angolare (CornerPattern)
#
# L'euristica angolare riconosce due pattern attorno agli angoli della
# scacchiera e assegna un bonus/malus in base alla situazione:
#
#   CASO A – Pedina nostra nell'angolo con 1–2 pedine avversarie adiacenti:
#     Premia le mosse che catturano verso le celle adiacenti all'angolo,
#     poi premia la seconda cattura verso la pedina rimanente.
#     In caso di mancata doppia cattura, premia il riposizionamento nell'angolo.
#
#   CASO B – Pedina avversaria nell'angolo con 1–2 pedine nostre adiacenti:
#     Premia l'attesa (non muovere le pedine adiacenti) finché l'avversario
#     non esce dall'angolo catturando; poi premia la contro-cattura immediata
#     e infine il riposizionamento nell'angolo.
#
#   Priorità: CASO B > CASO A (come da specifiche).
#   A parità di angoli equivalenti, si sceglie a caso.
#   La sequenza è euristica (flessibile), non hard-coded: i pesi bilanciano
#   questo comportamento con il resto della valutazione globale.
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

# ── nuovi pesi angolari ──────────────────────────────────────────────────────
_W_CORNER_PATTERN_A   = 15    # P6-A: bonus pattern CASO A (nostra nell'angolo)
_W_CORNER_PATTERN_B   = 20    # P6-B: bonus pattern CASO B (avversaria nell'angolo)
# Il peso di B > A rispecchia la priorità indicata nelle specifiche.


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]


def _get_corners(game):
    """Restituisce le coordinate dei quattro angoli della scacchiera."""
    n = game.size - 1
    return [(0, 0), (0, n), (n, 0), (n, n)]


def _corner_adjacent(game, cr, cc):
    """
    Restituisce le due celle adiacenti all'angolo (cr, cc).

    Per ogni angolo, le celle adiacenti sono quelle che distano 1 passo
    in direzione ortogonale (non diagonale) dall'angolo stesso.
    Queste sono le uniche celle da cui partono le catture descritte nei pattern.
    """
    n = game.size - 1
    adjacent = []

    # direzioni ortogonali verso l'interno
    if cr == 0:
        adjacent.append((1, cc))
    else:
        adjacent.append((n - 1, cc))

    if cc == 0:
        adjacent.append((cr, 1))
    else:
        adjacent.append((cr, n - 1))

    return adjacent


# ─────────────────────────────────────────────────────────────────────────────
# P6 – Euristica angolare
# ─────────────────────────────────────────────────────────────────────────────

def _corner_pattern_score(game, state, root_player):
    """
    P6: valuta i pattern angolari dal punto di vista di root_player.

    Restituisce un punteggio netto:
        bonus CASO B (angoli con pedina avversaria) – malus rispecchiato
      + bonus CASO A (angoli con nostra pedina)

    Per ciascun angolo viene calcolato il contributo del pattern più rilevante.

    ── CASO A ──────────────────────────────────────────────────────────────
    Condizione: la nostra pedina è nell'angolo E almeno una pedina avversaria
    è su una delle due celle adiacenti all'angolo.

    Bonus erogati:
      • Per ogni cella adiacente occupata dall'avversario → bonus "cattura
        disponibile verso adiacente" (la mossa dovrebbe essere una cattura).
      • Se entrambe le adiacenti sono occupate dall'avversario → bonus doppio
        (siamo nella situazione ideale del CASO A completo).
      • Bonus "ritorno all'angolo": se dopo una cattura la nostra pedina
        può tornare nell'angolo (o ci si trova già) → bonus finale di sequenza.

    ── CASO B ──────────────────────────────────────────────────────────────
    Condizione: la pedina avversaria è nell'angolo E almeno una delle due
    celle adiacenti è occupata dalla nostra pedina.

    Bonus erogati:
      • Bonus "attesa": premiamo il fatto che le nostre pedine adiacenti
        non si muovano (rimangano in posizione di agguato). Questo si traduce
        nel premio alla presenza delle nostre pedine adiacenti.
      • Se l'avversario ha appena catturato (last_move uscita dall'angolo),
        bonus "contro-cattura disponibile" per la nostra pedina sulla diagonale.
      • Bonus "ritorno angolo": se una nostra pedina può già occupare l'angolo
        (mossa di riposizionamento) → bonus finale di sequenza.
    """
    opponent  = game.opponent(root_player)
    corners   = _get_corners(game)
    score_A   = 0
    score_B   = 0

    for (cr, cc) in corners:
        adj = _corner_adjacent(game, cr, cc)
        corner_owner = state.board[cr][cc]

        # ── CASO A ──────────────────────────────────────────────────────────
        if corner_owner == root_player:
            opp_adj = [(r, c) for (r, c) in adj if state.board[r][c] == opponent]
            n_opp = len(opp_adj)

            if n_opp == 0:
                continue  # nessun avversario adiacente: pattern non attivo

            # Bonus base: presenza avversari adiacenti catturabili dall'angolo.
            # Ogni pedina avversaria adiacente vale un bonus.
            score_A += n_opp * 6

            # Bonus doppio: entrambe le adiacenti occupate (massimo vantaggio).
            if n_opp == 2:
                score_A += 8

            # Verifica se esiste una cattura effettiva dall'angolo verso le adiacenti.
            root_caps = [
                m for m in game._actions_for_player(state, root_player)
                if m[2] and m[0] == (cr, cc)
            ]
            cap_targets = {(tr, tc) for (_, (tr, tc), _) in root_caps}

            for (ar, ac) in opp_adj:
                if (ar, ac) in cap_targets:
                    # La cattura verso la pedina adiacente è realmente disponibile.
                    score_A += 10

            # Bonus ritorno all'angolo: se dopo una cattura potremmo tornare.
            # Verifichiamo se l'angolo è raggiungibile da una cella adiacente
            # con una mossa non catturante (livello angolo > livello adiacente).
            corner_level = _level(game, cr, cc)
            for (ar, ac) in adj:
                adj_level = _level(game, ar, ac)
                if corner_level > adj_level and state.board[ar][ac] == root_player:
                    # La nostra pedina nell'adiacente può rientrare nell'angolo.
                    score_A += 5

        # ── CASO B ──────────────────────────────────────────────────────────
        elif corner_owner == opponent:
            our_adj = [(r, c) for (r, c) in adj if state.board[r][c] == root_player]
            n_our = len(our_adj)

            if n_our == 0:
                continue  # nessuna nostra pedina adiacente: pattern non attivo

            # Bonus attesa: le nostre pedine adiacenti sono in posizione d'agguato.
            score_B += n_our * 7

            # Bonus se entrambe le adiacenti sono nostre (agguato completo).
            if n_our == 2:
                score_B += 6

            # Contro-cattura: l'avversario ha appena mosso dall'angolo?
            # Controlliamo se last_move è una cattura partita dall'angolo.
            last = state.last_move
            opp_just_moved_from_corner = (
                last is not None
                and last.get("type") == "move"
                and last.get("player") == opponent
                and last.get("from") == (cr, cc)
                and last.get("is_capture") is True
            )

            if opp_just_moved_from_corner:
                dest = last.get("to")
                if dest is not None:
                    # Verifica se possiamo catturare la pedina appena mossa.
                    root_caps = [
                        m for m in game._actions_for_player(state, root_player)
                        if m[2] and m[1] == dest
                    ]
                    if root_caps:
                        score_B += 18  # Contro-cattura immediata disponibile!

            # Bonus ritorno angolo: se una nostra pedina può occupare l'angolo
            # (possibile solo se l'angolo è libero, ma siamo nel CASO B quindi
            # è occupato dall'avversario; il bonus si applica alla situazione
            # post-cattura, cioè se la nostra cattura libererebbe l'angolo).
            # Verifichiamo se l'angolo è raggiungibile da una nostra pedina adiacente
            # simulando il livello: l'angolo ha livello massimo → è sempre raggiungibile
            # in termini di direzione, ma serve che sia libero.
            # Usiamo un bonus anticipatorio: se siamo adiacenti all'angolo e l'angolo
            # diventerà libero dopo una nostra cattura, premiamo la posizione.
            for (ar, ac) in our_adj:
                root_caps_from_adj = [
                    m for m in game._actions_for_player(state, root_player)
                    if m[2] and m[0] == (ar, ac) and m[1] == (cr, cc)
                ]
                if root_caps_from_adj:
                    # Possiamo catturare la pedina nell'angolo direttamente!
                    score_B += 15

    return score_A + score_B


# ─────────────────────────────────────────────────────────────────────────────
# Helpers originali (invariati)
# ─────────────────────────────────────────────────────────────────────────────

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
    opp_pieces  = state.count(opponent)

    root_moves = game._actions_for_player(state, root_player)
    opp_moves  = game._actions_for_player(state, opponent)

    root_caps    = [m for m in root_moves if m[2]]
    opp_caps     = [m for m in opp_moves  if m[2]]
    root_noncaps = [m for m in root_moves if not m[2]]
    opp_noncaps  = [m for m in opp_moves  if not m[2]]

    # P1
    cap_outer = (
        _capture_outer_bonus(game, root_caps)
      - _capture_outer_bonus(game, opp_caps)
    )
    # P2
    move_outer = (
        _move_outer_bonus(game, root_noncaps)
      - _move_outer_bonus(game, opp_noncaps)
    )
    # P3
    threat_pressure = (
        _capture_threat_score_from_caps(game, root_caps)
      - _capture_threat_score_from_caps(game, opp_caps)
    )
    # P4
    capture_dangerous = (
        _capture_dangerous_piece_bonus_from_moves(game, root_caps, opp_moves)
      - _capture_dangerous_piece_bonus_from_moves(game, opp_caps, root_moves)
    )
    # P5
    corner_setup = _corner_setup_bonus_limited(
        game, state, root_player, root_noncaps
    )

    # ── P6: euristica angolare ───────────────────────────────────────────────
    # Calcoliamo il punteggio del pattern angolare per root_player e per
    # l'opponent, poi prendiamo la differenza netta.
    # Questo permette di penalizzare anche le situazioni in cui è l'avversario
    # a sfruttare i pattern angolari a suo favore.
    corner_pattern_root = _corner_pattern_score(game, state, root_player)
    corner_pattern_opp  = _corner_pattern_score(game, state, opponent)

    # Il peso di B è già inglobato internamente in _corner_pattern_score
    # (i bonus del CASO B sono più alti). I pesi _W_CORNER_PATTERN_A/B
    # si applicano tramite un coefficiente unico sul netto, in modo
    # proporzionale alla differenza tra i due giocatori.
    # Usiamo _W_CORNER_PATTERN_B come moltiplicatore esterno (valore più alto).
    corner_pattern_net = corner_pattern_root - corner_pattern_opp

    score = (
        _W_PIECES             * (root_pieces - opp_pieces)
      + _W_MOBILITY           * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT      * (len(root_caps)  - len(opp_caps))
      + _W_CAPTURE_OUTER      * cap_outer
      + _W_MOVE_OUTER         * move_outer
      + _W_THREAT_PRESSURE    * threat_pressure
      + _W_CAPTURE_DANGEROUS  * capture_dangerous
      + _W_CORNER_SETUP       * corner_setup
      + _W_CORNER_PATTERN_B   * corner_pattern_net   # P6
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
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    deadline   = time.perf_counter() + timeout - _TIME_MARGIN
    best_move  = random.choice(legal_moves)
    best_value = -math.inf
    depth      = 1

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
                best_move  = move
                best_value = value
            depth += 1
        except _Timeout:
            break

    return best_move