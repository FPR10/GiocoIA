import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Iterative-deepening alpha-beta con valutazione Monte Carlo pesata
#
# Struttura:
#   • _get_player_corners      – determina gli angoli iniziali del giocatore
#   • _get_quarters            – determina i quarti "nostri" e "avversari"
#   • _corner_constraint_move  – vincolo prime 4 mosse (angoli obbligati)
#   • _encirclement_score      – P3: misura il grado di accerchiamento
#   • _move_weight             – peso di una mossa per le simulazioni MC
#   • _mc_simulate             – simulazione Monte Carlo pesata da uno stato
#   • _mc_evaluate             – valutazione di uno stato via N simulazioni MC
#   • _alphabeta               – alpha-beta con time check
#   • playerStrategy           – iterative deepening con guardia al tempo
#
# Regole strategiche per le simulazioni MC (fortemente pesate):
#   R1. Vincolo angoli: prime 4 mosse obbligate dalle pedine d'angolo
#   R2. Priorità sulla pedina attaccante più esterna (catture)
#   R3. Priorità sul bersaglio più interno avversario (= più pericoloso)
#   R4. Fuga dall'accerchiamento: se siamo circondati, spostarsi all'esterno
#   R5. Comportamento aggressivo nelle fasi finali (≤20 pedine totali)
#   R6. Cattura base: qualsiasi cattura vale sempre un bonus fisso
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN      = 0.12   # secondi di margine prima del timeout
_MC_SIMULATIONS   = 12     # simulazioni MC per nodo foglia (bilancio velocità/qualità)
_MC_ROLLOUT_DEPTH = 30     # profondità massima di ogni rollout


# ─────────────────────────────────────────────────────────────────────────────
# Costanti di peso per le simulazioni Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
_W_CAPTURE_BASE        = 110  # R6: bonus fisso per qualsiasi cattura
_W_ATTACKER_OUTER      =  60  # R2: bonus per ogni livello di distanza dell'attaccante
_W_TARGET_INNER        =  -30  # R3: bonus per catturare pedine più INTERNE (più pericolose)
_W_MOVE_OUTER          =  55  # mosse non-catturanti verso esterno (nessun avversario vicino)
_W_ENCIRCLEMENT_ESCAPE =  100  # R4: bonus forte per muoversi all'esterno quando accerchiati
_W_ENCIRCLEMENT_THRESH =   1  # R4: numero minimo avversari adiacenti per considerarsi accerchiati
_W_ENDGAME_CAPTURE     = 200  # R5: peso cattura in fase finale
_W_ENDGAME_OUTER_MOVE  =  60  # R5: peso mossa verso esterno in fase finale
_W_ENDGAME_THRESHOLD   =  20  # soglia pedine totali per fase finale
_W_CORNER_FORCE        = 9999 # R1: peso enorme per mosse obbligate da angoli
_W_INNER_PENALTY       =  0.05 # moltiplicatore penalità per mosse verso il centro


# ─────────────────────────────────────────────────────────────────────────────
# Stato globale per il vincolo sugli angoli (aggiornato a ogni chiamata esterna)
# ─────────────────────────────────────────────────────────────────────────────
_corner_state = {}   # inizializzato in playerStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers geometrici
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]


def _get_player_corners(game, state, player):
    """Restituisce le celle angolo che, nello stato iniziale, appartenevano a player."""
    s = game.size - 1
    corners = [(0, 0), (0, s), (s, 0), (s, s)]
    if player == "Blue":
        return [(r, c) for r, c in corners if (r + c) % 2 == 0]
    else:
        return [(r, c) for r, c in corners if (r + c) % 2 != 0]


def _get_quarters(game, player):
    """Restituisce (our_quarters, opp_quarters) come insiemi di celle."""
    half = game.size // 2
    s = game.size
    quadrants = {
        "TL": {(r, c) for r in range(half) for c in range(half)},
        "TR": {(r, c) for r in range(half) for c in range(half, s)},
        "BL": {(r, c) for r in range(half, s) for c in range(half)},
        "BR": {(r, c) for r in range(half, s) for c in range(half, s)},
    }
    corner_to_quad = {
        (0, 0): "TL", (0, s - 1): "TR",
        (s - 1, 0): "BL", (s - 1, s - 1): "BR",
    }
    our_corners = _get_player_corners(game, None, player)
    our_quads = set()
    for corner in our_corners:
        q = corner_to_quad.get(corner)
        if q:
            our_quads.add(q)
    opp_quads = set(quadrants.keys()) - our_quads
    our_cells = set()
    opp_cells = set()
    for q in our_quads:
        our_cells |= quadrants[q]
    for q in opp_quads:
        opp_cells |= quadrants[q]
    return our_cells, opp_cells


# ─────────────────────────────────────────────────────────────────────────────
# R1 – Vincolo iniziale: prime mosse obbligate dagli angoli
# ─────────────────────────────────────────────────────────────────────────────

def _corner_constraint_move(game, state, player, corner_tracker):
    """Restituisce la mossa obbligata da un angolo, se esiste, altrimenti None."""
    if not corner_tracker:
        return None

    legal_moves = game._actions_for_player(state, player)
    cap_map = {}
    for move in legal_moves:
        (fr, fc), (tr, tc), is_cap = move
        if is_cap:
            cap_map.setdefault((fr, fc), []).append(move)

    for corner, status in list(corner_tracker.items()):
        if status == 'done':
            continue
        cr, cc = corner
        if state.board[cr][cc] != player:
            corner_tracker[corner] = 'done'
            continue
        caps_from_corner = cap_map.get(corner, [])
        if caps_from_corner:
            caps_from_corner.sort(key=lambda m: _level(game, m[1][0], m[1][1]))
            return caps_from_corner[0]

    return None


def _init_corner_tracker(game, state, player):
    """Inizializza il tracker degli angoli per il giocatore."""
    corners = _get_player_corners(game, state, player)
    tracker = {}
    for corner in corners:
        cr, cc = corner
        if state.board[cr][cc] == player:
            tracker[corner] = 'pending'
    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# R4 – Rilevamento accerchiamento
# ─────────────────────────────────────────────────────────────────────────────

def _encirclement_score(game, state, fr, fc, player):
    """Conta quante pedine avversarie adiacenti (raggio 2) hanno livello >= nostro.

    Un valore alto significa che siamo circondati da avversari altrettanto o più
    esterni: è il segnale di accerchiamento. Restituisce il conteggio.
    """
    opponent = game.opponent(player)
    our_level = _level(game, fr, fc)
    count = 0
    radius = 2
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = fr + dr, fc + dc
            if not game.in_bounds(nr, nc):
                continue
            if state.board[nr][nc] == opponent:
                if _level(game, nr, nc) >= our_level:
                    count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# R5 – Fase finale
# ─────────────────────────────────────────────────────────────────────────────

def _is_endgame(state):
    total = sum(1 for r in state.board for c in r if c is not None)
    return total <= _W_ENDGAME_THRESHOLD


def _endgame_advantage(state, player):
    """True se siamo avvantaggiati in fase finale (≥ pedine avversarie)."""
    opp = "Blue" if player == "Red" else "Red"
    return state.count(player) >= state.count(opp)


# ─────────────────────────────────────────────────────────────────────────────
# Calcolo peso di una mossa per le simulazioni MC
# ─────────────────────────────────────────────────────────────────────────────

def _move_weight(game, state, move, player, corner_tracker_sim):
    """Assegna un peso (≥ 0.001) a una mossa da usare nel campionamento MC.

    Incorpora tutte le regole R1–R6.
    """
    (fr, fc), (tr, tc), is_cap = move
    weight = 1.0

    # ── R1: mossa obbligata da angolo ─────────────────────────────────────
    forced = _corner_constraint_move(game, state, player, corner_tracker_sim)
    if forced is not None:
        if move == forced:
            return float(_W_CORNER_FORCE)
        else:
            return 0.001   # quasi-zero: non scegliere altre mosse

    endgame = _is_endgame(state)
    ml = _max_level(game)

    if is_cap:
        # ── R6: bonus fisso per qualsiasi cattura ─────────────────────────
        weight += _W_CAPTURE_BASE

        # ── R2: attaccante più esterno → cattura più preziosa ─────────────
        attacker_level = _level(game, fr, fc)
        weight += _W_ATTACKER_OUTER * attacker_level

        # ── R3: bersaglio più INTERNO = più pericoloso → catturarlo vale ──
        #   Usiamo (max_level - livello_destinazione): più è interno,
        #   più alto il bonus (logica opposta rispetto alla versione precedente)
        target_level = _level(game, tr, tc)
        inner_bonus = ml - target_level   # alto se bersaglio è interno
        weight += _W_TARGET_INNER * (inner_bonus + 1)

        # ── R5: fase finale – cattura fortemente premiata ─────────────────
        if endgame:
            multiplier = 2 if _endgame_advantage(state, player) else 1
            weight += _W_ENDGAME_CAPTURE * multiplier

    else:
        # ── mosse non-catturanti ──────────────────────────────────────────
        dest_level = _level(game, tr, tc)
        src_level  = _level(game, fr, fc)
        delta = dest_level - src_level   # positivo = ci allontaniamo dal centro

        # ── R4: fuga dall'accerchiamento ──────────────────────────────────
        #   Se la pedina in (fr,fc) è circondata da avversari altrettanto o
        #   più esterni, spingiamo FORTEMENTE verso mosse che la allontanano
        #   dal centro (delta > 0). Questo è il corrispettivo di P3 in nostro.
        enc = _encirclement_score(game, state, fr, fc, player)
        if enc >= _W_ENCIRCLEMENT_THRESH:
            if delta > 0:
                # Vogliamo uscire: bonus proporzionale all'accerchiamento e al delta
                weight += _W_ENCIRCLEMENT_ESCAPE * delta * enc
            else:
                # Muoversi verso il centro mentre siamo accerchiati = penalità
                weight *= _W_INNER_PENALTY
        else:
            # Nessun accerchiamento: bonus standard per mosse verso l'esterno
            if delta > 0:
                weight += _W_MOVE_OUTER * delta
            else:
                # Penalità leggera per mosse verso il centro in assenza di motivo
                weight *= 0.5

        # ── R5: fase finale – solo mosse verso esterno ────────────────────
        if endgame:
            if delta > 0:
                multiplier = 2 if _endgame_advantage(state, player) else 1
                weight += _W_ENDGAME_OUTER_MOVE * multiplier
            else:
                weight *= _W_INNER_PENALTY   # fase finale: non tornare al centro

    return max(weight, 0.001)


def _weighted_choice(moves, weights):
    """Campionamento pesato: sceglie una mossa in base ai pesi."""
    total = sum(weights)
    if total <= 0:
        return random.choice(moves)
    r = random.uniform(0, total)
    cumulative = 0.0
    for move, w in zip(moves, weights):
        cumulative += w
        if r <= cumulative:
            return move
    return moves[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Simulazione Monte Carlo pesata (rollout)
# ─────────────────────────────────────────────────────────────────────────────

def _mc_simulate(game, state, root_player, max_depth, corner_tracker_p1, corner_tracker_p2):
    """Esegue un rollout pesato fino a max_depth o terminale.

    Restituisce +1 (vittoria root_player), -1 (sconfitta), 0/±0.5 (draw/limite).
    I tracker degli angoli sono copiati in modo da non alterare lo stato esterno.
    """
    current_state = state
    opponent = game.opponent(root_player)
    # Copia leggera dei tracker per non modificare quelli della ricerca
    ct = {
        root_player: dict(corner_tracker_p1),
        opponent:    dict(corner_tracker_p2),
    }

    for _ in range(max_depth):
        if game.is_terminal(current_state):
            break

        current_player = current_state.to_move
        legal_moves = game.actions(current_state)

        if not legal_moves:
            current_state = game.pass_turn(current_state)
            continue

        tracker = ct.get(current_player, {})
        weights = [_move_weight(game, current_state, m, current_player, tracker)
                   for m in legal_moves]

        chosen = _weighted_choice(legal_moves, weights)

        # Aggiorna il tracker angoli: se la pedina lascia l'angolo, segna 'done'
        (fr, fc), (tr, tc), is_cap = chosen
        if (fr, fc) in tracker and tracker[(fr, fc)] == 'pending':
            # La pedina si è mossa dall'angolo (sia cattura che spostamento)
            tracker[(fr, fc)] = 'done'

        current_state = game.result(current_state, chosen)

    winner = game.winner(current_state)
    if winner == root_player:
        return 1
    elif winner is not None:
        return -1
    # Valutazione rapida se limite raggiunto: differenza pedine normalizzata
    my_pieces  = current_state.count(root_player)
    opp_pieces = current_state.count(game.opponent(root_player))
    if my_pieces > opp_pieces:
        return 0.5
    elif my_pieces < opp_pieces:
        return -0.5
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione MC di uno stato (media di N simulazioni)
# ─────────────────────────────────────────────────────────────────────────────

def _mc_evaluate(game, state, root_player, n_sims, ct_root, ct_opp):
    """Valuta uno stato tramite n_sims simulazioni MC pesate."""
    winner = game.winner(state)
    if winner == root_player:
        return 100_000.0
    if winner is not None:
        return -100_000.0

    total = 0.0
    for _ in range(n_sims):
        total += _mc_simulate(game, state, root_player, _MC_ROLLOUT_DEPTH, ct_root, ct_opp)
    return total / n_sims


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con valutazione MC ai nodi foglia
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _order_moves_mc(game, moves):
    """Ordinamento mosse per alpha-beta: catture di pedine interne prima (R3),
    poi catture esterne (R2), poi mosse verso l'esterno (R4).
    """
    ml = _max_level(game)

    def priority(m):
        (fr, fc), (tr, tc), is_cap = m
        if is_cap:
            # Prima le catture di pedine interne (più pericolose)
            target_innermost = ml - _level(game, tr, tc)   # alto = più interno
            attacker_outer   = _level(game, fr, fc)
            return (0, -target_innermost, -attacker_outer)
        return (1, -_level(game, tr, tc), 0)

    caps    = sorted([m for m in moves if m[2]],     key=priority)
    noncaps = sorted([m for m in moves if not m[2]], key=priority)
    return caps + noncaps


def _alphabeta(game, state, depth, alpha, beta, maximizing, root_player,
               deadline, n_sims, ct_root, ct_opp):
    """Alpha-beta con valutazione MC ai nodi foglia.

    Lancia _Timeout se si supera 'deadline'.
    """
    if time.perf_counter() >= deadline:
        raise _Timeout()

    legal_moves = game.actions(state)

    if depth == 0 or game.is_terminal(state) or not legal_moves:
        return _mc_evaluate(game, state, root_player, n_sims, ct_root, ct_opp), None

    ordered = _order_moves_mc(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(
                game, child, depth - 1, alpha, beta, False,
                root_player, deadline, n_sims, ct_root, ct_opp
            )
            if child_val > value:
                value = child_val
                best_moves = [move]
            elif child_val == value:
                best_moves.append(move)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(
                game, child, depth - 1, alpha, beta, True,
                root_player, deadline, n_sims, ct_root, ct_opp
            )
            if child_val < value:
                value = child_val
                best_moves = [move]
            elif child_val == value:
                best_moves.append(move)
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, (random.choice(best_moves) if best_moves else None)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point: iterative deepening con guardia al tempo
# ─────────────────────────────────────────────────────────────────────────────

# Tracker persistenti tra le chiamate (memoria dell'avanzamento mosse angolo)
_persistent_ct_root = None
_persistent_ct_opp  = None
_persistent_player  = None


def playerStrategy(game, state, timeout=3):
    """Strategia MC con iterative-deepening alpha-beta e timeout sicuro.

    Usa simulazioni Monte Carlo pesate come euristica ai nodi foglia,
    con regole fortemente pesate (R1–R6).
    """
    global _persistent_ct_root, _persistent_ct_opp, _persistent_player

    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    player   = state.to_move
    opponent = game.opponent(player)

    # Inizializza i tracker degli angoli al primo turno del giocatore
    if _persistent_player != player:
        _persistent_player  = player
        _persistent_ct_root = _init_corner_tracker(game, state, player)
        _persistent_ct_opp  = _init_corner_tracker(game, state, opponent)

    ct_root = _persistent_ct_root
    ct_opp  = _persistent_ct_opp

    # ── R1: verifica mossa obbligata da angolo ────────────────────────────
    forced = _corner_constraint_move(game, state, player, ct_root)
    if forced is not None and forced in legal_moves:
        # La pedina lascia l'angolo: marca come 'done'
        (fr, fc), (tr, tc), is_cap = forced
        if (fr, fc) in ct_root:
            ct_root[(fr, fc)] = 'done'
        return forced

    # ── Iterative deepening alpha-beta con MC ─────────────────────────────
    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move  = random.choice(legal_moves)
    best_value = -math.inf

    n_sims = _MC_SIMULATIONS

    depth = 1
    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, player,
                deadline, n_sims,
                ct_root, ct_opp,
            )
            if move is not None:
                best_move  = move
                best_value = value
            depth += 1
        except _Timeout:
            break

    return best_move