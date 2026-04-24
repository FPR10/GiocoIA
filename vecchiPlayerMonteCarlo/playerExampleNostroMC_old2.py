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
#   • _move_weight             – peso di una mossa per le simulazioni MC
#   • _mc_simulate             – simulazione Monte Carlo pesata da uno stato
#   • _mc_evaluate             – valutazione di uno stato via N simulazioni MC
#   • _alphabeta               – alpha-beta con time check
#   • playerStrategy           – iterative deepening con guardia al tempo
#
# Regole strategiche per le simulazioni MC (fortemente pesate):
#   R1. Vincolo angoli: prime 4 mosse obbligate dalle pedine d'angolo
#   R2. Priorità sulla pedina attaccante più esterna (catture)
#   R3. Priorità sul bersaglio più esterno (catture)
#   R4. Fuga verso l'esterno: se circondati o meno esterni dell'avversario
#   R5. Comportamento aggressivo nelle fasi finali (≤16 pedine totali)
#   R6. Anti-cluster: penalità per mosse che portano verso il centro
#       quando le nostre pedine sono già agglomerate lì
#   R7. Pressione esterna: bonus nostre pedine esterne vicino ad avversari
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN      = 0.12   # secondi di margine prima del timeout
_MC_SIMULATIONS   = 20     # simulazioni MC per nodo foglia (alzato: più robustezza)
_MC_ROLLOUT_DEPTH = 30     # profondità massima di ogni rollout


# ─────────────────────────────────────────────────────────────────────────────
# Costanti di peso per le simulazioni Monte Carlo (fortemente differenziate)
# ─────────────────────────────────────────────────────────────────────────────
_W_CAPTURE_BASE       = 100   # peso base per qualsiasi cattura (ora effettivamente usato)
_W_ATTACKER_OUTER     =  40   # R2: bonus per ogni livello di distanza dell'attaccante
_W_TARGET_OUTER       =  10   # R3: bonus per ogni livello di distanza del bersaglio
_W_MOVE_OUTER         =  40   # mossa non-catturante verso esterno (zona neutra)
_W_ESCAPE_OUTER       =  80   # R4: bonus fuga verso esterno quando circondati/inferiori
_W_ENDGAME_CAPTURE    = 200   # R5: peso cattura in fase finale
_W_ENDGAME_OUTER_MOVE =  60   # R5: peso mossa verso esterno in fase finale
_W_ENDGAME_THRESHOLD  =  16   # R5: soglia pedine totali per fase finale (abbassato)
_W_CLUSTER_PENALTY    =  60   # R6: penalità mossa verso il centro se già agglomerati
_W_OUTER_PRESSURE     =  50   # R7: bonus nostre pedine esterne vicino ad avversari esterni
_W_CORNER_FORCE       = 9999  # R1: peso enorme per mosse obbligate da angoli


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
# R5 – Fase finale
# ─────────────────────────────────────────────────────────────────────────────

def _is_endgame(state):
    total = sum(1 for r in state.board for c in r if c is not None)
    return total <= _W_ENDGAME_THRESHOLD


def _endgame_advantage(state, player):
    opp = "Blue" if player == "Red" else "Red"
    return state.count(player) >= state.count(opp)


# ─────────────────────────────────────────────────────────────────────────────
# R4 – Consapevolezza dell'accerchiamento e fuga verso l'esterno
# ─────────────────────────────────────────────────────────────────────────────

def _is_encircled_or_outleveled(game, state, fr, fc, player):
    """True se la pedina in (fr,fc) è in una situazione di svantaggio posizionale:
    - è circondata da avversari (≥2 nelle 8 direzioni), OPPURE
    - ha un avversario più esterno (livello più alto) nelle vicinanze (raggio 2).

    Questa è la condizione di "accerchiamento" che deve spingere la pedina
    verso l'esterno per mettere l'avversario tra sé e il centro.
    """
    opponent = game.opponent(player)
    our_level = _level(game, fr, fc)

    # Conta avversari adiacenti diretti
    adj_opp = 0
    for dr, dc in game.DIRECTIONS:
        nr, nc = fr + dr, fc + dc
        if game.in_bounds(nr, nc) and state.board[nr][nc] == opponent:
            adj_opp += 1
    if adj_opp >= 2:
        return True

    # Controlla se c'è un avversario più esterno nel raggio 2
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            nr, nc = fr + dr, fc + dc
            if not game.in_bounds(nr, nc):
                continue
            if state.board[nr][nc] == opponent and _level(game, nr, nc) > our_level:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# R6 – Anti-cluster: rilevamento agglomerazione al centro
# ─────────────────────────────────────────────────────────────────────────────

def _is_clustered_near_center(game, state, fr, fc, player):
    """True se la pedina in (fr,fc) è circondata da almeno 2 proprie pedine
    con livello ≤ soglia (zona interna/centrale).

    Questa condizione penalizza le mosse verso il centro quando siamo già
    agglomerati lì — il difetto principale del vecchio MC.
    """
    ml = _max_level(game)
    inner_threshold = ml // 2   # metà del livello massimo = zona interna
    our_level = _level(game, fr, fc)
    if our_level > inner_threshold:
        return False   # già esterna: non siamo agglomerati al centro

    cluster_count = 0
    for dr, dc in game.DIRECTIONS:
        nr, nc = fr + dr, fc + dc
        if game.in_bounds(nr, nc) and state.board[nr][nc] == player:
            if _level(game, nr, nc) <= inner_threshold:
                cluster_count += 1
    return cluster_count >= 2


# ─────────────────────────────────────────────────────────────────────────────
# R7 – Pressione esterna: nostre pedine esterne vicino ad avversari esterni
# ─────────────────────────────────────────────────────────────────────────────

def _outer_pressure_score(game, state, fr, fc, player):
    """Restituisce un bonus se la pedina in (fr,fc) è in zona esterna e
    ha avversari anch'essi in zona esterna nelle vicinanze.

    Corrisponde esattamente a _outer_pressure_bonus di playerExampleNostro,
    ma a livello di singola pedina per usarla nel peso MC.
    """
    ml = _max_level(game)
    threshold = ml - 2
    if _level(game, fr, fc) < threshold:
        return 0
    opponent = game.opponent(player)
    bonus = 0
    for dr, dc in game.DIRECTIONS:
        nr, nc = fr + dr, fc + dc
        if game.in_bounds(nr, nc) and state.board[nr][nc] == opponent:
            if _level(game, nr, nc) >= threshold:
                bonus += 1
    return bonus


# ─────────────────────────────────────────────────────────────────────────────
# Calcolo peso di una mossa per le simulazioni MC
# ─────────────────────────────────────────────────────────────────────────────

def _move_weight(game, state, move, player, corner_tracker_sim):
    """Assegna un peso (≥ 1) a una mossa da usare nel campionamento MC.

    Incorpora tutte le regole R1–R7.
    """
    (fr, fc), (tr, tc), is_cap = move
    weight = 1.0

    # ── R1: mossa obbligata da angolo ─────────────────────────────────────
    forced = _corner_constraint_move(game, state, player, corner_tracker_sim)
    if forced is not None:
        if move == forced:
            return float(_W_CORNER_FORCE)
        else:
            return 0.0001   # quasi-zero: altre mosse non devono essere scelte

    endgame = _is_endgame(state)

    if is_cap:
        # ── Peso base cattura (FIX: _W_CAPTURE_BASE ora usato) ────────────
        weight += _W_CAPTURE_BASE

        # ── R2: attaccante più esterno ────────────────────────────────────
        attacker_level = _level(game, fr, fc)
        weight += _W_ATTACKER_OUTER * attacker_level

        # ── R3: bersaglio più esterno ─────────────────────────────────────
        target_level = _level(game, tr, tc)
        weight += _W_TARGET_OUTER * target_level

        # ── R7: pressione esterna ─────────────────────────────────────────
        weight += _W_OUTER_PRESSURE * _outer_pressure_score(game, state, fr, fc, player)

        # ── R5: fase finale – cattura fortemente premiata ─────────────────
        if endgame:
            if _endgame_advantage(state, player):
                weight += _W_ENDGAME_CAPTURE * 2
            else:
                weight += _W_ENDGAME_CAPTURE

    else:
        # ── mossa non-catturante ──────────────────────────────────────────
        dest_level = _level(game, tr, tc)
        src_level  = _level(game, fr, fc)
        delta = dest_level - src_level

        # ── R4: fuga verso l'esterno se accerchiati / meno esterni ────────
        encircled = _is_encircled_or_outleveled(game, state, fr, fc, player)

        if delta > 0:
            if encircled:
                # Fuga aggressiva: bonus molto più alto per uscire
                weight += _W_ESCAPE_OUTER * delta
            else:
                weight += _W_MOVE_OUTER * delta

            # ── R7: pressione esterna dopo lo spostamento ─────────────────
            weight += _W_OUTER_PRESSURE * _outer_pressure_score(game, state, tr, tc, player)

            # ── R5: fase finale – mosse verso esterno ─────────────────────
            if endgame:
                mult = 2 if _endgame_advantage(state, player) else 1
                weight += _W_ENDGAME_OUTER_MOVE * mult

        else:
            # ── R6: anti-cluster ──────────────────────────────────────────
            # Mossa verso il centro: penalizzata fortemente se siamo già
            # agglomerati lì (il difetto principale del vecchio giocatore).
            if _is_clustered_near_center(game, state, fr, fc, player):
                weight *= 0.01   # quasi impossibile: non vogliamo aggravare il cluster
            elif endgame:
                # In endgame penalizziamo comunque le mosse verso il centro
                weight *= 0.1

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

    Restituisce +1 (vittoria root_player), -1 (sconfitta), 0 (draw/limite).
    """
    current_state = state
    ct = {
        root_player: dict(corner_tracker_p1),
        game.opponent(root_player): dict(corner_tracker_p2),
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
        weights = [_move_weight(game, current_state, m, current_player, tracker) for m in legal_moves]

        chosen = _weighted_choice(legal_moves, weights)
        current_state = game.result(current_state, chosen)

    winner = game.winner(current_state)
    if winner == root_player:
        return 1
    elif winner is not None:
        return -1
    # Valutazione rapida se limite raggiunto
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
# Ordinamento mosse per alpha-beta
# ─────────────────────────────────────────────────────────────────────────────

def _order_moves_mc(game, moves):
    """Catture esterne prima, poi mosse verso esterno."""
    def priority(m):
        (fr, fc), (tr, tc), is_cap = m
        if is_cap:
            return (0, -_level(game, tr, tc))
        return (1, -_level(game, tr, tc))
    caps    = sorted([m for m in moves if m[2]],     key=priority)
    noncaps = sorted([m for m in moves if not m[2]], key=priority)
    return caps + noncaps


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con valutazione MC ai nodi foglia
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alphabeta(game, state, depth, alpha, beta, maximizing, root_player,
               deadline, n_sims, ct_root, ct_opp):
    """Alpha-beta con valutazione MC ai nodi foglia. Lancia _Timeout se scaduto."""
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

_persistent_ct_root = None
_persistent_ct_opp  = None
_persistent_player  = None


def playerStrategy(game, state, timeout=3):
    """Strategia MC con iterative-deepening alpha-beta e timeout sicuro.

    Usa simulazioni Monte Carlo pesate come euristica ai nodi foglia,
    con regole fortemente pesate (R1–R7).
    """
    global _persistent_ct_root, _persistent_ct_opp, _persistent_player

    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    player   = state.to_move
    opponent = game.opponent(player)

    if _persistent_player != player:
        _persistent_player  = player
        _persistent_ct_root = _init_corner_tracker(game, state, player)
        _persistent_ct_opp  = _init_corner_tracker(game, state, opponent)

    ct_root = _persistent_ct_root
    ct_opp  = _persistent_ct_opp

    # ── R1: verifica mossa obbligata da angolo ────────────────────────────
    forced = _corner_constraint_move(game, state, player, ct_root)
    if forced is not None and forced in legal_moves:
        (fr, fc), (tr, tc), is_cap = forced
        if (fr, fc) in ct_root:
            ct_root[(fr, fc)] = 'done'
        return forced

    # ── Iterative deepening alpha-beta con MC ─────────────────────────────
    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move  = random.choice(legal_moves)
    best_value = -math.inf
    n_sims     = _MC_SIMULATIONS

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