import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con Iterative Deepening + Monte Carlo guidato alle foglie
#
# Il codice funziona correttamente sia quando il nostro giocatore è "Blue"
# sia quando è "Red": tutta la logica usa `player` come parametro esplicito
# e non assume mai un colore fisso.
#
# Regole strategiche (R1–R6):
#
#   R1. Preferire mosse che partono dal settore di casa del giocatore.
#       Blue: casa = Q1 (top-left, angolo (0,0)) + Q4 (bottom-right, angolo (7,7))
#       Red:  casa = Q2 (top-right, angolo (0,7)) + Q3 (bottom-left, angolo (7,0))
#
#   R2. Prediligi catture più esterne: livello-destinazione alto = periferia.
#
#   R3. Mosse verso l'esterno nei quadranti avversari ricevono peso maggiore.
#       Se tutte le catture disponibili sono troppo interne (livello ≤ soglia),
#       le mosse non-catturanti ricevono un bonus extra (riposizionamento).
#
#   R4. Avanzare verso zone con pedine avversarie esterne e scoperte
#       (nessuna nostra pedina copre quella zona verso il centro).
#
#   R5. Preferire catture nel settore di casa; preferire arretramenti-cattura
#       (dest_level < src_level) nel settore avversario.
#       Intensità proporzionale alla fase di gioco (forte all'inizio, debole alla fine).
#
#   R6. Tra le pedine che possono catturare, preferire quelle più lontane dal
#       centro (livello sorgente alto). Questa regola è SEPARATA da R2 perché
#       agisce sull'attaccante, non sulla destinazione.
#       Motivazione: le pedine esterne hanno già raggiunto posizioni periferiche
#       vantaggiose e devono essere usate per catturare, mentre quelle centrali
#       devono ancora avanzare verso l'esterno.
#
# Gestione PASS e bug-fix:
#   - Se il giocatore non ha mosse ma la partita non è terminata, si passa il
#     turno con "PASS" sia in alpha-beta che nei rollout MC.
#   - _pass_count in _alphabeta evita la ricorsione infinita se entrambi
#     i giocatori sono bloccati contemporaneamente.
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN       = 0.15   # secondi di margine prima del timeout
_INITIAL_PIECES    = 64     # pedine totali a inizio partita (32+32 su 8×8)

# ── Parametri Monte Carlo ────────────────────────────────────────────────────
_MC_SIMULATIONS    = 20      # simulazioni per foglia alpha-beta
_MC_ROLLOUT_DEPTH  = 26     # passi massimi per rollout

# ── Pesi statici (indipendenti dalla fase) ───────────────────────────────────
_W_CAP_BASE         = 15.0  # R2: bonus fisso per qualsiasi cattura
_W_CAP_DEST_LEVEL   =  9.0  # R2: catture più esterne (× livello destinazione)
_W_CAP_SRC_LEVEL    =  12.0  # R6: pedina attaccante più lontana dal centro (× livello sorgente)
_W_NONCAP_OUTER_ADV =  10.0  # R3: non-catturante verso esterno in settore avversario
_W_NONCAP_OUTER_OWN =  4.0  # R3: non-catturante verso esterno in settore di casa
_W_HOME_SECTOR      =  3.0  # R1: bonus se la mossa parte dal settore di casa
_W_PUSH_EXPOSED     =  6.0  # R4: avanzare verso pedine avversarie esterne scoperte
_W_NONCAP_INNER_BONUS = 3.0 # R3: bonus non-catturanti se catture troppo interne

# ── Pesi dinamici per R5 (scalati per phase) ────────────────────────────────
_W_R5_HOME_CAP     = 10.0   # R5: bonus cattura in settore di casa
_W_R5_AWAY_RETREAT =  8.0   # R5: bonus arretramento-cattura in settore avversario

# ── Soglie ───────────────────────────────────────────────────────────────────
# R3: livello sotto cui una cattura è "troppo interna"
_INNER_CAP_THRESHOLD = 3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers di base
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """Livello massimo della scacchiera (angoli)."""
    return game.distance_levels[0][0]


def _home_sector(player, r, c, size):
    """True se (r,c) appartiene al settore di casa di `player`.

    Settore di casa = quadranti che contengono gli angoli occupati
    dal giocatore nella disposizione iniziale a scacchiera:

      Blue → celle con (r+c) pari  → angoli (0,0) e (7,7)
               home = Q1 (r<half, c<half)  OPPURE  Q4 (r>=half, c>=half)

      Red  → celle con (r+c) dispari → angoli (0,7) e (7,0)
               home = Q2 (r<half, c>=half)  OPPURE  Q3 (r>=half, c<half)

    La funzione non assume mai un colore fisso.
    """
    half = size // 2
    if player == "Blue":
        return (r < half and c < half) or (r >= half and c >= half)
    else:
        return (r < half and c >= half) or (r >= half and c < half)


def _game_phase(state):
    """Fase di gioco ∈ [0, 1]: 1.0 = inizio partita, ~0.0 = fine partita.

    Calcolata come pedine_totali_attuali / pedine_iniziali.
    Usata per scalare l'intensità di R5.
    """
    total = state.count("Blue") + state.count("Red")
    return total / _INITIAL_PIECES


def _has_own_piece_behind(game, state, player, r, c):
    """R4: True se esiste una nostra pedina adiacente verso il centro (livello inferiore)."""
    cur_level = _level(game, r, c)
    for dr, dc in game.DIRECTIONS:
        nr, nc = r + dr, c + dc
        if game.in_bounds(nr, nc):
            if _level(game, nr, nc) < cur_level and state.board[nr][nc] == player:
                return True
    return False


def _exposed_opponent_count(game, state, player, tr, tc):
    """R4: pedine avversarie esterne e scoperte nelle vicinanze della destinazione (tr,tc).

    'Esposta' = livello >= max_level - 2  E  nessuna pedina avversaria la copre verso il centro.
    """
    opponent  = game.opponent(player)
    threshold = _max_level(game) - 2
    count     = 0
    for dr, dc in game.DIRECTIONS:
        for dist in range(1, 3):
            nr, nc = tr + dr * dist, tc + dc * dist
            if not game.in_bounds(nr, nc):
                break
            cell = state.board[nr][nc]
            if cell == opponent:
                if (_level(game, nr, nc) >= threshold
                        and not _has_own_piece_behind(game, state, opponent, nr, nc)):
                    count += 1
                break
    return count


def _compute_caps_are_inner(game, moves):
    """True se esistono catture ma tutte hanno dest_level <= _INNER_CAP_THRESHOLD."""
    caps = [m for m in moves if m[2]]
    if not caps:
        return False
    return all(_level(game, m[1][0], m[1][1]) <= _INNER_CAP_THRESHOLD for m in caps)


# ─────────────────────────────────────────────────────────────────────────────
# Peso strategico di una mossa  (R1–R6)
# ─────────────────────────────────────────────────────────────────────────────

def _move_weight(game, state, player, move, caps_are_inner, phase):
    """Peso strategico di `move` per `player`.

    Parametri
    ---------
    caps_are_inner : bool
        True se tutte le catture disponibili hanno dest_level <= _INNER_CAP_THRESHOLD (R3).
    phase : float
        Fase di gioco ∈ [0,1]: 1.0=inizio, ~0=fine. Scala l'intensità di R5.
    """
    (fr, fc), (tr, tc), is_cap = move
    dest_level = _level(game, tr, tc)
    src_level  = _level(game, fr, fc)
    size       = state.size

    in_home_src = _home_sector(player, fr, fc, size)

    weight = 1.0   # floor: nessuna mossa viene mai esclusa
    #chaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat
    if dest_level < src_level:
        weight -= 2.0 * (src_level - dest_level)

    # ── Mosse catturanti ─────────────────────────────────────────────────────
    if is_cap:
        # R2: bonus fisso + catture più esterne preferite (destinazione)
        weight += _W_CAP_BASE + _W_CAP_DEST_LEVEL * dest_level

        # R6: preferire come attaccante la pedina più lontana dal centro.
        #     Agisce sul livello della SORGENTE, separato da R2.
        #     Pedine esterne (src_level alto) catturano prima di quelle centrali.
        weight += _W_CAP_SRC_LEVEL * src_level

        # R5 (scalata per fase):
        #   - in home: bonus per catturare (qualsiasi direzione)
        #   - in away: bonus per arretrare catturando (dest_level < src_level)
        r5_scale = phase
        if in_home_src:
            weight += _W_R5_HOME_CAP * r5_scale
        else:
            if dest_level < src_level:   # arretramento in territorio avversario
                weight += _W_R5_AWAY_RETREAT * r5_scale

    # ── Mosse non-catturanti ─────────────────────────────────────────────────
    else:
        # R3: nei quadranti avversari spingere verso l'esterno ha peso maggiore
        if not in_home_src:
            weight += _W_NONCAP_OUTER_ADV * dest_level
        else:
            weight += _W_NONCAP_OUTER_OWN * dest_level

        # R3: bonus extra se le catture disponibili sono tutte troppo interne
        if caps_are_inner:
            weight += _W_NONCAP_INNER_BONUS

        # R4: avanzare verso pedine avversarie esterne scoperte
        exposed = _exposed_opponent_count(game, state, player, tr, tc)
        weight += _W_PUSH_EXPOSED * exposed

    # R1: bonus se la mossa parte dal settore di casa
    if in_home_src:
        weight += _W_HOME_SECTOR

    return max(weight, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Selezione pesata (roulette wheel) per i rollout MC
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_choice(game, state, player, moves, phase):
    """Seleziona una mossa con probabilità proporzionale al peso strategico R1-R6."""
    if not moves:
        return None
    caps_are_inner = _compute_caps_are_inner(game, moves)
    weights = [_move_weight(game, state, player, m, caps_are_inner, phase)
               for m in moves]
    total = sum(weights)
    r = random.uniform(0, total)
    cum = 0.0
    for move, w in zip(moves, weights):
        cum += w
        if r <= cum:
            return move
    return moves[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento mosse per alpha-beta
# ─────────────────────────────────────────────────────────────────────────────

def _order_moves(game, state, player, moves, phase):
    """Ordina le mosse in ordine decrescente di peso strategico R1-R6."""
    caps_are_inner = _compute_caps_are_inner(game, moves)
    return sorted(moves,
                  key=lambda m: -_move_weight(game, state, player, m,
                                              caps_are_inner, phase))


# ─────────────────────────────────────────────────────────────────────────────
# Rollout guidato
# ─────────────────────────────────────────────────────────────────────────────

def _rollout(game, state, root_player, max_depth):
    """Simula la partita per max_depth passi con mosse scelte per peso R1-R6.

    Ritorna:
      1.0  se root_player vince
      0.0  se root_player perde
      frazione di pedine proprie residue se il rollout esaurisce i passi
    """
    current    = state
    opponent   = game.opponent(root_player)
    pass_count = 0

    for _ in range(max_depth):
        if game.is_terminal(current):
            break

        mover = current.to_move
        moves = game.actions(current)

        if not moves:
            current = game.result(current, "PASS")
            pass_count += 1
            if pass_count >= 4:
                break
            continue

        pass_count = 0
        phase  = _game_phase(current)
        move   = _weighted_choice(game, current, mover, moves, phase)
        current = game.result(current, move)

    winner = game.winner(current)
    if winner == root_player:
        return 1.0
    if winner == opponent:
        return 0.0

    rp = current.count(root_player)
    op = current.count(opponent)
    return rp / (rp + op) if (rp + op) > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione MC di uno stato foglia
# ─────────────────────────────────────────────────────────────────────────────

def _mc_leaf_value(game, state, root_player, n_simulations, rollout_depth):
    """Media di N rollout guidati → valore in [-100, +100].

    Stati terminali certi restituiscono ±100_000 per dominare sui valori MC.
    """
    winner = game.winner(state)
    if winner == root_player:
        return 100_000
    if winner is not None:
        return -100_000

    total = sum(_rollout(game, state, root_player, rollout_depth)
                for _ in range(n_simulations))
    mean = total / n_simulations
    return (mean - 0.5) * 200.0


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con timeout e valutazione MC alle foglie
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alphabeta(game, state, depth, alpha, beta, maximizing,
               root_player, deadline, n_sims, rollout_depth,
               _pass_count=0):
    """Alpha-beta ricorsivo con MC alle foglie e gestione corretta del PASS.

    _pass_count conta i PASS consecutivi: se entrambi i giocatori non hanno
    mosse, si valuta come foglia per evitare ricorsione infinita.
    Quando si scende nell'albero normalmente, _pass_count si azzera.
    """
    if time.perf_counter() >= deadline:
        raise _Timeout()

    legal_moves = game.actions(state)

    # ── Gestione PASS ─────────────────────────────────────────────────────────
    if not legal_moves and not game.is_terminal(state):
        if _pass_count >= 2:
            return _mc_leaf_value(game, state, root_player, n_sims, rollout_depth), None
        passed = game.result(state, "PASS")
        return _alphabeta(game, passed, depth, alpha, beta, not maximizing,
                          root_player, deadline, n_sims, rollout_depth,
                          _pass_count + 1)

    # ── Nodo foglia ───────────────────────────────────────────────────────────
    if depth == 0 or game.is_terminal(state) or not legal_moves:
        return _mc_leaf_value(game, state, root_player, n_sims, rollout_depth), None

    mover   = state.to_move
    phase   = _game_phase(state)
    ordered = _order_moves(game, state, mover, legal_moves, phase)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(
                game, child, depth - 1, alpha, beta, False,
                root_player, deadline, n_sims, rollout_depth, 0
            )
            if child_val > value:
                value = child_val; best_moves = [move]
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
                root_player, deadline, n_sims, rollout_depth, 0
            )
            if child_val < value:
                value = child_val; best_moves = [move]
            elif child_val == value:
                best_moves.append(move)
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, (random.choice(best_moves) if best_moves else None)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """Strategia ibrida: Iterative Deepening Alpha-Beta + Monte Carlo guidato.

    Funziona correttamente sia quando il giocatore è "Blue" sia "Red".

    Regole strategiche attive
    ─────────────────────────
    R1. Mosse che partono dal settore di casa ricevono bonus.
    R2. Catture più esterne preferite (livello destinazione alto).
    R3. Movimenti verso esterno nei quadranti avversari; bonus non-catturanti
        se catture troppo interne.
    R4. Avanzare verso pedine avversarie esterne scoperte.
    R5. Catture preferite in home-sector; arretramenti-cattura preferiti in
        away-sector. Intensità proporzionale alla fase di gioco.
    R6. Tra le pedine che possono catturare, preferire quelle più lontane dal
        centro (livello sorgente alto). Regola separata da R2: agisce
        sull'ATTACCANTE, non sulla destinazione.
    """
    if game.is_terminal(state):
        return None

    legal_moves = game.actions(state)
    if not legal_moves:
        return "PASS"

    root_player = state.to_move
    deadline    = time.perf_counter() + timeout - _TIME_MARGIN
    best_move   = random.choice(legal_moves)

    depth = 1
    while True:
        if time.perf_counter() >= deadline:
            break
        n_sims = max(2, _MC_SIMULATIONS - (depth - 1))
        try:
            _, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, root_player,
                deadline, n_sims, _MC_ROLLOUT_DEPTH,
            )
            if move is not None:
                best_move = move
            depth += 1
        except _Timeout:
            break

    return best_move