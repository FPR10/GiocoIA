import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Iterative-deepening alpha-beta con regole strategiche avanzate
#
# Struttura:
#   • evaluate_state   – funzione euristica multi-feature
#   • order_moves      – ordinamento delle mosse (catture esterne prima)
#   • alphabeta        – alpha-beta con move ordering
#   • playerStrategy   – iterative deepening con guardia al tempo + regole
#
# Principi strategici incorporati:
#   P1.  Preferire catture con la pedina più lontana dal centro (src level alto)
#   P2.  Tra le catture, preferire la pedina avversaria più lontana dal centro
#   P3.  Mosse non-catturanti verso destinazioni più esterne
#   P4.  Pressione nelle zone esterne dove ci sono pedine avversarie esterne
#   P5.  Setup angolo: bonus se una mossa non-catturante apre una cattura
#
# Regole di fase (sovra-scrivono l'alpha-beta nelle fasi iniziali):
#   INITIAL  – prime mosse vincolate: la pedina d'angolo mangia le due vicine
#   TACTICAL – N mosse dopo INITIAL: difensivo nei quarti avversari,
#              aggressivo nei quarti propri
#   RETREAT  – se l'avversario ha pedine più esterne nella stessa zona,
#              spostati verso l'esterno per aggirarlo
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15   # secondi di margine di sicurezza prima del timeout

# ── Parametri di fase configurabili ──────────────────────────────────────────
# Numero di mosse NOSTRE nella fase tattica dopo quella iniziale
_TACTICAL_MOVES = 4   # regola 3: quante mosse in modalità tattica

# ── Pesi euristica ────────────────────────────────────────────────────────────
_W_PIECES          = 50
_W_MOBILITY        =  2
_W_CAPTURE_COUNT   =  5
_W_CAPTURE_SRC_OUT =  6   # P1: bonus per sorgente lontana dal centro
_W_CAPTURE_TGT_OUT =  4   # P2: bonus per target lontano dal centro
_W_CAPTURE_INNER   =  3   # legacy: bonus per catturare pedine interne
_W_MOVE_OUTER      =  3   # P3: bonus mosse non-catturanti verso esterno
_W_OUTER_PRESSURE  =  4   # P4: nostre pedine esterne vicino ad avversari
_W_CORNER_SETUP    =  5   # P5: setup angolo


# ─────────────────────────────────────────────────────────────────────────────
# Stato di partita (persistente tra chiamate a playerStrategy per la stessa
# istanza di gioco – si usa un dizionario globale indicizzato per id(game))
# ─────────────────────────────────────────────────────────────────────────────
_game_state_store = {}   # { game_id: GamePhaseState }


class _GamePhaseState:
    """Traccia le fasi strategiche della partita per un singolo giocatore."""

    def __init__(self, player):
        self.player = player
        # Contatore delle mosse effettive del giocatore (non del turno globale)
        self.our_move_count = 0
        # Quante mosse vincolate dell'INITIAL sono state eseguite
        self.initial_done = 0
        # Quante mosse del target INITIAL ci aspettiamo (2 o 4)
        self.initial_target = 0   # calcolato al primo accesso
        self.initial_computed = False
        # Contatore mosse nella fase TACTICAL
        self.tactical_count = 0
        # Flag: siamo ancora in fase INITIAL?
        self.in_initial = True
        # Flag: siamo ancora in fase TACTICAL?
        self.in_tactical = False


def _get_phase_state(game, player):
    key = (id(game), player)
    if key not in _game_state_store:
        _game_state_store[key] = _GamePhaseState(player)
    return _game_state_store[key]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers geometrici
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]   # angolo = massima distanza


def _corners(size):
    """Restituisce le quattro celle angolo della scacchiera."""
    s = size - 1
    return [(0, 0), (0, s), (s, 0), (s, s)]


def _our_corners(game, state, player):
    """Restituisce gli angoli occupati dal giocatore nello stato iniziale.

    Nella disposizione a scacchiera di Zola l'angolo (0,0) è sempre Blue
    (r+c pari → Blue), (0,1) è Red, ecc.  Usiamo la logica del board iniziale.
    """
    corners = _corners(state.size)
    return [
        (r, c) for (r, c) in corners
        if state.board[r][c] == player
        or _initial_owner(r, c) == player
    ]


def _initial_owner(r, c):
    """Restituisce il colore del proprietario iniziale della cella (r, c)."""
    return "Blue" if (r + c) % 2 == 0 else "Red"


def _corner_neighbors(r, c, size):
    """Le due celle adiacenti ortogonali all'angolo (r, c)."""
    neighbors = []
    if r == 0:
        neighbors.append((1, c))
    else:
        neighbors.append((r - 1, c))
    if c == 0:
        neighbors.append((r, 1))
    else:
        neighbors.append((r, c - 1))
    return [n for n in neighbors if 0 <= n[0] < size and 0 <= n[1] < size]


def _quadrant_of(r, c, size):
    """Restituisce il quadrante (0-3) cui appartiene la cella.

    I quadranti corrispondono ai 4 angoli:
      0 → (0,0), 1 → (0, size-1), 2 → (size-1, 0), 3 → (size-1, size-1)
    """
    mid = size / 2
    if r < mid and c < mid:
        return 0
    if r < mid and c >= mid:
        return 1
    if r >= mid and c < mid:
        return 2
    return 3


def _our_quadrants(size, player):
    """Quadranti il cui angolo appartiene a 'player' nella disposizione iniziale."""
    corners = _corners(size)
    return {
        _quadrant_of(r, c, size)
        for (r, c) in corners
        if _initial_owner(r, c) == player
    }


# ─────────────────────────────────────────────────────────────────────────────
# REGOLA INITIAL: forza le prime mosse degli angoli nostri
# ─────────────────────────────────────────────────────────────────────────────

def _initial_constraint_move(game, state, player, phase):
    """Restituisce la prossima mossa vincolata della fase INITIAL, o None.

    La fase INITIAL impone che la pedina d'angolo (nei quarti nostri) mangi
    le due pedine avversarie adiacenti, nell'ordine in cui la cattura è
    disponibile.  Se la prima cattura è stata subito replicata dall'avversario
    e la pedina d'angolo non c'è più, la fase termina anticipatamente.
    """
    if not phase.in_initial:
        return None

    size = state.size
    our_corner_cells = [
        (r, c) for (r, c) in _corners(size)
        if _initial_owner(r, c) == player
    ]

    # Per ogni angolo nostro, cerchiamo una cattura verso i vicini adiacenti
    # disponibile ORA (la pedina d'angolo potrebbe essere già stata mangiata).
    candidate_moves = []
    for (cr, cc) in our_corner_cells:
        if state.board[cr][cc] != player:
            # La nostra pedina d'angolo non c'è più: saltiamo
            continue
        neighbors = _corner_neighbors(cr, cc, size)
        for (nr, nc) in neighbors:
            # La cattura queen-like: la pedina d'angolo cattura la vicina
            # adiacente se si trova a livello <= nostro e la cella è avversaria.
            cap_move = ((cr, cc), (nr, nc), True)
            legal = game.actions(state)
            if cap_move in legal:
                candidate_moves.append(cap_move)

    if not candidate_moves:
        # Nessuna mossa vincolata disponibile: uscita dalla fase INITIAL
        phase.in_initial = False
        phase.in_tactical = True
        return None

    # Priorità: prima le catture che fanno avanzare il contatore in modo
    # simmetrico (prendiamo la prima disponibile).
    move = candidate_moves[0]
    phase.initial_done += 1
    # Stimiamo il target: al massimo 4 (2 angoli × 2 vicini)
    # Usciamo dalla fase INITIAL dopo aver eseguito tutte le catture disponibili
    # o quando non ce ne sono più.
    return move


def _check_initial_completion(phase, game, state, player):
    """Controlla se la fase INITIAL è completata e aggiorna lo stato."""
    if not phase.in_initial:
        return
    size = state.size
    our_corner_cells = [
        (r, c) for (r, c) in _corners(size)
        if _initial_owner(r, c) == player
    ]
    has_any = False
    legal = game.actions(state)
    for (cr, cc) in our_corner_cells:
        if state.board[cr][cc] != player:
            continue
        for (nr, nc) in _corner_neighbors(cr, cc, size):
            if ((cr, cc), (nr, nc), True) in legal:
                has_any = True
                break
        if has_any:
            break
    if not has_any:
        phase.in_initial = False
        phase.in_tactical = True


# ─────────────────────────────────────────────────────────────────────────────
# REGOLA TACTICAL (regola 3): difensivo nei quarti avversari, aggressivo
# nei quarti nostri
# ─────────────────────────────────────────────────────────────────────────────

def _tactical_move(game, state, player, phase):
    """Logica tattica post-INITIAL per _TACTICAL_MOVES mosse nostre.

    - Quarti avversari → difensivo: prova ad arretrare DIETRO una pedina
      avversaria (mossa non-catturante verso l'esterno nella zona).
    - Quarti nostri → aggressivo: cattura quanto più possibile, preferendo
      sorgente più esterna (P1) e target più esterno (P2).

    Restituisce la mossa scelta o None (fallback ad alpha-beta).
    """
    if not phase.in_tactical or phase.tactical_count >= _TACTICAL_MOVES:
        phase.in_tactical = False
        return None

    size = state.size
    our_quads = _our_quadrants(size, player)
    opponent = game.opponent(player)
    legal = game.actions(state)

    # ── Aggressivo nei quarti nostri ─────────────────────────────────────────
    # Catture disponibili NEI quarti nostri, ordinate per:
    #   1. livello sorgente (più alto = più esterna, P1)
    #   2. livello target (più alto = più esterna, P2)
    aggressive_caps = [
        m for m in legal
        if m[2] and _quadrant_of(m[0][0], m[0][1], size) in our_quads
    ]
    if aggressive_caps:
        best = max(
            aggressive_caps,
            key=lambda m: (_level(game, m[0][0], m[0][1]),
                           _level(game, m[1][0], m[1][1]))
        )
        phase.tactical_count += 1
        return best

    # Catture in qualsiasi quadrante (fallback aggressivo globale)
    all_caps = [m for m in legal if m[2]]
    if all_caps:
        best = max(
            all_caps,
            key=lambda m: (_level(game, m[0][0], m[0][1]),
                           _level(game, m[1][0], m[1][1]))
        )
        phase.tactical_count += 1
        return best

    # ── Difensivo nei quarti avversari ───────────────────────────────────────
    # Mossa non-catturante che porta la nostra pedina DIETRO una avversaria
    # (la nostra cella destinazione ha livello maggiore dell'avversaria più vicina
    # nella stessa direzione).
    adv_quads = {0, 1, 2, 3} - our_quads
    defensive_moves = []
    for m in legal:
        if m[2]:
            continue
        (fr, fc), (tr, tc), _ = m
        if _quadrant_of(fr, fc, size) not in adv_quads:
            continue
        dest_level = _level(game, tr, tc)
        # Controlla se ci sono pedine avversarie nella stessa zona con livello < dest_level
        for r2 in range(size):
            for c2 in range(size):
                if state.board[r2][c2] == opponent:
                    if _quadrant_of(r2, c2, size) == _quadrant_of(fr, fc, size):
                        if _level(game, r2, c2) < dest_level:
                            defensive_moves.append(m)
                            break

    if defensive_moves:
        best = max(defensive_moves, key=lambda m: _level(game, m[1][0], m[1][1]))
        phase.tactical_count += 1
        return best

    # Nessuna mossa tattica trovata: delega ad alpha-beta
    phase.tactical_count += 1
    return None


# ─────────────────────────────────────────────────────────────────────────────
# REGOLA RETREAT (regola 4): aggira l'avversario che arretra
# ─────────────────────────────────────────────────────────────────────────────

def _retreat_move(game, state, player):
    """Se l'avversario ha pedine più esterne nella stessa zona, spostati
    verso l'esterno per aggirarlo.

    Restituisce la mossa migliore di tipo "arretramento difensivo" o None.
    """
    size = state.size
    opponent = game.opponent(player)
    legal = game.actions(state)

    # Per ogni nostra pedina, controlla se nella stessa zona (quadrante) c'è
    # almeno un avversario più esterno (livello più alto).
    retreat_moves = []
    for m in legal:
        if m[2]:
            continue   # solo mosse non-catturanti
        (fr, fc), (tr, tc), _ = m
        our_level = _level(game, fr, fc)
        dest_level = _level(game, tr, tc)
        quad = _quadrant_of(fr, fc, size)
        # Verifica: c'è un avversario con livello > our_level nello stesso quadrante?
        for r2 in range(size):
            for c2 in range(size):
                if state.board[r2][c2] == opponent:
                    if _quadrant_of(r2, c2, size) == quad:
                        if _level(game, r2, c2) > our_level and dest_level > our_level:
                            retreat_moves.append(m)
                            break

    if not retreat_moves:
        return None

    # Scegli la mossa che porta più all'esterno
    return max(retreat_moves, key=lambda m: _level(game, m[1][0], m[1][1]))


# ─────────────────────────────────────────────────────────────────────────────
# Feature euristiche (P1–P5 aggiornate)
# ─────────────────────────────────────────────────────────────────────────────

def _capture_src_outer_bonus(game, captures):
    """P1 – catture la cui sorgente è lontana dal centro."""
    return sum(_level(game, fr, fc) for ((fr, fc), _, _) in captures)


def _capture_tgt_outer_bonus(game, captures):
    """P2 – catture il cui target è lontano dal centro."""
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)


def _capture_inner_target_bonus(game, captures):
    """Legacy P4 – premia le catture di pedine avversarie molto interne."""
    ml = _max_level(game)
    return sum(ml - _level(game, tr, tc) for (_, (tr, tc), flag) in captures if flag)


def _move_outer_bonus(game, non_captures):
    """P3 – mosse non-catturanti verso l'esterno."""
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in non_captures)


def _outer_pressure_bonus(game, state, player):
    """P4 – nostre pedine esterne vicino ad avversari esterni."""
    opponent = game.opponent(player)
    threshold = _max_level(game) - 2
    bonus = 0
    for r in range(state.size):
        for c in range(state.size):
            if state.board[r][c] != player:
                continue
            if _level(game, r, c) < threshold:
                continue
            for dr, dc in game.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if game.in_bounds(nr, nc) and state.board[nr][nc] == opponent:
                    if _level(game, nr, nc) >= threshold:
                        bonus += 1
    return bonus


def _corner_setup_bonus(game, state, player, non_captures):
    """P5 – mosse non-catturanti che aprono catture in periferia."""
    if state.to_move != player:
        return 0
    ml = _max_level(game)
    threshold = ml - 1
    bonus = 0
    for move in non_captures:
        child = game.result(state, move)
        new_captures = [m for m in game._actions_for_player(child, player) if m[2]]
        bonus += sum(1 for (_, (tr, tc), _) in new_captures
                     if _level(game, tr, tc) >= threshold)
    return bonus


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica principale
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

    root_moves   = game._actions_for_player(state, root_player)
    opp_moves    = game._actions_for_player(state, opponent)

    root_caps    = [m for m in root_moves if m[2]]
    opp_caps     = [m for m in opp_moves  if m[2]]
    root_noncaps = [m for m in root_moves if not m[2]]
    opp_noncaps  = [m for m in opp_moves  if not m[2]]

    # P1: sorgente lontana dal centro per le catture
    cap_src_out = (
        _capture_src_outer_bonus(game, root_caps)
        - _capture_src_outer_bonus(game, opp_caps)
    )
    # P2: target lontano dal centro per le catture
    cap_tgt_out = (
        _capture_tgt_outer_bonus(game, root_caps)
        - _capture_tgt_outer_bonus(game, opp_caps)
    )
    # legacy inner target
    cap_inner = (
        _capture_inner_target_bonus(game, root_caps)
        - _capture_inner_target_bonus(game, opp_caps)
    )
    # P3: non-captures verso esterno
    move_outer = (
        _move_outer_bonus(game, root_noncaps)
        - _move_outer_bonus(game, opp_noncaps)
    )
    # P4: pressione esterna
    outer_pressure = (
        _outer_pressure_bonus(game, state, root_player)
        - _outer_pressure_bonus(game, state, opponent)
    )
    # P5: setup angolo
    corner_setup = _corner_setup_bonus(game, state, root_player, root_noncaps)

    score = (
        _W_PIECES          * (root_pieces - opp_pieces)
      + _W_MOBILITY        * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT   * (len(root_caps) - len(opp_caps))
      + _W_CAPTURE_SRC_OUT * cap_src_out
      + _W_CAPTURE_TGT_OUT * cap_tgt_out
      + _W_CAPTURE_INNER   * cap_inner
      + _W_MOVE_OUTER      * move_outer
      + _W_OUTER_PRESSURE  * outer_pressure
      + _W_CORNER_SETUP    * corner_setup
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento delle mosse (P1 + P2 aggiornati)
# ─────────────────────────────────────────────────────────────────────────────

def order_moves(game, moves):
    """Ordina le mosse per massimizzare l'efficacia del pruning alpha-beta.

    Priorità catture:
      1. sorgente più esterna (livello alto, P1)
      2. target più esterno (livello alto, P2)
    Priorità non-catturanti:
      destinazione più esterna.
    """
    def move_priority(m):
        (fr, fc), (tr, tc), is_cap = m
        if is_cap:
            return (0, -_level(game, fr, fc), -_level(game, tr, tc))
        else:
            return (1, -_level(game, tr, tc), 0)

    return sorted(moves, key=move_priority)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con time check
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alphabeta(game, state, depth, alpha, beta, maximizing, root_player, deadline):
    if time.perf_counter() >= deadline:
        raise _Timeout()

    legal_moves = game.actions(state)

    if depth == 0 or game.is_terminal(state) or not legal_moves:
        return evaluate_state(game, state, root_player), None

    ordered = order_moves(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(game, child, depth - 1, alpha, beta,
                                      False, root_player, deadline)
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
            child_val, _ = _alphabeta(game, child, depth - 1, alpha, beta,
                                      True, root_player, deadline)
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
# Entry-point: iterative deepening con guardia al tempo + regole di fase
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """Strategia principale con:
    - Fase INITIAL: mosse vincolate (pedine d'angolo mangiano le vicine)
    - Fase TACTICAL: N mosse difensive/aggressive dopo INITIAL
    - Regola RETREAT: aggiro dell'avversario che arretra
    - Iterative-deepening alpha-beta per il resto della partita
    """
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    player = state.to_move
    phase  = _get_phase_state(game, player)

    # ── Fase INITIAL ─────────────────────────────────────────────────────────
    if phase.in_initial:
        _check_initial_completion(phase, game, state, player)

    if phase.in_initial:
        forced = _initial_constraint_move(game, state, player, phase)
        if forced is not None:
            phase.our_move_count += 1
            return forced
        # Se non ci sono mosse vincolate disponibili, chiudiamo la fase
        phase.in_initial = False
        phase.in_tactical = True

    # ── Fase TACTICAL ─────────────────────────────────────────────────────────
    if phase.in_tactical and phase.tactical_count < _TACTICAL_MOVES:
        tactical = _tactical_move(game, state, player, phase)
        if tactical is not None and tactical in legal_moves:
            phase.our_move_count += 1
            return tactical
        # Se non trovata, usciamo dalla fase tattica e andiamo ad alpha-beta
        phase.in_tactical = False

    # ── Regola RETREAT (regola 4) ─────────────────────────────────────────────
    # Si applica solo se non ci sono catture disponibili per noi
    our_caps = [m for m in legal_moves if m[2]]
    if not our_caps:
        retreat = _retreat_move(game, state, player)
        if retreat is not None and retreat in legal_moves:
            phase.our_move_count += 1
            return retreat

    # ── Iterative-deepening alpha-beta ────────────────────────────────────────
    deadline   = time.perf_counter() + timeout - _TIME_MARGIN
    best_move  = random.choice(legal_moves)
    best_value = -math.inf

    depth = 1
    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, player,
                deadline,
            )
            if move is not None:
                best_move  = move
                best_value = value
            depth += 1
        except _Timeout:
            break

    phase.our_move_count += 1
    return best_move