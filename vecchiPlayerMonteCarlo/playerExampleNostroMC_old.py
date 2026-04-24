import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con Iterative Deepening + Monte Carlo guidato (con pesi)
#
# Struttura:
#   • Fase di apertura obbligata  – vincolo angoli (regola a.)
#   • Simulazione MC guidata      – rollout non casuali ma diretti da regole
#                                   PESATE tramite RuleWeights
#   • mc_evaluate                 – chiama le simulazioni per stimare la bontà
#                                   di uno stato nodo foglia
#   • alphabeta                   – alpha-beta con move ordering + MC al foglio
#   • playerStrategy              – iterative deepening con guardia al tempo
#
# Principi strategici incorporati nelle simulazioni MC (tutti pesabili):
#   W1. Peso cattura con source esterno (livello source alto)
#   W2. Peso cattura con target esterno (livello target alto)
#   W3. Comportamento aggressivo nel proprio settore (mosse non catturanti)
#   W4. Adattamento se avversario esterno è nelle vicinanze
#   W5. Peso della componente "pedine" nella quick-eval dei rollout non conclusi
#   W6. Peso della componente "mobilità" nella quick-eval dei rollout non conclusi
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN      = 0.18   # secondi di margine prima del timeout
_MC_SIMULATIONS   = 16     # simulazioni MC per nodo foglia (bilanciato con tempo)
_MC_ROLLOUT_DEPTH = 20     # passi massimi di ogni rollout
_OPENING_MOVES    = 4      # mosse obbligate massime dalla fase di apertura
_POST_OPENING_N   = 8      # turni di "fase tattica" dopo l'apertura


# ─────────────────────────────────────────────────────────────────────────────
# Pesi delle regole – modifica questi valori per cambiare la strategia MC
# ─────────────────────────────────────────────────────────────────────────────

class RuleWeights:
    """Pesi assegnabili alle regole che guidano i rollout Monte Carlo.

    Tutti i valori sono float non negativi.  Il valore di default è 1.0 per
    ogni peso, che riproduce il comportamento originale.

    Parametri
    ---------
    w1_capture_src : float
        Importanza del livello della pedina *attaccante* nella scelta delle
        catture.  Aumentarlo privilegia attacchi dalle pedine più esterne.

    w2_capture_tgt : float
        Importanza del livello del *bersaglio* nella scelta delle catture.
        Aumentarlo privilegia l'eliminazione delle pedine avversarie più esterne.

    w3_aggressive_own_sector : float
        Spinta ad uscire verso l'esterno con mosse non-catturanti quando la
        pedina si trova nel proprio settore.  Diminuirlo rende il giocatore
        più "conservativo".

    w4_evade_external_enemy : float
        Priorità delle mosse di aggiramento quando un avversario più esterno è
        adiacente.  Impostarlo a 0 disabilita completamente questa regola.

    w5_piece_eval : float
        Peso della differenza di pedine nella valutazione rapida al termine
        del rollout (quick-eval).  Il complementare è w6_mobility_eval.

    w6_mobility_eval : float
        Peso della differenza di mobilità nella quick-eval.
        w5 e w6 vengono normalizzati automaticamente (somma → 1.0),
        quindi solo il loro rapporto conta.
    """

    def __init__(
        self,
        w1_capture_src: float          = 23.0,
        w2_capture_tgt: float          = 14.0,
        w3_aggressive_own_sector: float = 9.0,
        w4_evade_external_enemy: float  = 11.0,
        w5_piece_eval: float            = 0.7,
        w6_mobility_eval: float         = 0.3,
    ):
        self.w1_capture_src           = w1_capture_src
        self.w2_capture_tgt           = w2_capture_tgt
        self.w3_aggressive_own_sector = w3_aggressive_own_sector
        self.w4_evade_external_enemy  = w4_evade_external_enemy
        self.w5_piece_eval            = w5_piece_eval
        self.w6_mobility_eval         = w6_mobility_eval

    # Normalizza w5/w6 in modo che la loro somma sia sempre 1.0
    @property
    def piece_eval_norm(self) -> float:
        total = self.w5_piece_eval + self.w6_mobility_eval
        return self.w5_piece_eval / total if total > 0 else 1.0

    @property
    def mobility_eval_norm(self) -> float:
        total = self.w5_piece_eval + self.w6_mobility_eval
        return self.w6_mobility_eval / total if total > 0 else 0.0


# ── Pesi di default usati da playerStrategy ──────────────────────────────────
# Modifica questi valori per regolare il comportamento senza toccare il codice.

DEFAULT_WEIGHTS = RuleWeights(
    w1_capture_src           = 1.0,   # importanza del source esterno per cattura
    w2_capture_tgt           = 1.0,   # importanza del target esterno per cattura
    w3_aggressive_own_sector = 1.0,   # aggressività nel proprio settore
    w4_evade_external_enemy  = 1.0,   # evasione da avversari esterni vicini
    w5_piece_eval            = 0.7,   # peso pedine nella quick-eval
    w6_mobility_eval         = 0.3,   # peso mobilità nella quick-eval
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers geometrici
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]   # angolo = massimo livello


def _get_corner_positions(size):
    """Restituisce le 4 posizioni angolari della scacchiera."""
    s = size - 1
    return [(0, 0), (0, s), (s, 0), (s, s)]


def _find_initial_corners(game, player):
    """Identifica gli angoli iniziali del giocatore (indipendente dal colore).

    Su scacchiera 8x8 con disposizione a scacchiera: Blue occupa celle (r+c)%2==0,
    Red occupa celle (r+c)%2==1.  Gli angoli sono (0,0),(0,7),(7,0),(7,7).
    (0,0): (0+0)%2=0 → Blue;  (0,7): (0+7)%2=1 → Red;
    (7,0): (7+0)%2=1 → Red;   (7,7): (7+7)%2=0 → Blue.
    """
    size = game.size
    corners = _get_corner_positions(size)
    def initial_color(r, c):
        return "Blue" if (r + c) % 2 == 0 else "Red"
    return [(r, c) for (r, c) in corners if initial_color(r, c) == player]


def _get_player_quarters(game, player):
    """Restituisce i due 'quarti' della scacchiera appartenenti al giocatore."""
    size = game.size
    half = size // 2
    player_corners = _find_initial_corners(game, player)
    quarters = []
    for (r, c) in player_corners:
        r_start = 0    if r < half else half
        c_start = 0    if c < half else half
        quarters.append((r_start, r_start + half, c_start, c_start + half))
    return quarters


def _in_quarter(r, c, quarter):
    r0, r1, c0, c1 = quarter
    return r0 <= r < r1 and c0 <= c < c1


def _is_own_quarter(game, r, c, player):
    for q in _get_player_quarters(game, player):
        if _in_quarter(r, c, q):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Fase di apertura: vincolo angoli (regola a.)
# ─────────────────────────────────────────────────────────────────────────────

def _opening_forced_move(game, state, player, opening_move_count):
    """Restituisce la mossa obbligata dalla fase di apertura, o None."""
    if opening_move_count >= _OPENING_MOVES:
        return None

    all_moves = game.actions(state)
    captures = [m for m in all_moves if m[2]]

    ml = _max_level(game)
    corner_captures = [
        m for m in captures
        if _level(game, m[0][0], m[0][1]) >= ml - 1
    ]

    if corner_captures:
        corner_captures.sort(key=lambda m: -_level(game, m[0][0], m[0][1]))
        return corner_captures[0]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Selezione mossa guidata per i rollout MC  (con pesi)
# ─────────────────────────────────────────────────────────────────────────────

def _guided_move(game, state, player, phase_turn, weights: RuleWeights):
    """Seleziona la mossa migliore secondo le regole guidate PESATE per i rollout.

    Il punteggio di ogni mossa è calcolato usando i pesi in `weights` invece
    di priorità fisse.  Questo consente di dare importanza diversa a ogni
    principio strategico senza modificare la struttura del codice.

    Catture:
        score = w1 * src_level + w2 * tgt_level

    Mosse non catturanti:
        score = w4 * evade_bonus + w3 * own_sector_bonus + dest_level
    """
    all_moves = game._actions_for_player(state, player)
    if not all_moves:
        return None

    captures    = [m for m in all_moves if m[2]]
    noncaptures = [m for m in all_moves if not m[2]]
    opponent    = game.opponent(player)

    # ── Catture: punteggio pesato su source level (W1) e target level (W2) ──
    if captures:
        def capture_score(m):
            src_level = _level(game, m[0][0], m[0][1])
            tgt_level = _level(game, m[1][0], m[1][1])
            return weights.w1_capture_src * src_level + weights.w2_capture_tgt * tgt_level

        best_score = max(capture_score(m) for m in captures)
        best_caps  = [m for m in captures if capture_score(m) == best_score]
        return random.choice(best_caps)

    # ── Nessuna cattura: mosse non-catturanti guidate ────────────────────────
    if not noncaptures:
        return None

    def has_external_enemy_nearby(r, c):
        own_level = _level(game, r, c)
        for dr, dc in game.DIRECTIONS:
            nr, nc = r + dr, c + dc
            if game.in_bounds(nr, nc):
                if state.board[nr][nc] == opponent:
                    if _level(game, nr, nc) > own_level:
                        return True
        return False

    def noncap_score(m):
        fr, fc = m[0]
        tr, tc = m[1]
        dest_level = _level(game, tr, tc)
        in_own     = _is_own_quarter(game, fr, fc, player)

        # W4: bonus se c'è un avversario esterno vicino (evasione)
        evade_bonus = weights.w4_evade_external_enemy * dest_level \
                      if has_external_enemy_nearby(fr, fc) else 0.0

        # W3: bonus aggressività nel proprio settore
        sector_bonus = weights.w3_aggressive_own_sector * dest_level \
                       if in_own else 0.0

        # Il dest_level è comunque incluso come base comune
        return evade_bonus + sector_bonus + dest_level

    best_score = max(noncap_score(m) for m in noncaptures)
    best_ncs   = [m for m in noncaptures if noncap_score(m) == best_score]
    return random.choice(best_ncs)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout MC guidato (singola simulazione)
# ─────────────────────────────────────────────────────────────────────────────

def _mc_rollout(game, state, root_player, depth, phase_turn, weights: RuleWeights):
    """Esegue un rollout guidato fino a depth passi o terminazione.

    Restituisce:
      +1  se root_player vince
      -1  se l'avversario vince
       0  se partita non terminata entro depth passi (valutiamo con quick_eval)
    """
    current = state
    for _ in range(depth):
        winner = game.winner(current)
        if winner == root_player:
            return 1.0
        if winner is not None:
            return -1.0

        player = current.to_move
        move   = _guided_move(game, current, player, phase_turn, weights)
        if move is None:
            current = game.pass_turn(current) if not game.actions(current) else current
            break
        current   = game.result(current, move)
        phase_turn += 1

    winner = game.winner(current)
    if winner == root_player:
        return 1.0
    if winner is not None:
        return -1.0
    return _quick_eval_normalized(game, current, root_player, weights)


def _quick_eval_normalized(game, state, root_player, weights: RuleWeights):
    """Valutazione rapida normalizzata in [-1, +1] per rollout non conclusi.

    Usa i pesi W5 e W6 (normalizzati internamente a somma 1.0) per bilanciare
    differenza di pedine e differenza di mobilità.
    """
    opponent = game.opponent(root_player)
    rp = state.count(root_player)
    op = state.count(opponent)
    total = rp + op
    if total == 0:
        return 0.0

    piece_score = (rp - op) / total

    rp_moves  = len(game._actions_for_player(state, root_player))
    op_moves  = len(game._actions_for_player(state, opponent))
    mob_total = rp_moves + op_moves
    mob_score = (rp_moves - op_moves) / mob_total if mob_total > 0 else 0.0

    return weights.piece_eval_norm * piece_score + weights.mobility_eval_norm * mob_score


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione MC di uno stato nodo-foglia
# ─────────────────────────────────────────────────────────────────────────────

def mc_evaluate(game, state, root_player, n_sims, rollout_depth, phase_turn,
                deadline, weights: RuleWeights):
    """Stima il valore di uno stato tramite n_sims rollout guidati.

    Restituisce un valore in [-100000, +100000] compatibile con alpha-beta.
    """
    winner = game.winner(state)
    if winner == root_player:
        return 100_000
    if winner is not None:
        return -100_000

    total = 0.0
    for _ in range(n_sims):
        if time.perf_counter() >= deadline:
            break
        total += _mc_rollout(game, state, root_player, rollout_depth,
                              phase_turn, weights)

    avg = total / n_sims if n_sims > 0 else 0.0
    return avg * 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Move ordering per alpha-beta
# ─────────────────────────────────────────────────────────────────────────────

def _order_moves(game, moves, weights: RuleWeights):
    """Ordina le mosse per massimizzare il pruning.

    I pesi W1 e W2 vengono usati anche qui per rendere coerente l'ordinamento
    con la politica dei rollout.
    """
    def score(m):
        (fr, fc), (tr, tc), is_cap = m
        src = _level(game, fr, fc)
        tgt = _level(game, tr, tc)
        if is_cap:
            weighted = weights.w1_capture_src * src + weights.w2_capture_tgt * tgt
            return (0, -weighted)
        return (1, -tgt)

    return sorted(moves, key=score)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con MC al foglio
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alphabeta(game, state, depth, alpha, beta, maximizing,
               root_player, deadline, phase_turn,
               n_sims, rollout_depth, weights: RuleWeights):
    """Alpha-beta ricorsivo con valutazione MC alle foglie."""
    if time.perf_counter() >= deadline:
        raise _Timeout()

    legal_moves = game.actions(state)

    if depth == 0 or game.is_terminal(state) or not legal_moves:
        return mc_evaluate(game, state, root_player,
                           n_sims, rollout_depth, phase_turn,
                           deadline, weights), None

    ordered    = _order_moves(game, legal_moves, weights)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(
                game, child, depth - 1, alpha, beta, False,
                root_player, deadline, phase_turn + 1,
                n_sims, rollout_depth, weights
            )
            if child_val > value:
                value      = child_val
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
                root_player, deadline, phase_turn + 1,
                n_sims, rollout_depth, weights
            )
            if child_val < value:
                value      = child_val
                best_moves = [move]
            elif child_val == value:
                best_moves.append(move)
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, (random.choice(best_moves) if best_moves else None)


# ─────────────────────────────────────────────────────────────────────────────
# Stima del turno di partita (per decidere la fase)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_game_turn(game, state):
    """Stima approssimativa del turno corrente basata sui pezzi rimasti."""
    size        = game.size
    total_start = size * size // 2
    remaining   = state.count(state.to_move)
    return total_start - remaining


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point: iterative deepening + apertura obbligata + MC
# ─────────────────────────────────────────────────────────────────────────────

# Contatore globale delle mosse di apertura (persistente tra le chiamate)
_opening_counter = {}


def playerStrategy(game, state, timeout=3, weights: RuleWeights = None):
    """Strategia con iterative-deepening alpha-beta + Monte Carlo guidato.

    Parametri
    ---------
    game, state, timeout
        Come atteso dal framework ZolaGame.
    weights : RuleWeights, opzionale
        Pesi delle regole MC.  Se None, usa DEFAULT_WEIGHTS.
        Esempio di utilizzo con pesi personalizzati:

            from playerExampleNostroMC import playerStrategy, RuleWeights

            my_weights = RuleWeights(
                w1_capture_src           = 2.0,   # privilegia molto le pedine esterne
                w2_capture_tgt           = 0.5,   # meno importanza al target esterno
                w3_aggressive_own_sector = 1.5,
                w4_evade_external_enemy  = 0.0,   # disabilita l'evasione
                w5_piece_eval            = 0.9,
                w6_mobility_eval         = 0.1,
            )

            def myStrategy(game, state, timeout=3):
                return playerStrategy(game, state, timeout, weights=my_weights)

    Fasi:
      1. Apertura obbligata (max _OPENING_MOVES mosse): catture dagli angoli
      2. Fase tattica post-apertura (_POST_OPENING_N turni): MC più aggressivo
      3. Fase generale: alpha-beta + MC con rollout guidati
    """
    global _opening_counter

    if weights is None:
        weights = DEFAULT_WEIGHTS

    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    player   = state.to_move
    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    # ── Inizializza il contatore di apertura per questo giocatore ────────────
    if player not in _opening_counter:
        _opening_counter[player] = 0

    opening_count = _opening_counter[player]

    # ── Fase di apertura obbligata ───────────────────────────────────────────
    forced = _opening_forced_move(game, state, player, opening_count)
    if forced is not None and forced in legal_moves:
        _opening_counter[player] += 1
        return forced

    # ── Stima fase di partita ────────────────────────────────────────────────
    phase_turn = _estimate_game_turn(game, state)

    if phase_turn < _POST_OPENING_N:
        n_sims        = _MC_SIMULATIONS + 4
        rollout_depth = _MC_ROLLOUT_DEPTH
    else:
        n_sims        = _MC_SIMULATIONS
        rollout_depth = _MC_ROLLOUT_DEPTH

    # ── Iterative deepening alpha-beta + MC ──────────────────────────────────
    best_move = random.choice(legal_moves)   # fallback sicuro

    depth = 1
    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, player,
                deadline, phase_turn,
                n_sims, rollout_depth, weights
            )
            if move is not None:
                best_move = move
            depth += 1
        except _Timeout:
            break

    # Aggiorna contatore apertura se la mossa scelta era da un angolo
    ml = _max_level(game)
    if best_move is not None:
        fr, fc = best_move[0]
        if _level(game, fr, fc) >= ml - 1:
            _opening_counter[player] = min(
                _opening_counter[player] + 1, _OPENING_MOVES
            )

    return best_move