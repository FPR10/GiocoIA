import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Iterative-deepening alpha-beta e timeout
#
# Struttura:
#   • evaluate_state   – funzione euristica multi-feature
#   • order_moves      – ordinamento delle mosse (catture esterne prima)
#   • alphabeta        – alpha-beta con move ordering
#   • playerStrategy   – iterative deepening con guardia al tempo
#
# Principi strategici incorporati:
#   P1. Preferire catture verso l'esterno (livello destinazione alto)
#   P2. Nelle mosse non-catturanti preferire destinazioni più esterne
#   P3. Pressione nelle zone esterne dove ci sono pedine avversarie esterne
#   P4. Catturare la pedina avversaria più interna (livello basso) = più pericolosa
#   P5. Setup angolo: bonus se una mossa non-catturante apre una cattura al turno dopo
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.15   # secondi di margine di sicurezza prima del timeout


# ─────────────────────────────────────────────────────────────────────────────
# Costanti di peso dell'euristica  (modifica qui per bilanciare)
# ─────────────────────────────────────────────────────────────────────────────
_W_PIECES             = 100   # differenza pedine residue
_W_MOBILITY           = 3     # differenza mosse legali disponibili
_W_CAPTURE_COUNT      = 15    # differenza numero catture disponibili
_W_CAPTURE_OUTER      = 4     # bonus catture verso livelli esterni
_W_CAPTURE_DANGEROUS  = 2     # bonus cattura pedine pericolose
_W_MOVE_OUTER         = 2     # bonus mosse non catturanti verso esterno
_W_THREAT_PRESSURE    = 2     # pressione tattica tramite catture reali
_W_FUTURE_SETUP       = 1     # setup tattico futuro


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    """Livello massimo presente nella scacchiera (angoli)."""
    s = game.size - 1
    return game.distance_levels[0][0]   # angolo = massima distanza


def _capture_outer_bonus(game, captures):
    """P1 – somma dei livelli di destinazione delle catture disponibili.

    Catture verso celle più esterne (livello alto) sono premiate.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in captures)





def _move_outer_bonus(game, non_captures):
    """P2 – somma dei livelli di destinazione delle mosse non-catturanti.

    Mosse che portano più all'esterno (livello alto) sono premiate.
    """
    return sum(_level(game, tr, tc) for (_, (tr, tc), _) in non_captures)





def _capture_threat_score(game, state, player):
    """
    NUOVA P3:Valuta la pressione tattica reale di player:
    conta le catture effettivamente disponibili secondo le regole di Zola.

    Più è esterna la pedina che cattura, più la minaccia è forte.
    Più è interna la pedina catturata, più può essere utile eliminarla.

    Questa nuova euristica è più costosa della vecchia, perché scansiona linee in 8 direzioni.
    """
    moves = game._actions_for_player(state, player)
    score = 0
    max_level = _max_level(game)

    for move in moves:
        (fr, fc), (tr, tc), is_capture = move
        if not is_capture:
            continue

        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)

        attacker_bonus = src_level
        target_inner_bonus = max_level - dst_level + 1

        score += 3 * attacker_bonus + target_inner_bonus

    return score




def _capture_dangerous_piece_bonus(game, state, captures, opponent):
    """
    P4 nuova veloce: premia le catture di pedine avversarie pericolose.

    Una pedina è pericolosa se:
      - è esterna;
      - ha almeno una cattura disponibile.
    """
    opponent_moves = game._actions_for_player(state, opponent)
    threatening_pieces = set()

    for move in opponent_moves:
        (fr, fc), (tr, tc), is_capture = move
        if is_capture:
            threatening_pieces.add((fr, fc))

    bonus = 0

    for move in captures:
        (fr, fc), (tr, tc), is_capture = move
        if not is_capture:
            continue

        target_level = _level(game, tr, tc)

        # Più la pedina catturata è esterna, più vale eliminarla.
        bonus += 2 * target_level

        # Se quella pedina aveva almeno una cattura disponibile,
        # eliminarla è molto importante.
        if (tr, tc) in threatening_pieces:
            bonus += 12

    return bonus

def _future_threat_setup_bonus(game, state, player, non_captures):
    """
    P5 nuova: setup tattico veloce.

    Premia le mosse non catturanti che aumentano la pressione di cattura
    del giocatore dopo la mossa.

    Per non rallentare troppo alpha-beta, valuta solo le 8 mosse
    non catturanti più promettenti, cioè quelle che aumentano di più
    il livello di distanza dal centro.
    """
    if state.to_move != player:
        return 0

    before_score = _capture_threat_score(game, state, player)

    candidate_moves = sorted(
        non_captures,
        key=lambda m: _level(game, m[1][0], m[1][1]) - _level(game, m[0][0], m[0][1]),
        reverse=True
    )[:8]

    bonus = 0

    for move in candidate_moves:
        child = game.result(state, move)
        after_score = _capture_threat_score(game, child, player)

        improvement = after_score - before_score

        if _moved_piece_is_capturable(game, child, move, player):
            improvement -= 20

        if improvement > 0:
            bonus += improvement

    return bonus

def _moved_piece_is_capturable(game, state_after_move, move, player):
    """
    Controlla se la pedina appena mossa può essere catturata subito dall'avversario.
    """
    (_, _), (tr, tc), _ = move
    opponent = game.opponent(player)

    opp_moves = game._actions_for_player(state_after_move, opponent)

    for opp_move in opp_moves:
        (ofr, ofc), (otr, otc), is_cap = opp_move
        if is_cap and (otr, otc) == (tr, tc):
            return True

    return False



# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica principale
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_state(game, state, root_player):
    """Valutazione euristica dal punto di vista di root_player.

    Feature combinate:
      • differenza pedine residue                          (P base)
      • differenza mobilità totale                         (P base)
      • differenza catture disponibili                     (P base)
      • catture esterne proprie vs avversarie              (P1)
      • catture di pedine interne avversarie               (P4)
      • mosse non-catturanti verso esterno                 (P2)
      • pressione esterna (nostre pedine vicino avversari) (P3)
      • setup angolo: mosse che aprono catture in periferia(P5)
    """
    winner = game.winner(state)
    if winner == root_player:
        return 100_000
    if winner is not None:
        return -100_000

    opponent = game.opponent(root_player)

    # ── conteggi base ────────────────────────────────────────────────────────
    root_pieces = state.count(root_player)
    opp_pieces  = state.count(opponent)

    root_moves    = game._actions_for_player(state, root_player)
    opp_moves     = game._actions_for_player(state, opponent)

    root_caps     = [m for m in root_moves if m[2]]
    opp_caps      = [m for m in opp_moves  if m[2]]
    root_noncaps  = [m for m in root_moves if not m[2]]
    opp_noncaps   = [m for m in opp_moves  if not m[2]]

    # ── feature strategiche ──────────────────────────────────────────────────
    # P1: catture verso esterno, non deve pesare troppo
    #A volte conviene catturare una pedina più centrale perché quella pedina può diventare molto mobile o aprire nuove linee.
    cap_outer = _capture_outer_bonus(game, root_caps) - _capture_outer_bonus(game, opp_caps)

    # P4: catturare pedine più interne (più pericolose)
    capture_dangerous = (
    _capture_dangerous_piece_bonus(game, state, root_caps, opponent)
    - _capture_dangerous_piece_bonus(game, state, opp_caps, root_player)
)

    # P2: mosse non-catturanti verso esterno
    #le mosse non catturanti in Zola servono spesso a portare le proprie pedine in zone più esterne, da cui poi possono controllare linee di cattura verso il centro.
    #ATTENZIONE AL PESO DI QUESTO, POTREBBE CAUSARE MOSSE INUTILI TATTICAMENTE, PESO _W_MOVE_OUTER = 2 potrebbe essere più adatto
    move_outer = (
        _move_outer_bonus(game, root_noncaps)
        - _move_outer_bonus(game, opp_noncaps)
    )

   
    #P3
    threat_pressure = (
    _capture_threat_score(game, state, root_player)
    - _capture_threat_score(game, state, opponent)
)

    # P5: setup angolo (solo per noi, costoso da calcolare per entrambi)
    future_setup = _future_threat_setup_bonus(
    game,
    state,
    root_player,
    root_noncaps
)

    score = (
        _W_PIECES         * (root_pieces - opp_pieces)
      + _W_MOBILITY       * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT  * (len(root_caps)  - len(opp_caps))
      + _W_CAPTURE_OUTER  * cap_outer
      + _W_CAPTURE_DANGEROUS * capture_dangerous
      + _W_MOVE_OUTER     * move_outer
      + _W_THREAT_PRESSURE * threat_pressure
      #_W_OUTER_PRESSURE * outer_pressure
      + _W_FUTURE_SETUP * future_setup
    )
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento delle mosse (move ordering)
# ─────────────────────────────────────────────────────────────────────────────

#ALPHA BETA CON QUESTA FUNZIONE RIESCE AD ESPLORARE PRIMA I RAMI PIU PROMETTENTI

def order_moves(game, moves):
    """
    Ordina le mosse per migliorare il pruning alpha-beta.

    Prima considera:
      1. catture;
      2. catture fatte da pedine esterne;
      3. catture verso destinazioni esterne;
      4. mosse non catturanti che aumentano di più il livello.
    """
    def move_priority(m):
        (fr, fc), (tr, tc), is_cap = m
        src_level = _level(game, fr, fc)
        dst_level = _level(game, tr, tc)

        if is_cap:
            return (0, -src_level, -dst_level)

        delta = dst_level - src_level
        return (1, -delta, -dst_level)

    return sorted(moves, key=move_priority)

def _quiescence(game, state, alpha, beta, maximizing, root_player, deadline, q_depth=2):
    """
    Ricerca tattica sulle sole catture.

    Serve per evitare di valutare come buona una posizione in cui
    l'avversario ha una cattura immediata forte.

    q_depth limita la profondità extra, così non rischiamo di sforare il timeout.
    """
    if time.perf_counter() >= deadline:
        raise _Timeout()

    stand_pat = evaluate_state(game, state, root_player)

    if q_depth == 0 or game.is_terminal(state):
        return stand_pat

    legal_moves = game.actions(state)

    if not legal_moves:
        passed_state = game.pass_turn(state)
        return _quiescence(
            game,
            passed_state,
            alpha,
            beta,
            not maximizing,
            root_player,
            deadline,
            q_depth - 1,
        )

    capture_moves = [m for m in legal_moves if m[2]]

    # Se non ci sono catture, la posizione è abbastanza "quieta":
    # possiamo usare l'euristica normale.
    if not capture_moves:
        return stand_pat

    ordered_captures = order_moves(game, capture_moves)

    if maximizing:
        value = stand_pat

        for move in ordered_captures:
            child = game.result(state, move)
            child_value = _quiescence(
                game,
                child,
                alpha,
                beta,
                False,
                root_player,
                deadline,
                q_depth - 1,
            )

            value = max(value, child_value)
            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    else:
        value = stand_pat

        for move in ordered_captures:
            child = game.result(state, move)
            child_value = _quiescence(
                game,
                child,
                alpha,
                beta,
                True,
                root_player,
                deadline,
                q_depth - 1,
            )

            value = min(value, child_value)
            beta = min(beta, value)

            if alpha >= beta:
                break

        return value

# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta con time check
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    """Eccezione interna lanciata quando il tempo è scaduto."""

 #MICHAEL HA MODELLATO LA NECESSITA' DI PASSARE IL TURNO QUANDO UN GIOCATORE NON HA MOSSE
def _alphabeta(game, state, depth, alpha, beta, maximizing, root_player, deadline):
    if time.perf_counter() >= deadline:
        raise _Timeout()

    #La partita è realmente finita?
    if game.is_terminal(state):
        return evaluate_state(game, state, root_player), None
    '''sono a profondità 0
    ci sono catture?
    sì → continuo solo sulle catture
    no → valuto'''

    if depth == 0:
        return _quiescence(
            game,
            state,
            alpha,
            beta,
            maximizing,
            root_player,
            deadline,
            q_depth=2,
        ), None

    #Se la partita non è finita,cioè il giocatore ha ancora pedine, allora controlli le mosse e se non ci sono mosse passi il turno
    legal_moves = game.actions(state)

    # Caso importante: il giocatore corrente non ha mosse,
    # quindi deve saltare il turno, non valutare lo stato come foglia.
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

    ordered = order_moves(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            child = game.result(state, move)
            child_val, _ = _alphabeta(
                game, child, depth - 1,
                alpha, beta,
                False,
                root_player,
                deadline,
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
                game, child, depth - 1,
                alpha, beta,
                True,
                root_player,
                deadline,
            )

            if child_val < value:
                value = child_val
                best_moves = [move]
            elif child_val == value:
                best_moves.append(move)

            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, random.choice(best_moves) if best_moves else None

# ─────────────────────────────────────────────────────────────────────────────
# Entry-point: iterative deepening con guardia al tempo
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """Strategia con iterative-deepening alpha-beta e timeout sicuro.

    Esegue ricerche a profondità crescente (1, 2, 3, …).  Quando sta per
    scadere il tempo concesso (timeout - _TIME_MARGIN), restituisce la
    migliore mossa trovata fino a quel momento.  Se nessuna ricerca completa
    è mai terminata, ricade su una mossa casuale (non dovrebbe accadere con
    timeout ≥ 1 s, ma è la rete di sicurezza richiesta).
    """
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    deadline = time.perf_counter() + timeout - _TIME_MARGIN

    best_move  = random.choice(legal_moves)   # fallback sicuro
    best_value = -math.inf

    depth = 1
    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, state.to_move,
                deadline,
            )
            if move is not None:
                best_move  = move
                best_value = value
            depth += 1
        except _Timeout:
            # Tempo scaduto durante questa iterazione: usa il risultato precedente
            break

    return best_move