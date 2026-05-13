import math
import random
import time

'''
Gruppo composto da:
- Francesco Pio Ruffo - mat. 277306
- Giuseppe Pio Raho - mat. 276870
- Michael Parise - mat. 276667
'''


# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Strategia con iterative deepening e alpha-beta pruning
#
# Euristica composta da quattro regole (R1-R4) con pesi configurabili.
# Ottimizzazioni implementate per raggiungere depth=4 nel tempo limite:
#
#   1. Le mosse di ogni nodo vengono generate una sola volta e riutilizzate
#      sia per l'alpha-beta sia per la valutazione euristica, eliminando
#      chiamate duplicate a _actions_for_player.
#
#   2. Il conteggio delle pedine è aggiornato incrementalmente durante
#      la ricerca (delta ±1 per cattura), senza riscandire l'intera board.
#
#   3. R4 (corner_setup) usa un'approssimazione O(1): invece di espandere
#      i figli delle mosse non catturanti, stima il bonus contando
#      direttamente le celle di alto livello adiacenti alle destinazioni
#      dei non-cattura, con lo stesso spirito della regola ma senza
#      il costo di espansione dei nodi figlio.
#
#   4. game.winner() viene chiamato solo dopo aver verificato i conteggi
#      (early-exit se nessun contatore è a zero), evitando le due chiamate
#      a player_has_moves nei casi non-terminali, che sono la maggioranza.
#
#   5. L'ordinamento delle mosse usa una chiave numerica unica invece di
#      una tupla, riducendo il costo complessivo di sorting.
#
#   6. Il TIME_MARGIN è fissato a 0.08 s: con l'iterative deepening,
#      il margine copre solo la latenza di ritorno dalla ricorsione,
#      non l'intera iterazione.
#
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.08

# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici
# ─────────────────────────────────────────────────────────────────────────────

_W_PIECES             = 80
_W_MOBILITY           = 9
_W_CAPTURE_COUNT      = 2

_W_POSITION           = 6
_W_CAPTURE_OUTER      = 1
_W_THREAT_PRESSURE    = 1
_W_CAPTURE_DANGEROUS  = 1
_W_CORNER_SETUP       = 4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]


def _positional_value(game, state, player):
    """
    Valore posizionale assoluto: somma dei quadrati dei livelli occupati
    dal giocatore, meno la somma analoga dell'avversario.
    Celle di livello più alto contribuiscono esponenzialmente di più.
    """
    opponent = game.opponent(player)
    player_val = 0
    opp_val    = 0

    dl = game.distance_levels
    for r in range(state.size):
        row   = state.board[r]
        dl_r  = dl[r]
        for c in range(state.size):
            cell = row[c]
            if cell is None:
                continue
            lv = dl_r[c]
            if cell == player:
                player_val += lv * lv
            else:
                opp_val += lv * lv

    return player_val - opp_val


def _capture_outer_bonus(game, captures):
    """R1: bonus sulle catture in posizioni esterne (livello basso)."""
    return sum(game.distance_levels[tr][tc] for (_, (tr, tc), _) in captures)


def _capture_threat_score_from_caps(game, captures):
    """R2: pressione offensiva basata sulla qualità delle catture disponibili."""
    score     = 0
    max_level = _max_level(game)
    dl        = game.distance_levels
    for move in captures:
        (fr, fc), (tr, tc), _ = move
        src_level = dl[fr][fc]
        dst_level = dl[tr][tc]
        score += 3 * src_level + (max_level - dst_level + 1)
    return score


def _capture_dangerous_piece_bonus_from_moves(game, captures, opponent_moves):
    """R3: bonus per catturare pedine avversarie minacciose o ben posizionate."""
    threatening_pieces = {
        (fr, fc)
        for (fr, fc), _, is_cap in opponent_moves
        if is_cap
    }

    bonus = 0
    dl    = game.distance_levels
    for (fr, fc), (tr, tc), _ in captures:
        target_level = dl[tr][tc]
        bonus += 2 * target_level
        if (tr, tc) in threatening_pieces:
            bonus += 12
    return bonus


def _corner_setup_bonus_approx(game, state, player, non_captures):
    """
    R4 approssimata: stima le opportunità di cattura verso celle centrali
    che si aprirebbero dopo le mosse non catturanti di livello più alto.

    Per ciascuna delle (fino a) 8 destinazioni più esterne tra le mosse
    non catturanti, conta le pedine avversarie raggiungibili in line retta
    che si trovano a livello >= max_level-1. Ogni pedina raggiungibile
    vale 1 punto di bonus.

    Questa approssimazione evita l'espansione esplicita dei nodi figlio
    mantenendo la stessa semantica di R4 a costo O(k) anziché O(k·N²).
    """
    if not non_captures:
        return 0

    max_level  = _max_level(game)
    threshold  = max_level - 1
    dl         = game.distance_levels
    board      = state.board
    size       = state.size
    opponent   = game.opponent(player)
    directions = game.DIRECTIONS

    candidates = sorted(
        non_captures,
        key=lambda m: dl[m[1][0]][m[1][1]],
        reverse=True
    )[:8]

    bonus = 0
    for move in candidates:
        _, (tr, tc), _ = move
        dst_level = dl[tr][tc]
        # Pedine avversarie di alto livello raggiungibili in queen-line
        # dalla destinazione → proxy delle catture che si aprono
        for dr, dc in directions:
            nr, nc = tr + dr, tc + dc
            while 0 <= nr < size and 0 <= nc < size and board[nr][nc] is None:
                nr += dr
                nc += dc
            if (0 <= nr < size and 0 <= nc < size
                    and board[nr][nc] == opponent
                    and dl[nr][nc] >= threshold):
                bonus += 1
    return bonus


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica
#
# Riceve le mosse già calcolate dal chiamante per evitare di ricalcolarle.
# Combina le quattro regole R1-R4 con i rispettivi pesi.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_with_moves(game, state, root_player,
                         root_moves, opp_moves,
                         root_pieces, opp_pieces):
    """
    Valuta lo stato dal punto di vista di root_player.
    Riceve mosse e conteggi già calcolati per evitare ridondanze.
    """
    root_caps    = [m for m in root_moves if m[2]]
    opp_caps     = [m for m in opp_moves  if m[2]]
    root_noncaps = [m for m in root_moves if not m[2]]

    positional = _positional_value(game, state, root_player)

    cap_outer = (
        _capture_outer_bonus(game, root_caps)
        - _capture_outer_bonus(game, opp_caps)
    )

    threat_pressure = (
        _capture_threat_score_from_caps(game, root_caps)
        - _capture_threat_score_from_caps(game, opp_caps)
    )

    capture_dangerous = (
        _capture_dangerous_piece_bonus_from_moves(game, root_caps, opp_moves)
        - _capture_dangerous_piece_bonus_from_moves(game, opp_caps, root_moves)
    )

    corner_setup = _corner_setup_bonus_approx(
        game, state, root_player, root_noncaps
    )

    score = (
        _W_PIECES            * (root_pieces - opp_pieces)
      + _W_MOBILITY          * (len(root_moves) - len(opp_moves))
      + _W_CAPTURE_COUNT     * (len(root_caps)  - len(opp_caps))
      + _W_POSITION          * positional
      + _W_CAPTURE_OUTER     * cap_outer
      + _W_THREAT_PRESSURE   * threat_pressure
      + _W_CAPTURE_DANGEROUS * capture_dangerous
      + _W_CORNER_SETUP      * corner_setup
    )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Ordinamento mosse
# ─────────────────────────────────────────────────────────────────────────────

def _order_moves(game, moves):
    """
    Ordina le mosse per massimizzare l'efficacia del pruning alpha-beta.

    Criteri in ordine di priorità:
      1. Catture prima delle non-catture.
      2. Destinazione di livello più alto (decrescente).
      3. Delta livello src→dst come criterio di spareggio (peso 0.01).

    Una chiave numerica unica sostituisce la tupla per ridurre il costo
    di confronto durante il sorting.
    """
    dl = game.distance_levels
    BIG = 1000

    def _key(move):
        (fr, fc), (tr, tc), is_capture = move
        dst = dl[tr][tc]
        if is_capture:
            return -(BIG + dst)           # catture per prime, dst decrescente
        src = dl[fr][fc]
        delta = dst - src
        return -(dst + delta * 0.01)     # non-cattura: dst primario, delta secondario

    return sorted(moves, key=_key)


# ─────────────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-beta
# ─────────────────────────────────────────────────────────────────────────────

def _alphabeta(game, state, depth, alpha, beta, maximizing,
               root_player, deadline,
               root_pieces, opp_pieces):
    """
    Ricerca alpha-beta con le seguenti ottimizzazioni:

      - Le mosse vengono generate una sola volta per nodo e riusate
        sia per l'espansione sia per la valutazione euristica alle foglie.
      - Il conteggio delle pedine è aggiornato incrementalmente (±1 per
        cattura) senza riscandire la board.
      - Il controllo di stato terminale sfrutta i contatori come early-exit:
        se nessun contatore è a zero, winner() non viene chiamato.
      - In caso di assenza di mosse legali, il turno viene passato senza
        decrementare depth, per non consumare profondità su stati forzati.
    """
    if time.perf_counter() >= deadline:
        raise _Timeout()

    opponent = game.opponent(root_player)

    # ── Controllo terminale via contatori ────────────────────────────────────
    # Evita di chiamare winner() (che riscansisce la board) nei casi normali.
    if root_pieces == 0:
        return -100_000, None
    if opp_pieces == 0:
        return 100_000, None

    # ── Generazione mosse del turno corrente ─────────────────────────────────
    cur_player  = state.to_move
    legal_moves = game._actions_for_player(state, cur_player)

    # Nessuna mossa disponibile: passa il turno
    if not legal_moves:
        passed_state = game.pass_turn(state)
        return _alphabeta(
            game, passed_state, depth - 1,
            alpha, beta, not maximizing,
            root_player, deadline,
            root_pieces, opp_pieces,
        )

    # ── Foglia: valutazione euristica ────────────────────────────────────────
    if depth == 0:
        # Le mosse dell'avversario servono alla valutazione; vengono calcolate
        # solo qui, alla foglia, per non appesantire i nodi interni.
        adv_player = game.opponent(cur_player)
        opp_moves  = game._actions_for_player(state, adv_player)

        if maximizing:
            r_moves, r_pieces, o_moves, o_pieces = (
                legal_moves, root_pieces, opp_moves, opp_pieces)
        else:
            r_moves, r_pieces, o_moves, o_pieces = (
                opp_moves, root_pieces, legal_moves, opp_pieces)

        return _evaluate_with_moves(
            game, state, root_player,
            r_moves, o_moves, r_pieces, o_pieces
        ), None

    # ── Espansione con pruning ────────────────────────────────────────────────
    ordered = _order_moves(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            (fr, fc), (tr, tc), is_cap = move
            child = game.result(state, move)

            # Aggiornamento incrementale: una cattura rimuove una pedina
            # avversaria (se cur_player == root_player) o una propria.
            new_root_pieces = root_pieces
            new_opp_pieces  = opp_pieces
            if is_cap:
                if cur_player == root_player:
                    new_opp_pieces  -= 1
                else:
                    new_root_pieces -= 1

            child_value, _ = _alphabeta(
                game, child, depth - 1,
                alpha, beta, False,
                root_player, deadline,
                new_root_pieces, new_opp_pieces,
            )
            if child_value > value:
                value      = child_value
                best_moves = [move]
            elif child_value == value:
                best_moves.append(move)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for move in ordered:
            (fr, fc), (tr, tc), is_cap = move
            child = game.result(state, move)

            new_root_pieces = root_pieces
            new_opp_pieces  = opp_pieces
            if is_cap:
                if cur_player == root_player:
                    new_opp_pieces  -= 1
                else:
                    new_root_pieces -= 1

            child_value, _ = _alphabeta(
                game, child, depth - 1,
                alpha, beta, True,
                root_player, deadline,
                new_root_pieces, new_opp_pieces,
            )
            if child_value < value:
                value      = child_value
                best_moves = [move]
            elif child_value == value:
                best_moves.append(move)
            beta = min(beta, value)
            if alpha >= beta:
                break

    return value, best_moves[0] if best_moves else None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def playerStrategy(game, state, timeout=3):
    """
    Strategia principale con iterative deepening.

    Itera dalla depth=1 aumentando di uno ad ogni ciclo finché il tempo
    a disposizione lo consente. La migliore mossa trovata nell'ultima
    iterazione completata viene restituita; in caso di timeout a depth=1,
    viene comunque restituita una mossa casuale tra quelle legali.
    """
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    deadline  = time.perf_counter() + timeout - _TIME_MARGIN
    best_move = random.choice(legal_moves)
    depth     = 1
    limite =0

    root_player = state.to_move
    opponent    = game.opponent(root_player)
    root_pieces = state.count(root_player)
    opp_pieces  = state.count(opponent)

    while True:
        if time.perf_counter() >= deadline:
            break
        try:
            value, move = _alphabeta(
                game, state, depth,
                -math.inf, math.inf,
                True, root_player, deadline,
                root_pieces, opp_pieces,
            )
            if move is not None:
                best_move = move
            limite = depth
            depth += 1
        except _Timeout:
            break
    #print(f"[PROFONDITA'] → profondità raggiunta: {limite}")
    return best_move