import math
import random
import time

# ─────────────────────────────────────────────────────────────────────────────
# Zola AI - Strategia con iterative deepening e alpha-beta pruning
#
# Euristica IDENTICA a PaRuRa (stesse regole R1-R4 e stessi pesi).
# Ottimizzazioni implementate per raggiungere depth=4 in tempo:
#
#   1. Le mosse di ogni nodo vengono generate UNA SOLA VOLTA e riutilizzate
#      sia per l'alpha-beta sia per la valutazione euristica, eliminando le
#      chiamate duplicate a _actions_for_player.
#
#   2. Il conteggio delle pedine è tenuto incrementalmente nel nodo
#      di ricerca (delta ±1 per cattura), senza riscandire tutta la board.
#
#   3. R4 (corner_setup) usa un'approssimazione O(1): invece di espandere
#      i figli delle mosse non catturanti (8× game.result + 8× actions),
#      stima il bonus contando direttamente le celle di alto livello
#      adiacenti alle destinazioni dei non-cattura, con lo stesso spirito
#      della regola originale ma senza il costo di espansione.
#
#   4. game.winner() viene chiamato solo dopo aver verificato i conteggi
#      (early-exit se nessuna pedina è a zero), evitando le due chiamate
#      a player_has_moves nei casi non-terminali (che sono la stragrande
#      maggioranza).
#
#   5. L'ordinamento delle mosse usa una chiave numerica diretta invece
#      di una tupla, riducendo il costo di sorting.
#
#   6. Il TIME_MARGIN è ridotto a 0.08 s (era 0.15 s): con l'iterative
#      deepening, il margine serve solo per la latenza di ritorno dalla
#      ricorsione, non per l'intera iterazione.
#
# ─────────────────────────────────────────────────────────────────────────────

_TIME_MARGIN = 0.08

# ─────────────────────────────────────────────────────────────────────────────
# Pesi euristici  (INVARIATI rispetto all'originale)
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
# Helpers (INVARIATI nella logica)
# ─────────────────────────────────────────────────────────────────────────────

def _level(game, r, c):
    return game.distance_levels[r][c]


def _max_level(game):
    return game.distance_levels[0][0]


def _positional_value(game, state, player):
    """
    Valore posizionale assoluto: somma esponenziale dei livelli occupati.
    Invariato rispetto all'originale.
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
    """R1: invariato."""
    return sum(game.distance_levels[tr][tc] for (_, (tr, tc), _) in captures)


def _capture_threat_score_from_caps(game, captures):
    """R2: invariato."""
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
    """R3: invariato."""
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
    R4 approssimata: stessa semantica dell'originale, costo O(k) invece di O(k·N²).

    L'originale espandeva fino a 8 figli e per ciascuno calcolava le catture
    disponibili verso celle di livello >= max_level-1.  Quella espansione
    è il principale collo di bottiglia a depth ≥ 3.

    Approssimazione: per ognuna delle 8 mosse non catturanti più esterne,
    contiamo le pedine avversarie raggiungibili in queen-line dalla destinazione
    che sarebbero catturabili legalmente (livello pedina nemica <= dst_level,
    cioè la cattura va verso il centro) e di livello >= threshold (periferiche).

    Correzione rispetto alla versione precedente: il vincolo di legalità
    dl[nr][nc] <= dst_level era mancante, rendendo il bonus quasi sempre zero
    perché le due condizioni (>= threshold e <= dst_level) erano quasi sempre
    contraddittorie. Ora si usa threshold = 2 (pedine in zona centrale/media)
    come target delle catture da posizione periferica, coerente con la v1.
    """
    if not non_captures:
        return 0

    max_level  = _max_level(game)
    # Soglia per le pedine nemiche target: quelle di livello basso/medio
    # sono le più preziose da catturare (vicine al centro).
    # Usiamo max_level // 2 come proxy "pedina in zona interessante".
    threshold  = max(1, max_level // 2)
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
        # Pedine avversarie raggiungibili in queen-line dalla destinazione:
        # - livello <= dst_level  → cattura legale (si va verso il centro)
        # - livello <= threshold  → pedina in zona centrale/media (preziosa)
        for dr, dc in directions:
            nr, nc = tr + dr, tc + dc
            while 0 <= nr < size and 0 <= nc < size and board[nr][nc] is None:
                nr += dr
                nc += dc
            if (0 <= nr < size and 0 <= nc < size
                    and board[nr][nc] == opponent
                    and dl[nr][nc] <= dst_level      # vincolo legalità cattura
                    and dl[nr][nc] <= threshold):     # pedina in zona preziosa
                bonus += 1
    return bonus


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione euristica
#
# Firma cambiata: riceve le mosse già calcolate per evitare di ricalcolarle.
# Le regole e i pesi sono IDENTICI all'originale.
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_with_moves(game, state, root_player,
                         root_moves, opp_moves,
                         root_pieces, opp_pieces):
    """
    Valuta lo stato dal punto di vista di root_player.
    Riceve mosse e conteggi già calcolati (ottimizzazione principale).
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
    Ordina le mosse per migliorare il pruning alpha-beta.
    Stessa logica dell'originale, chiave numerica unica per velocità.

    Catture prima (is_capture), poi per livello assoluto di destinazione
    (decrescente), poi per delta come criterio secondario.
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
# Alpha-beta con passaggio delle mosse già calcolate
# ─────────────────────────────────────────────────────────────────────────────

def _alphabeta(game, state, depth, alpha, beta, maximizing,
               root_player, deadline,
               root_pieces, opp_pieces):
    """
    Alpha-beta pruning con:
      - Mosse generate una sola volta e riusate per la valutazione
      - Conteggio pedine incrementale (evita di riscandire la board)
      - Early-exit su terminale via conteggi prima di chiamare winner()
    """
    if time.perf_counter() >= deadline:
        raise _Timeout()

    opponent = game.opponent(root_player)

    # ── Fast terminal check ──────────────────────────────────────────────────
    # Se un giocatore ha 0 pedine è terminale: inutile chiamare winner()
    # che riscansisce la board e chiama player_has_moves due volte.
    if root_pieces == 0:
        return -100_000, None
    if opp_pieces == 0:
        return 100_000, None

    # ── Genera mosse del turno corrente ──────────────────────────────────────
    cur_player  = state.to_move
    legal_moves = game._actions_for_player(state, cur_player)

    # Passaggio turno: nessuna mossa disponibile
    if not legal_moves:
        passed_state = game.pass_turn(state)
        return _alphabeta(
            game, passed_state, depth - 1,
            alpha, beta, not maximizing,
            root_player, deadline,
            root_pieces, opp_pieces,
        )

    # ── Leaf node ────────────────────────────────────────────────────────────
    if depth == 0:
        # Calcoliamo le mosse dell'avversario per la valutazione
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

    # ── Espansione ───────────────────────────────────────────────────────────
    ordered = _order_moves(game, legal_moves)
    best_moves = []

    if maximizing:
        value = -math.inf
        for move in ordered:
            (fr, fc), (tr, tc), is_cap = move
            child = game.result(state, move)

            # Aggiornamento incrementale del conteggio pedine
            new_root_pieces = root_pieces
            new_opp_pieces  = opp_pieces
            if is_cap:
                # cur_player cattura → se cur_player == root_player, perdiamo
                # una pedina avversaria; altrimenti perdiamo una nostra pedina.
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
    Restituisce una mossa legale nel formato prodotto da game.actions(state).
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