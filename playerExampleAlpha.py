import math
import random


# Profondita di ricerca fissa. Puoi cambiarla senza toccare il resto del file.
SEARCH_DEPTH = 2


def evaluate_state(game, state, root_player):
    """Valuta lo stato dal punto di vista di root_player."""
    winner = game.winner(state)
    if winner == root_player:
        return 10_000
    if winner == game.opponent(root_player):
        return -10_000
    if winner is not None:
        return 0

    opponent = game.opponent(root_player)
    root_count = state.count(root_player)
    opponent_count = state.count(opponent)
    root_mobility = len(game._actions_for_player(state, root_player))
    opponent_mobility = len(game._actions_for_player(state, opponent))

    return 3 * (root_count - opponent_count) + (root_mobility - opponent_mobility)


def alphabeta(game, state, depth, alpha, beta, maximizing_player, root_player):
    legal_moves = game.actions(state)
    if depth == 0 or game.is_terminal(state) or not legal_moves:
        return evaluate_state(game, state, root_player), None

    best_moves = []

    if maximizing_player:
        value = -math.inf
        for move in legal_moves:
            child_state = game.result(state, move)
            child_value, _ = alphabeta(
                game,
                child_state,
                depth - 1,
                alpha,
                beta,
                False,
                root_player,
            )

            if child_value > value:
                value = child_value
                best_moves = [move]
            elif child_value == value:
                best_moves.append(move)

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value, random.choice(best_moves) if best_moves else None

    value = math.inf
    for move in legal_moves:
        child_state = game.result(state, move)
        child_value, _ = alphabeta(
            game,
            child_state,
            depth - 1,
            alpha,
            beta,
            True,
            root_player,
        )

        if child_value < value:
            value = child_value
            best_moves = [move]
        elif child_value == value:
            best_moves.append(move)

        beta = min(beta, value)
        if alpha >= beta:
            break

    return value, random.choice(best_moves) if best_moves else None


def playerStrategy(game, state, timeout=3):
    """Strategia di esempio: alpha-beta con profondita fissa SEARCH_DEPTH."""
    legal_moves = game.actions(state)
    if not legal_moves:
        return None

    _, best_move = alphabeta(
        game,
        state,
        SEARCH_DEPTH,
        -math.inf,
        math.inf,
        True,
        state.to_move,
    )
    return best_move
