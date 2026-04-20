import random


def playerStrategy(game, state, timeout=3):
    """Strategia di esempio: sceglie una mossa legale a caso."""
    legal_moves = game.actions(state)
    return random.choice(legal_moves) if legal_moves else None
