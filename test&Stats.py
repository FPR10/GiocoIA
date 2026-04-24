import time
import random

from ZolaGameS import ZolaGame  # <-- cambia con il nome reale del tuo file

import playerExampleNostroEU as playerA
import playerExampleNostro_v1 as playerB


NUM_GAMES = 10
TIMEOUT = 3


def play_single_game(game_id):
    # Alterna chi inizia
    first_player = "Red" if game_id % 2 == 0 else "Blue"
    game = ZolaGame(size=8, first_player=first_player)
    state = game.initial

    # Alterna anche quale AI gioca Red/Blue
    if game_id % 2 == 0:
        players = {
            "Red": playerA.playerStrategy,
            "Blue": playerB.playerStrategy
        }
        mapping = {"Red": "A", "Blue": "B"}
    else:
        players = {
            "Red": playerB.playerStrategy,
            "Blue": playerA.playerStrategy
        }
        mapping = {"Red": "B", "Blue": "A"}

    move_count = 0

    while not game.is_terminal(state):
        current = state.to_move
        legal_moves = game.actions(state)

        if not legal_moves:
            state = game.pass_turn(state)
            continue

        strategy = players[current]

        try:
            move = strategy(game, state, TIMEOUT)
        except Exception:
            move = None

        if move not in legal_moves:
            move = random.choice(legal_moves)

        state = game.result(state, move)
        move_count += 1

    winner = game.winner(state)

    # Traduci vincitore in A/B (non Red/Blue)
    winner_label = mapping[winner]

    return winner_label, move_count


def progress_bar(current, total, bar_length=30):
    fraction = current / total
    filled = int(fraction * bar_length)
    bar = "█" * filled + "-" * (bar_length - filled)
    return f"[{bar}] {current}/{total}"


def main():
    results = {"A": 0, "B": 0}
    total_moves = 0
    start_time = time.perf_counter()

    for i in range(1, NUM_GAMES + 1):
        game_start = time.perf_counter()

        winner, moves = play_single_game(i)

        results[winner] += 1
        total_moves += moves

        game_time = time.perf_counter() - game_start

        print(
            f"{progress_bar(i, NUM_GAMES)} | "
            f"Partita {i} → Vincitore: Player {winner} | "
            f"Mosse: {moves} | "
            f"Tempo: {game_time:.2f}s"
        )

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 40)
    print("RISULTATI FINALI")
    print("=" * 40)
    print(f"Partite totali: {NUM_GAMES}")
    print(f"Vittorie Player A: {results['A']}")
    print(f"Vittorie Player B: {results['B']}")
    print(f"Mosse medie      : {total_moves / NUM_GAMES:.2f}")
    print(f"Tempo totale     : {total_time:.2f}s")
    print("=" * 40)


if __name__ == "__main__":
    main()