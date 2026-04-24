import time
import random

from ZolaGameS import ZolaGame  # <-- CAMBIA con il nome del tuo file

import playerExampleNostroEU as playerA
import GiocoIA.vecchiPlayerEuristici.playerExampleNostro_v1 as playerB


NUM_GAMES = 10
TIMEOUT = 3


# ─────────────────────────────────────────────
# Barra di avanzamento per SINGOLA partita
# ─────────────────────────────────────────────
def progress_bar_moves(moves, estimated_max=200, bar_length=30):
    if moves > estimated_max:
        estimated_max = moves

    fraction = moves / estimated_max
    filled = int(fraction * bar_length)
    bar = "█" * filled + "-" * (bar_length - filled)

    return f"[{bar}] mosse: {moves}/{estimated_max}"


# ─────────────────────────────────────────────
# Singola partita
# ─────────────────────────────────────────────
def play_single_game(game_id):
    # Alterna chi inizia
    first_player = "Red" if game_id % 2 == 0 else "Blue"

    game = ZolaGame(size=8, first_player=first_player)
    state = game.initial

    # Alterna anche i player
    if game_id % 2 == 0:
        players = {"Red": playerA.playerStrategy, "Blue": playerB.playerStrategy}
        mapping = {"Red": "A", "Blue": "B"}
    else:
        players = {"Red": playerB.playerStrategy, "Blue": playerA.playerStrategy}
        mapping = {"Red": "B", "Blue": "A"}

    move_count = 0

    print(f"\n▶ Partita {game_id} iniziata | Primo: {first_player}")

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

        # Aggiorna barra ogni 5 mosse
        if move_count % 5 == 0:
            print(progress_bar_moves(move_count), end="\r")

    winner = game.winner(state)
    winner_label = mapping[winner]

    # stampa finale della barra
    print(progress_bar_moves(move_count) + " ✔")

    print(f"✔ Fine partita {game_id} → Vincitore: Player {winner_label} | Mosse: {move_count}")

    return winner_label, move_count


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
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

        print(f"⏱ Tempo partita: {game_time:.2f}s")

    total_time = time.perf_counter() - start_time

    # ─────────────────────────────────────────
    # RISULTATI FINALI
    # ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📊 RISULTATI FINALI")
    print("=" * 50)

    print(f"Partite totali   : {NUM_GAMES}")
    print(f"Vittorie Player A: {results['A']}")
    print(f"Vittorie Player B: {results['B']}")
    print(f"Mosse medie      : {total_moves / NUM_GAMES:.2f}")
    print(f"Tempo totale     : {total_time:.2f}s")

    # winrate
    winrate_A = (results["A"] / NUM_GAMES) * 100
    winrate_B = (results["B"] / NUM_GAMES) * 100

    print(f"Winrate A        : {winrate_A:.1f}%")
    print(f"Winrate B        : {winrate_B:.1f}%")

    print("=" * 50)


if __name__ == "__main__":
    main()