import tkinter as tk
from tkinter import simpledialog, messagebox
import random
import concurrent.futures
import threading
import time

# EXAMPLE VERSION
# #######################
import playerExampleRandom as playerBmodule
import playerExampleAlpha as playerRmodule
# #######################

class Game:
    """Classe astratta per giochi a due giocatori."""

    def actions(self, state):
        """Restituisce le mosse legali disponibili nello stato corrente."""
        raise NotImplementedError

    def result(self, state, move):
        """Restituisce il nuovo stato ottenuto applicando la mossa."""
        raise NotImplementedError

    def is_terminal(self, state):
        """Restituisce True se lo stato e terminale."""
        return not self.actions(state)

    def utility(self, state, player):
        """Valore terminale dello stato dal punto di vista di player."""
        raise NotImplementedError


class Board:
    """Rappresentazione dello stato della scacchiera di Zola.

    Ogni cella contiene:
      - "Red"
      - "Blue"
      - None
    """

    def __init__(self, size, board=None, to_move="Red", last_move=None):
        self.size = size
        if board is None:
            self.board = [[None for _ in range(size)] for _ in range(size)]
        else:
            self.board = board
        self.to_move = to_move
        self.last_move = last_move

    def copy(self):
        new_board = [row[:] for row in self.board]
        return Board(self.size, new_board, self.to_move, self.last_move)

    def count(self, player):
        total = 0
        for row in self.board:
            for cell in row:
                if cell == player:
                    total += 1
        return total


def compute_distance_levels(size):
    """Restituisce la matrice dei livelli di distanza dal centro.

    La distanza viene misurata rispetto al punto centrale geometrico della
    scacchiera. Celle con uguale distanza euclidea appartengono allo stesso
    livello. Il livello 1 e quello piu vicino al centro.

    Per evitare problemi numerici si confrontano le distanze quadratiche
    scalate:
        ((2r - (size-1))^2 + (2c - (size-1))^2)

    Su una scacchiera 8x8 il risultato e:
        9 8 7 6 6 7 8 9
        8 6 5 4 4 5 6 8
        7 5 3 2 2 3 5 7
        6 4 2 1 1 2 4 6
        6 4 2 1 1 2 4 6
        7 5 3 2 2 3 5 7
        8 6 5 4 4 5 6 8
        9 8 7 6 6 7 8 9
    """
    if size <= 0 or size % 2 != 0:
        raise ValueError("La scacchiera deve avere dimensione pari positiva.")

    scaled_distances = {}
    unique_values = set()

    for r in range(size):
        for c in range(size):
            value = (2 * r - (size - 1)) ** 2 + (2 * c - (size - 1)) ** 2
            scaled_distances[(r, c)] = value
            unique_values.add(value)

    ordered_values = sorted(unique_values)
    level_of = {value: index + 1 for index, value in enumerate(ordered_values)}

    return [
        [level_of[scaled_distances[(r, c)]] for c in range(size)]
        for r in range(size)
    ]


class ZolaGame(Game):
    """Implementazione del gioco Zola su scacchiera 8x8.

    Regole principali:
      - La scacchiera parte piena, con disposizione a scacchiera di pedine Red/Blue.
      - Mossa non catturante: re-like verso una cella adiacente vuota, ma solo se
        la destinazione aumenta la distanza dal centro.
      - Mossa catturante: queen-like lungo una direzione retta fino a una pedina
        avversaria raggiungibile attraverso zero o piu celle vuote. La destinazione
        deve mantenere o diminuire la distanza dal centro.
      - Se il giocatore di turno non ha mosse, salta automaticamente il turno.
      - Vince chi cattura tutte le pedine avversarie.
    """

    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, size=8, first_player="Red"):
        if size % 2 != 0:
            raise ValueError("Zola richiede una scacchiera di dimensione pari.")
        if first_player not in {"Red", "Blue"}:
            raise ValueError("Il primo giocatore deve essere 'Red' oppure 'Blue'.")

        self.size = size
        self.first_player = first_player
        self.distance_levels = compute_distance_levels(size)
        self.initial = Board(size, self._initial_board(), to_move=first_player)

    def _initial_board(self):
        board = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                row.append("Blue" if (r + c) % 2 == 0 else "Red")
            board.append(row)
        return board

    @staticmethod
    def opponent(player):
        return "Blue" if player == "Red" else "Red"

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def get_distance_level(self, r, c):
        return self.distance_levels[r][c]

    def get_all_distance_levels(self):
        return [row[:] for row in self.distance_levels]

    def actions(self, state):
        return self._actions_for_player(state, state.to_move)

    def _actions_for_player(self, state, player):
        enemy = self.opponent(player)
        moves = []

        for r in range(state.size):
            for c in range(state.size):
                if state.board[r][c] != player:
                    continue

                current_level = self.get_distance_level(r, c)

                # Mosse non catturanti: adiacenza in 8 direzioni, verso un livello maggiore.
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not self.in_bounds(nr, nc):
                        continue
                    if state.board[nr][nc] is not None:
                        continue
                    if self.get_distance_level(nr, nc) > current_level:
                        moves.append(((r, c), (nr, nc), False))

                # Mosse catturanti: movimento queen-like fino al primo pezzo incontrato.
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    while self.in_bounds(nr, nc) and state.board[nr][nc] is None:
                        nr += dr
                        nc += dc

                    if not self.in_bounds(nr, nc):
                        continue
                    if state.board[nr][nc] != enemy:
                        continue
                    if self.get_distance_level(nr, nc) <= current_level:
                        moves.append(((r, c), (nr, nc), True))

        return moves

    def player_has_moves(self, state, player):
        return bool(self._actions_for_player(state, player))

    def pass_turn(self, state):
        if self.actions(state):
            raise ValueError("Il giocatore corrente ha mosse legali: non puo passare.")

        new_state = state.copy()
        skipped_player = state.to_move
        new_state.to_move = self.opponent(skipped_player)
        new_state.last_move = {
            "player": skipped_player,
            "type": "pass",
        }
        return new_state

    def result(self, state, move):
        if move == "PASS":
            return self.pass_turn(state)

        new_state = state.copy()
        (fr, fc), (tr, tc), is_capture = move
        current_player = state.to_move
        enemy = self.opponent(current_player)

        if state.board[fr][fc] != current_player:
            raise ValueError("La cella di partenza non contiene una pedina del giocatore di turno.")

        if is_capture:
            if state.board[tr][tc] != enemy:
                raise ValueError("La destinazione di una cattura deve contenere una pedina avversaria.")
        else:
            if state.board[tr][tc] is not None:
                raise ValueError("La destinazione di una mossa non catturante deve essere vuota.")

        new_state.board[fr][fc] = None
        new_state.board[tr][tc] = current_player
        new_state.to_move = self.opponent(current_player)
        new_state.last_move = {
            "player": current_player,
            "type": "capture" if is_capture else "move",
            "from": (fr, fc),
            "to": (tr, tc),
            "captured": (tr, tc) if is_capture else None,
        }
        return new_state

    def winner(self, state):
        red_count = state.count("Red")
        blue_count = state.count("Blue")

        if blue_count == 0:
            return "Red"
        if red_count == 0:
            return "Blue"

        red_has_moves = self.player_has_moves(state, "Red")
        blue_has_moves = self.player_has_moves(state, "Blue")

        if not red_has_moves and not blue_has_moves:
            # Situazione teoricamente non prevista dalle regole; inseriamo un fallback difensivo.
            if red_count > blue_count:
                return "Red"
            if blue_count > red_count:
                return "Blue"
            return self.opponent(state.to_move)

        return None

    def is_terminal(self, state):
        return self.winner(state) is not None

    def utility(self, state, player="Red"):
        winner = self.winner(state)
        if winner is None:
            raise ValueError("La utility e definita soltanto per stati terminali.")
        return 1 if winner == player else -1


def random_player(game, state, timeout=3):
    moves = game.actions(state)
    return random.choice(moves) if moves else None


class ZolaGUI:
    LIGHT_SQUARE = "#EEEED2"
    DARK_SQUARE = "#B58863"
    SOURCE_HIGHLIGHT = "#FFF59D"
    DEST_HIGHLIGHT = "#A5D6A7"
    LAST_FROM_HIGHLIGHT = "#FFE082"
    LAST_TO_HIGHLIGHT = "#FFCC80"

    def __init__(self, game, player_types, time_out=3, player_names=None):
        self.game = game
        self.player_types = player_types
        self.time_out = time_out
        self.player_names = player_names or {"Red": "Rosso", "Blue": "Blu"}

        self.state_history = [game.initial]
        self.current_index = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.game_start_time = time.perf_counter()
        self.thinking_time = {"Red": 0.0, "Blue": 0.0}

        self.waiting_for_human = False
        self.human_move = None
        self.legal_moves_cache = []
        self.selectable_sources = set()
        self.selected_source = None
        self.available_moves_from_source = []

        self.auto_mode = False
        self.show_auto = (
            self.player_types.get("Blue") == "ai"
            and self.player_types.get("Red") == "ai"
        )

        self.root = tk.Tk()
        self.root.title("Zola")
        self.root.geometry("640x760")
        self.root.configure(bg="white")

        self.title_label = tk.Label(
            self.root,
            text="Zola - scacchiera 8x8",
            bg="white",
            font=("Helvetica", 16, "bold"),
        )
        self.title_label.pack(pady=(12, 6))

        self.board_frame = tk.Frame(self.root, bg="white")
        self.board_frame.pack(pady=10)

        self.info_frame = tk.Frame(self.root, bg="white")
        self.info_frame.pack(pady=5)

        self.players_frame = tk.Frame(self.info_frame, bg="white")
        self.players_frame.pack(pady=(0, 6))

        red_role = "AI" if self.player_types.get("Red") == "ai" else "Umano"
        blue_role = "AI" if self.player_types.get("Blue") == "ai" else "Umano"

        self.red_name_label = tk.Label(
            self.players_frame,
            text=f"Rosso: {self.player_names['Red']} ({red_role})",
            bg="white",
            fg="firebrick",
            font=("Helvetica", 12, "bold"),
        )
        self.red_name_label.grid(row=0, column=0, padx=16)

        self.blue_name_label = tk.Label(
            self.players_frame,
            text=f"Blu: {self.player_names['Blue']} ({blue_role})",
            bg="white",
            fg="royalblue",
            font=("Helvetica", 12, "bold"),
        )
        self.blue_name_label.grid(row=0, column=1, padx=16)

        self.controls_frame = tk.Frame(self.root, bg="white")
        self.controls_frame.pack(pady=10)

        self.cells = [[None for _ in range(self.game.size)] for _ in range(self.game.size)]
        for r in range(self.game.size):
            for c in range(self.game.size):
                lbl = tk.Label(
                    self.board_frame,
                    text="",
                    width=3,
                    height=1,
                    borderwidth=2,
                    relief="ridge",
                    font=("Helvetica", 26, "bold"),
                    bg=self._square_color(r, c),
                )
                lbl.grid(row=r, column=c, padx=2, pady=2, ipadx=8, ipady=6)
                lbl.bind("<Button-1>", lambda e, row=r, col=c: self.cell_clicked(row, col))
                self.cells[r][c] = lbl

        self.prev_button = tk.Button(
            self.controls_frame,
            text="Precedente",
            command=self.prev_move,
            font=("Helvetica", 12),
            padx=10,
            pady=5,
        )
        self.prev_button.grid(row=0, column=0, padx=5)

        self.next_button = tk.Button(
            self.controls_frame,
            text="Successivo",
            command=lambda: self.next_move(),
            font=("Helvetica", 12),
            padx=10,
            pady=5,
        )
        self.next_button.grid(row=0, column=1, padx=5)

        if self.show_auto:
            self.auto_button = tk.Button(
                self.controls_frame,
                text="Auto",
                highlightbackground="red",
                command=self.toggle_auto,
                font=("Helvetica", 12),
                padx=10,
                pady=5,
            )
            self.auto_button.grid(row=0, column=2, padx=5)

        self.score_label = tk.Label(
            self.info_frame,
            text="",
            bg="white",
            font=("Helvetica", 12, "bold"),
        )
        self.score_label.pack()

        self.time_label = tk.Label(
            self.info_frame,
            text="",
            bg="white",
            font=("Helvetica", 11),
        )
        self.time_label.pack(pady=(4, 0))

        self.status_label = tk.Label(
            self.info_frame,
            text="",
            bg="white",
            font=("Helvetica", 12),
        )
        self.status_label.pack(pady=(4, 0))

        self.update_board()
        self._schedule_timer_update()

    def _square_color(self, r, c):
        return self.LIGHT_SQUARE if (r + c) % 2 == 0 else self.DARK_SQUARE

    def current_state(self):
        return self.state_history[self.current_index]

    def latest_state(self):
        return self.state_history[-1]

    @staticmethod
    def format_seconds(seconds):
        total_seconds = max(0, int(seconds))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def player_display_name(self, player):
        return self.player_names.get(player, player)

    def player_summary(self, player):
        role = "AI" if self.player_types.get(player) == "ai" else "Umano"
        return f"{self.player_display_name(player)} ({role})"

    def total_game_time(self):
        return time.perf_counter() - self.game_start_time

    def update_time_labels(self):
        total_time = self.format_seconds(self.total_game_time())
        red_time = self.format_seconds(self.thinking_time["Red"])
        blue_time = self.format_seconds(self.thinking_time["Blue"])
        self.time_label.config(
            text=(
                f"Tempo partita: {total_time}    "
                f"{self.player_display_name('Red')}: {red_time}    "
                f"{self.player_display_name('Blue')}: {blue_time}"
            )
        )

    def _schedule_timer_update(self):
        if not self.root.winfo_exists():
            return
        self.update_time_labels()
        self.root.after(200, self._schedule_timer_update)

    def update_board(self):
        state = self.current_state()

        for r in range(state.size):
            for c in range(state.size):
                cell = state.board[r][c]
                lbl = self.cells[r][c]
                base_bg = self._square_color(r, c)

                if cell is None:
                    lbl.config(text="", fg="black", bg=base_bg)
                else:
                    fg = "firebrick" if cell == "Red" else "royalblue"
                    lbl.config(text="●", fg=fg, bg=base_bg)

                lbl.config(relief="ridge", borderwidth=2)

        if state.last_move and state.last_move.get("type") != "pass":
            fr, fc = state.last_move["from"]
            tr, tc = state.last_move["to"]
            self.cells[fr][fc].config(bg=self.LAST_FROM_HIGHLIGHT, relief="solid", borderwidth=4)
            self.cells[tr][tc].config(bg=self.LAST_TO_HIGHLIGHT, relief="solid", borderwidth=4)

        if self.waiting_for_human and self.current_index == len(self.state_history) - 1:
            if self.selected_source is None:
                for r, c in self.selectable_sources:
                    self.cells[r][c].config(relief="solid", borderwidth=4)
            else:
                sr, sc = self.selected_source
                self.cells[sr][sc].config(bg=self.SOURCE_HIGHLIGHT, relief="solid", borderwidth=4)
                for move in self.available_moves_from_source:
                    tr, tc = move[1]
                    self.cells[tr][tc].config(bg=self.DEST_HIGHLIGHT, relief="solid", borderwidth=4)

        red_count = state.count("Red")
        blue_count = state.count("Blue")
        self.score_label.config(
            text=(
                f"{self.player_display_name('Red')} (Rosso): {red_count}    "
                f"{self.player_display_name('Blue')} (Blu): {blue_count}"
            )
        )
        self.update_time_labels()

        if self.game.is_terminal(state):
            winner = self.game.winner(state)
            winner_name = self.player_display_name(winner) if winner else "Pareggio"
            self.status_label.config(text=f"Vincitore: {winner_name}")
            return

        extra = ""
        if state.last_move and state.last_move.get("type") == "pass":
            extra = f" - {self.player_display_name(state.last_move['player'])} salta il turno"

        if self.waiting_for_human and self.current_index == len(self.state_history) - 1:
            if self.selected_source is None:
                prompt = " - seleziona una pedina"
            else:
                prompt = " - seleziona la destinazione"
        else:
            prompt = ""

        self.status_label.config(
            text=f"Turno: {self.player_display_name(state.to_move)}{prompt}{extra}"
        )

    def cell_clicked(self, r, c):
        if not self.waiting_for_human:
            return
        if self.current_index != len(self.state_history) - 1:
            return

        clicked = (r, c)

        if self.selected_source is None:
            if clicked in self.selectable_sources:
                self.selected_source = clicked
                self.available_moves_from_source = [
                    move for move in self.legal_moves_cache if move[0] == clicked
                ]
                self.update_board()
            return

        if clicked == self.selected_source:
            self.selected_source = None
            self.available_moves_from_source = []
            self.update_board()
            return

        for move in self.available_moves_from_source:
            if move[1] == clicked:
                self.human_move = move
                self.waiting_for_human = False
                self.selected_source = None
                self.available_moves_from_source = []
                self.selectable_sources = set()
                self.legal_moves_cache = []
                self.update_board()
                return

        if clicked in self.selectable_sources:
            self.selected_source = clicked
            self.available_moves_from_source = [
                move for move in self.legal_moves_cache if move[0] == clicked
            ]
            self.update_board()

    def prev_move(self):
        if self.waiting_for_human:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.update_board()

    def next_move(self):
        if self.waiting_for_human:
            return

        if self.current_index < len(self.state_history) - 1:
            self.current_index += 1
            self.update_board()
            return

        ai_vs_ai_manual = (
            self.player_types.get("Blue") == "ai"
            and self.player_types.get("Red") == "ai"
            and not self.auto_mode
        )
        if ai_vs_ai_manual and not self.game.is_terminal(self.latest_state()):
            self.play_turn()
            self.update_board()

    def toggle_auto(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_button.config(text="Auto ON", highlightbackground="green")
            threading.Thread(target=self.auto_play, daemon=True).start()
        else:
            self.auto_button.config(text="Auto", highlightbackground="red")

    def auto_play(self):
        while self.auto_mode and not self.game.is_terminal(self.latest_state()):
            self.play_turn()
            time.sleep(0.4)

        if self.game.is_terminal(self.latest_state()):
            self.show_game_over("La partita e terminata.")

    def play_turn(self):
        state = self.latest_state()
        if self.game.is_terminal(state):
            return

        current_player = state.to_move
        legal_moves = self.game.actions(state)

        if not legal_moves:
            skipped_state = self.game.pass_turn(state)
            self.state_history.append(skipped_state)
            self.current_index = len(self.state_history) - 1
            self.update_board()
            return

        move = None
        turn_start = time.perf_counter()

        if self.player_types[current_player] == "ai":
            if current_player == "Blue":
                strategy = playerBmodule.playerStrategy
            else:
                strategy = playerRmodule.playerStrategy
            future = self.executor.submit(strategy, self.game, state, self.time_out)
            try:
                move = future.result(timeout=self.time_out)
                print(f"AI {current_player} ha scelto la mossa {move}")
            except concurrent.futures.TimeoutError:
                future.cancel()
                move = None
            except Exception as exc:
                print(f"Errore nella strategia di {current_player}: {exc}")
                move = None

            if move not in legal_moves:
                move = random.choice(legal_moves)
                print(f"Time-out o mossa non valida per {current_player}, scelta casuale {move}")
        else:
            self.waiting_for_human = True
            self.human_move = None
            self.legal_moves_cache = legal_moves
            self.selectable_sources = {move[0] for move in legal_moves}
            self.selected_source = None
            self.available_moves_from_source = []
            self.update_board()

            while self.waiting_for_human:
                self.root.update()
                time.sleep(0.05)

            move = self.human_move
            self.update_board()

        self.thinking_time[current_player] += time.perf_counter() - turn_start

        new_state = self.game.result(state, move)
        self.state_history.append(new_state)
        self.current_index = len(self.state_history) - 1
        self.update_board()

    def run_game_loop(self):
        ai_vs_ai_manual = (
            self.player_types.get("Blue") == "ai"
            and self.player_types.get("Red") == "ai"
            and not self.auto_mode
        )

        if not ai_vs_ai_manual:
            def loop():
                while not self.game.is_terminal(self.latest_state()):
                    self.play_turn()
                    time.sleep(0.1)
            threading.Thread(target=loop, daemon=True).start()

        self.root.mainloop()

    def show_game_over(self, message):
        dialog = tk.Toplevel(self.root)
        dialog.title("Fine partita")
        dialog.geometry("420x220")

        tk.Label(dialog, text=message, font=("Helvetica", 12), pady=20).pack()
        tk.Label(
            dialog,
            text=(
                f"Tempo totale partita: {self.format_seconds(self.total_game_time())}\n"
                f"{self.player_display_name('Red')}: {self.format_seconds(self.thinking_time['Red'])}\n"
                f"{self.player_display_name('Blue')}: {self.format_seconds(self.thinking_time['Blue'])}"
            ),
            font=("Helvetica", 11),
            pady=10,
        ).pack()
        ok_button = tk.Button(dialog, text="OK", command=dialog.destroy, font=("Helvetica", 12), height=1)
        ok_button.pack(pady=10)

        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)


def main():
    root = tk.Tk()
    root.withdraw()

    mode = simpledialog.askinteger(
        "Seleziona modalita",
        "Seleziona modalita:\n1: Umano vs Umano\n2: Umano vs AI\n3: AI vs AI",
        minvalue=1,
        maxvalue=3,
        parent=root,
    )
    if mode is None:
        root.destroy()
        return

    timeout = simpledialog.askinteger(
        "Timeout",
        "Timeout per mossa AI (secondi):",
        minvalue=1,
        initialvalue=3,
        parent=root,
    )
    if timeout is None:
        timeout = 3

    red_name = simpledialog.askstring(
        "Nome Rosso",
        "Nome del giocatore rosso:",
        initialvalue="Rosso",
        parent=root,
    )
    if red_name is None or not red_name.strip():
        red_name = "Rosso"

    blue_name = simpledialog.askstring(
        "Nome Blu",
        "Nome del giocatore blu:",
        initialvalue="Blu",
        parent=root,
    )
    if blue_name is None or not blue_name.strip():
        blue_name = "Blu"

    if mode == 1:
        player_types = {"Blue": "human", "Red": "human"}
    elif mode == 2:
        human_player = simpledialog.askstring(
            "Giocatore umano",
            "Quale giocatore e umano? (Red o Blue):",
            parent=root,
        )
        human_player = (human_player or "Red").capitalize()
        if human_player != "Blue":
            human_player = "Red"

        player_types = {
            "Blue": "human" if human_player == "Blue" else "ai",
            "Red": "human" if human_player == "Red" else "ai",
        }
    else:
        player_types = {"Blue": "ai", "Red": "ai"}

    root.destroy()

    game = ZolaGame(size=8, first_player="Red")
    gui = ZolaGUI(
        game,
        player_types,
        time_out=timeout,
        player_names={"Red": red_name.strip(), "Blue": blue_name.strip()},
    )
    gui.run_game_loop()


if __name__ == "__main__":
    main()
