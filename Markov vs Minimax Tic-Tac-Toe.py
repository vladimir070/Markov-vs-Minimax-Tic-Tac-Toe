import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Глобальные переменные
board = [' '] * 9
current_player = 'X'
last_computer_move = None
game_history = []
buttons = []
markov_chain = None
game_count = 0
x_wins = 0
o_wins = 0
draws = 0
training_active = False
learning_rate = 0.5
epsilon = 0.1
human_player = 'X'
ai_player = 'O'
ai_type = "markov"  # Тип ИИ по умолчанию: цепь Маркова

game_history_list = []
x_wins_history = []
o_wins_history = []
draws_history = []

# Статистика во время обучения
training_game_count = 0
training_x_wins = 0
training_o_wins = 0
training_draws = 0

winning_combinations = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

def initialize_markov_chain():
    chain = {}
    for i in range(9):
        chain[i] = {}
        for j in range(9):
            chain[i][j] = 1.0
    return chain

def load_markov_chain(filename="markov_chain.json"):
    global markov_chain, game_count, x_wins, o_wins, draws, game_history_list, x_wins_history, o_wins_history, draws_history, training_game_count, training_x_wins, training_o_wins, training_draws
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            markov_chain = data.get('markov_chain', initialize_markov_chain())
            game_count = data.get('game_count', 0)
            x_wins = data.get('x_wins', 0)
            o_wins = data.get('o_wins', 0)
            draws = data.get('draws', 0)
            game_history_list = data.get('game_history_list', [])
            x_wins_history = data.get('x_wins_history', [])
            o_wins_history = data.get('o_wins_history', [])
            draws_history = data.get('draws_history', [])

            # Загрузка статистики обучения
            training_game_count = data.get('training_game_count', 0)
            training_x_wins = data.get('training_x_wins', 0)
            training_o_wins = data.get('training_o_wins', 0)
            training_draws = data.get('training_draws', 0)

    except (FileNotFoundError, json.JSONDecodeError):
        markov_chain = initialize_markov_chain()
        game_count, x_wins, o_wins, draws = 0, 0, 0, 0
        game_history_list, x_wins_history, o_wins_history, draws_history = [], [], [], []
        training_game_count, training_x_wins, training_o_wins, training_draws = 0, 0, 0, 0

    finally:
        if markov_chain is None:
            markov_chain = initialize_markov_chain()

def save_markov_chain(filename="markov_chain.json"):
    data = {
        'markov_chain': markov_chain,
        'game_count': game_count,
        'x_wins': x_wins,
        'o_wins': o_wins,
        'draws': draws,
        'game_history_list': game_history_list,
        'x_wins_history': x_wins_history,
        'o_wins_history': o_wins_history,
        'draws_history': draws_history,

        # Сохранение статистики обучения
        'training_game_count': training_game_count,
        'training_x_wins': training_x_wins,
        'training_o_wins': training_o_wins,
        'training_draws': training_draws
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def reset_board(update_gui=True):
    global board, current_player, last_computer_move, game_history
    board = [' '] * 9
    current_player = 'X'
    last_computer_move = None
    game_history = []
    if buttons and update_gui:
        for button in buttons:
            button.config(text=' ')

def check_winner(board, player):
    return any(all(board[i] == player for i in combo) for combo in winning_combinations)

def is_board_full(board):
    return ' ' not in board

def evaluate_board(board):
    if check_winner(board, human_player):
        return -1
    elif check_winner(board, ai_player):
        return 1
    elif is_board_full(board):
        return 0
    else:
        return None

def minimax(board, depth, maximizing_player, alpha, beta):
    result = evaluate_board(board)

    if result is not None:
        return result

    if maximizing_player:
        max_eval = float('-inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = ai_player
                eval = minimax(board, depth + 1, False, alpha, beta)
                board[i] = ' '
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = human_player
                eval = minimax(board, depth + 1, True, alpha, beta)
                board[i] = ' '
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

def find_best_move(board):
    best_move = None
    best_eval = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    for i in range(9):
        if board[i] == ' ':
            board[i] = ai_player
            move_eval = minimax(board, 0, False, alpha, beta)
            board[i] = ' '

            if move_eval > best_eval:
                best_eval = move_eval
                best_move = i
            alpha = max(alpha, move_eval)
    return best_move

def update_markov_chain(prev_move, next_move, reward):
    if prev_move is None or next_move is None: return
    if prev_move in markov_chain and next_move in markov_chain[prev_move]:
        markov_chain[prev_move][next_move] += learning_rate * reward
        total = sum(markov_chain[prev_move].values()) + 0.0001
        for move in markov_chain[prev_move]:
            markov_chain[prev_move][move] = markov_chain[prev_move][move] / total

def get_computer_move(board, chain, last_move, ai_type):
    if ai_type == "markov":
        possible_moves = [i for i, cell in enumerate(board) if cell == ' ']
        if not possible_moves: return None
        if last_move is None or random.random() < epsilon: return random.choice(possible_moves)
        probs = []
        for move in possible_moves:
            prob = 0.1 if last_move is None or last_move not in chain or move not in chain[last_move] else chain[last_move].get(move, 0.1)
            probs.append(prob)
        total = sum(probs)
        probs = [p / total for p in probs] if total > 0 else [1 / len(possible_moves)] * len(possible_moves)
        return np.random.choice(possible_moves, p=probs)
    elif ai_type == "minimax":
        best_move = find_best_move(board)
        possible_moves = [i for i, cell in enumerate(board) if cell == ' ']
        if best_move in possible_moves:
            return best_move
        else:
            return random.choice(possible_moves) if possible_moves else None
    else:
        return None

def play_move(position):
    global board, current_player, last_computer_move, game_history
    if board[position] == ' ':
        board[position] = current_player
        buttons[position].config(text=current_player)
        if check_winner(board, current_player):
            update_statistics(current_player)
            reset_board()
            return
        if is_board_full(board):
            update_statistics(None)
            reset_board()
            return
        # Ход компьютера после хода игрока
        current_player = 'O'
        computer_move()  # Вызываем ход компьютера после хода человека
        current_player = 'X'
def computer_move():
    global board, current_player, last_computer_move, game_history, ai_type
    move = get_computer_move(board, markov_chain, last_computer_move, ai_type)
    if move is not None:
        board[move] = current_player
        buttons[move].config(text=current_player)
        if last_computer_move is not None:
            game_history.append((last_computer_move, move))
        last_computer_move = move
        if check_winner(board, current_player):
            update_statistics(current_player)
            reset_board()
            return
        if is_board_full(board):
            update_statistics(None)
            reset_board()
            return

        current_player = 'X'

def train_markov_chain(num_games=100, update_gui=True):
    global board, current_player, last_computer_move, game_history, game_count, x_wins, o_wins, draws, training_active, game_history_list, x_wins_history, o_wins_history, draws_history, training_game_count, training_x_wins, training_o_wins, training_draws
    for _ in range(num_games):
        if not training_active: return

        training_game_count += 1  # Увеличиваем счетчик игр обучения
        board = [' '] * 9
        current_player = 'X'  # Minimax всегда ходит первым во время обучения
        last_computer_move = None
        game_history = []

        while True:
            # Ход Minimax (X)
            move_x = get_computer_move(board, markov_chain, last_computer_move, "minimax")
            if move_x is None: break
            board[move_x] = current_player
            if last_computer_move is not None:
                game_history.append((last_computer_move, move_x))
            last_computer_move = move_x

            if check_winner(board, current_player):
                training_x_wins += 1
                reward = -1  # Minimax проиграл (O выиграл), плохая награда
                break
            if is_board_full(board):
                training_draws += 1
                reward = 1  # Ничья - небольшая награда
                break

            current_player = 'O'  # Ход Markov

            # Ход Markov (O)
            move_o = get_computer_move(board, markov_chain, last_computer_move, "markov")
            if move_o is None: break
            board[move_o] = current_player
            if last_computer_move is not None:
                game_history.append((last_computer_move, move_o))
            last_computer_move = move_o

            if check_winner(board, current_player):
                training_o_wins += 1
                reward = 10  # Markov выиграл (O выиграл), хорошая награда
                break
            if is_board_full(board):
                training_draws += 1
                reward = 1 # Ничья - небольшая награда
                break

            current_player = 'X'  # Снова ход Minimax (X)

        for prev_move, next_move in game_history:
            update_markov_chain(prev_move, next_move, reward)

        # Добавляем данные в списки истории
        game_history_list.append(training_game_count)
        x_wins_history.append(training_x_wins)
        o_wins_history.append(training_o_wins)
        draws_history.append(training_draws)

        if training_game_count % 100 == 0:
            if update_gui:
                root.after(0, update_statistics_display)
            root.after(0, update_plot)
        save_markov_chain()

def update_statistics(winner):
    global game_count, x_wins, o_wins, draws, game_history_list, x_wins_history, o_wins_history, draws_history
    game_count += 1
    if winner == 'X': x_wins += 1
    elif winner == 'O': o_wins += 1
    else: draws += 1
    game_history_list.append(game_count)
    x_wins_history.append(x_wins)
    o_wins_history.append(o_wins)
    draws_history.append(draws)
    root.after(0, update_statistics_display)
    root.after(0, update_plot)

def update_statistics_display():
    game_count_label.config(text=f"Игр сыграно: {game_count}")
    x_wins_label.config(text=f"X Побед: {x_wins}")
    o_wins_label.config(text=f"O Побед: {o_wins}")
    draws_label.config(text=f"Ничьих: {draws}")
    training_stats_label.config(text=f"Обучение: Игр={training_game_count}, X={training_x_wins}, O={training_o_wins}, Ничьи={training_draws}")

def ai_vs_ai_training():
    global training_active, game_history_list, x_wins_history,o_wins_history, draws_history
    if not training_active:
        training_active = True
        ai_vs_ai_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        #Очистка списков статистики для обучения
        game_history_list = []
        x_wins_history = []
        o_wins_history = []
        draws_history = []
        threading.Thread(target=train_thread).start()

def check_training_status(): #Ф-ция для проверки статуса
    global training_game_count, training_active
    if training_active:
        if training_game_count > 1000 and training_draws / training_game_count >= 0.95:
            stop_ai_training()
        else:
            threading.Thread(target=train_thread).start() #Запуск потока
    else:
        ai_vs_ai_button.config(state=tk.NORMAL) #Разблокировка Кнопки
        stop_button.config(state=tk.DISABLED) #Блокировка Stop Botton

def stop_ai_training():
    global training_active
    training_active = False
    ai_vs_ai_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    messagebox.showinfo("Обучение", "Обучение остановлено. Прогресс сохранено.")

def update_plot():
    if not axes: return

    axes.clear()

    # Определите, какой текст использовать для надписей графиков
    x_label = 'X (Minimax)'
    o_label = 'O (Markov)'

    axes.plot(game_history_list, x_wins_history, label=x_label, color='blue')
    axes.plot(game_history_list, o_wins_history, label=o_label, color='red')
    axes.plot(game_history_list, draws_history, label='Ничьи', color='green')
    axes.set_xlabel('Игры')
    axes.set_ylabel('Количество')
    axes.set_title('Статистика игр')
    axes.legend()
    axes.grid(True)
    figure_canvas.draw()

def set_ai_type(type):
    global ai_type
    ai_type = type
    print(f"Тип ИИ изменен на: {ai_type}")
    update_plot() # Обновить график при смене типа ИИ

def reset_statistics():
    global game_count, x_wins, o_wins, draws, game_history_list, x_wins_history, o_wins_history, draws_history
    game_count, x_wins, o_wins, draws = 0, 0, 0, 0
    game_history_list, x_wins_history, o_wins_history, draws_history = [], [], [], []
    update_statistics_display()
    update_plot()

def reset_training_statistics():
    global training_game_count, training_x_wins, training_o_wins, training_draws, game_history_list, x_wins_history, o_wins_history, draws_history

    # Очищаем списки, чтобы график начинался с нуля
    game_history_list.clear()
    x_wins_history.clear()
    o_wins_history.clear()
    draws_history.clear()

    training_game_count, training_x_wins, training_o_wins, training_draws = 0, 0, 0, 0
    update_statistics_display()
    update_plot()

# Автосохранение
def auto_save():
    root.after(5000, auto_save)

def train_thread():
    train_markov_chain(100, update_gui=False)
    root.after(0, update_statistics_display)
    root.after(0, update_plot)
    root.after(0,save_markov_chain) #Сохранение модели
    root.after(0,check_training_status) #Проверка

# GUI
root = tk.Tk()
root.title("Крестики-нолики с цепью Маркова")

game_frame = tk.Frame(root)
game_frame.pack(side=tk.LEFT, padx=10, pady=10)

for i in range(9):
    button = tk.Button(game_frame, text=" ", font=("Arial", 24), width=3, height=1, command=lambda pos=i: play_move(pos))
    button.grid(row=i // 3, column=i % 3)
    buttons.append(button)

control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

statistics_frame = tk.Frame(control_frame)
statistics_frame.pack()

game_count_label = tk.Label(statistics_frame, text="Игр сыграно: 0")
game_count_label.pack()
x_wins_label = tk.Label(statistics_frame, text="X Побед: 0")
x_wins_label.pack()
o_wins_label = tk.Label(statistics_frame, text="O Побед: 0")
o_wins_label.pack()
draws_label = tk.Label(statistics_frame, text="Ничьих: 0")
draws_label.pack()

# Статистика во время обучения
training_stats_label = tk.Label(statistics_frame, text="Обучение: Игр=0, X=0, O=0, Ничьи=0")
training_stats_label.pack()

fig = plt.Figure(figsize=(6, 4), dpi=100)
axes = fig.add_subplot(111)
figure_canvas = FigureCanvasTkAgg(fig, master=control_frame)
figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Меню
menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Действия", menu=file_menu)

#Add The AI training in AI vs AI
ai_vs_ai_button = tk.Button(file_menu, text="Обучить ИИ (AI vs AI)", command=ai_vs_ai_training)
file_menu.add_command(label="Обучить ИИ (AI vs AI)", command=ai_vs_ai_training)

#Create "Stop" button
stop_button = tk.Button(control_frame, text="Стоп обучение", command=stop_ai_training, state=tk.DISABLED)

# Add "Stop" to menu
file_menu.add_command(label="Остановить обучение", command=stop_ai_training)
file_menu.add_separator()  # Разделитель

# Функция для создания пунктов меню выбора ИИ
def create_ai_selection_menu():
    ai_menu = tk.Menu(file_menu, tearoff=0)
    file_menu.add_cascade(label="Выбрать ИИ", menu=ai_menu)
    ai_menu.add_command(label="Играть с Markov ИИ", command=lambda: set_ai_type("markov"))
    ai_menu.add_command(label="Играть с Minimax ИИ", command=lambda: set_ai_type("minimax"))

#Create function to  clear training statistics
def create_reset_statitistics():
    ai_menu = tk.Menu(file_menu, tearoff=0)
    file_menu.add_cascade(label="Сбросить статистику", menu=ai_menu)
    ai_menu.add_command(label="Сбросить игровую статистику", command=reset_statistics)
    ai_menu.add_command(label="Сбросить статистику обучения", command=reset_training_statistics)

create_ai_selection_menu()
create_reset_statitistics()

# Add menu to main windpow
root.config(menu=menu_bar)

# Инициализация
load_markov_chain()
update_statistics_display()
update_plot()
reset_board()

# Автосохранение
auto_save()

root.mainloop()