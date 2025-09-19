# ============================
# FILE: train.py
# ============================
import os
import time
import random
import argparse
from collections import deque, defaultdict
import numpy as np
from gomoku.env import GomokuEnv
from gomoku.policy_value_net import PolicyValueNet
from gomoku.mcts import MCTSPlayer

def get_args():
    p = argparse.ArgumentParser()
    # 棋盘/搜索参数
    p.add_argument('--board-width', type=int, default=6)
    p.add_argument('--board-height', type=int, default=6)
    p.add_argument('--n-in-row', type=int, default=4)
    p.add_argument('--c-puct', type=float, default=5.0)
    p.add_argument('--n-playout', type=int, default=100)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--noise-eps', type=float, default=0.75)
    p.add_argument('--dirichlet-alpha', type=float, default=0.3)
    # 训练超参
    p.add_argument('--learn-rate', type=float, default=2e-3)
    p.add_argument('--buffer-size', type=int, default=5000)
    p.add_argument('--train-batch-size', type=int, default=128)
    p.add_argument('--update-epochs', type=int, default=5)
    p.add_argument('--kl-coeff', type=float, default=0.02)
    p.add_argument('--checkpoint-freq', type=int, default=20)
    p.add_argument('--game-batch-num', type=int, default=40)
    p.add_argument('--save-dir', type=str, default='./checkpoints')
    p.add_argument('--restore-model', type=str, default=None)
    return p.parse_args()

def augment(play_data, board_w, board_h):
    # 旋转/翻转数据增强
    ext = []
    for state, mcts_p, winner in play_data:
        for i in [1, 2, 3, 4]:
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts = np.rot90(np.flipud(mcts_p.reshape(board_h, board_w)), i)
            ext.append((equi_state, np.flipud(equi_mcts).flatten(), winner))
            equi_state2 = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts2 = np.fliplr(equi_mcts)
            ext.append((equi_state2, np.flipud(equi_mcts2).flatten(), winner))
    return ext

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = GomokuEnv(args.board_width, args.board_height, args.n_in_row)
    policy_value = PolicyValueNet(args.board_width, args.board_height, model_file=args.restore_model)
    player = MCTSPlayer(policy_value.policy_value_fn, c_puct=args.c_puct, n_playout=args.n_playout,
    is_selfplay=True, temperature=args.temperature, noise_eps=args.noise_eps,
    dirichlet_alpha=args.dirichlet_alpha, board_area=args.board_width * args.board_height)

    data_buffer = deque(maxlen=args.buffer_size)
    lr_multiplier = 1.0
    best_win_ratio = 0.0
    mcts_infer = 200

    def collect_selfplay(n_games=1):
        nonlocal env, player, data_buffer
        for _ in range(n_games):
            env.reset()
            states, mcts_probs, current_players = [], [], []
            while True:
                move, move_probs = player.get_action(env, return_prob=True)
                states.append(env.current_state())
                mcts_probs.append(move_probs)
                current_players.append(env.current_player)
                env.step(move)
                end, winner = env.game_end()
                if end:
                    winners_z = np.zeros(len(current_players), dtype=np.float32)
                    if winner != -1:
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0
                    player.reset_player()
                    play_data = list(zip(states, mcts_probs, winners_z))
                    # 数据增强
                    play_data = augment(play_data, args.board_width, args.board_height)
                    data_buffer.extend(play_data)
                    return len(play_data)

    def policy_update():
        nonlocal data_buffer, lr_multiplier
        mini_batch = random.sample(data_buffer, args.train_batch_size)
        state_batch = [d[0] for d in mini_batch]
        mcts_batch = [d[1] for d in mini_batch]
        winner_batch = [d[2] for d in mini_batch]
        old_probs, _ = policy_value.policy_value(state_batch)
        for _ in range(args.update_epochs):
            loss, entropy = policy_value.train_step(state_batch, mcts_batch, winner_batch, args.learn_rate * lr_multiplier)
            new_probs, _ = policy_value.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > args.kl_coeff * 4:
                break
        if kl > args.kl_coeff * 2 and lr_multiplier > 0.1:
            lr_multiplier /= 1.5
        elif kl < args.kl_coeff / 2 and lr_multiplier < 10:
            lr_multiplier *= 1.5
        return loss, entropy

    def policy_evaluate(n_games=10):
        from gomoku.mcts import MCTS, MCTSPlayer as EvalPlayer
        cur = EvalPlayer(policy_value.policy_value_fn, c_puct=args.c_puct, n_playout=args.n_playout,
                         is_selfplay=False, temperature=1e-3, board_area=args.board_width * args.board_height)
        pure = EvalPlayer(lambda e: ([(a, 1 / len(e.availables)) for a in e.availables], 0),
                          c_puct=args.c_puct, n_playout=mcts_infer, is_selfplay=False,
                          temperature=1e-3, board_area=args.board_width * args.board_height)
        win = defaultdict(int)
        for i in range(n_games):
            env.reset()
            p_map = {env.players[0]: cur, env.players[1]: pure}
            while True:
                player_in_turn = p_map[env.current_player]
                move = player_in_turn.get_action(env)
                env.step(move)
                end, winner = env.game_end()
                if end:
                    win[winner] += 1
                    break
        win_ratio = 1.0 * (win[1] + 0.5 * win[-1]) / n_games
        print(f"num_playouts:{mcts_infer}, win:{win[1]}, lose:{win[2]}, tie:{win[-1]}")
        return win_ratio

    start_t = time.time()
    try:
        for i in range(args.game_batch_num):
            ep_len = collect_selfplay(1)
            print(f"batch {i + 1}/{args.game_batch_num}, episode_len:{ep_len}")
            if len(data_buffer) > args.train_batch_size:
                loss, entropy = policy_update()
            if (i + 1) % args.checkpoint_freq == 0:
                print(f"checkpoint @ {i + 1}")
                win_ratio = policy_evaluate()
                latest = os.path.join(args.save_dir, 'newest_model.pt')
                policy_value.save_model(latest)
                if win_ratio > best_win_ratio:
                    best = os.path.join(args.save_dir, 'best_model.pt')
                    policy_value.save_model(best)
    except KeyboardInterrupt:
        print('Interrupted')
    print(f"time cost: {time.time() - start_t:.2f}s")

if __name__ == '__main__':
    main()


# ============================
# FILE: play_pyqt.py
# ============================
import argparse
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QBrush
from PyQt5.QtCore import Qt
from gomoku.env import GomokuEnv
from gomoku.policy_value_net import PolicyValueNet
from gomoku.mcts import MCTSPlayer

CELL_SIZE = 32
MARGIN = 40

class GobangWidget(QWidget):
    def __init__(self, env: GomokuEnv, mcts_player: MCTSPlayer, human_first=True):
        super().__init__()
        self.env = env
        self.player = mcts_player
        self.human_first = human_first
        self.setWindowTitle('Gobang - AlphaZero')
        w = self.env.board_width * CELL_SIZE + 2 * MARGIN
        h = self.env.board_height * CELL_SIZE + 2 * MARGIN
        self.setFixedSize(w, h)
        self.reset_game()

    def reset_game(self):
        self.env.reset()
        # human 是 player1(1) 还是 player2(2)
        self.human_id = self.env.players[0] if self.human_first else self.env.players[1]
        # 如果 AI 先手，先走一步
        if not self.human_first:
            ai_move = self.player.get_action(self.env)
            self.env.step(ai_move)
        self.update()

    def paintEvent(self, e):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        # 背景
        qp.fillRect(self.rect(), Qt.lightGray)
        # 棋盘网格
        qp.setPen(QPen(Qt.black, 2))
        for i in range(self.env.board_height + 1):
            y = MARGIN + i * CELL_SIZE
            qp.drawLine(MARGIN, y, MARGIN + self.env.board_width * CELL_SIZE, y)
        for j in range(self.env.board_width + 1):
            x = MARGIN + j * CELL_SIZE
            qp.drawLine(x, MARGIN, x, MARGIN + self.env.board_height * CELL_SIZE)
        # 棋子
        for pos, pid in self.env.states.items():
            r, c = divmod(pos, self.env.board_width)
            x = MARGIN + c * CELL_SIZE
            y = MARGIN + r * CELL_SIZE
            qp.setBrush(QBrush(Qt.black if pid == 1 else Qt.white))
            qp.drawEllipse(x - CELL_SIZE//2 + CELL_SIZE//2, y - CELL_SIZE//2 + CELL_SIZE//2, CELL_SIZE-4, CELL_SIZE-4)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        # 轮到人类落子？
        if self.env.current_player != self.human_id:
            return
        # 将点击位置映射为格点
        x, y = e.x() - MARGIN, e.y() - MARGIN
        if x < 0 or y < 0:
            return
        c = int(round(x / CELL_SIZE))
        r = int(round(y / CELL_SIZE))
        if not (0 <= r < self.env.board_height and 0 <= c < self.env.board_width):
            return
        move = r * self.env.board_width + c
        if move not in self.env.availables:
            return
        # 人类走
        self.env.step(move)
        self.update()
        end, winner = self.env.game_end()
        if end:
            print('Game End. Winner:', winner)
            return
        # AI 走
        ai_move = self.player.get_action(self.env)
        self.env.step(ai_move)
        self.update()

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--board-width', type=int, default=6)
    ap.add_argument('--board-height', type=int, default=6)
    ap.add_argument('--n-in-row', type=int, default=4)
    args = ap.parse_args()

    env = GomokuEnv(args.board_width, args.board_height, args.n_in_row)
    policy_value = PolicyValueNet(args.board_width, args.board_height, model_file=args.model)
    mcts_player = MCTSPlayer(policy_value.policy_value_fn, c_puct=5, n_playout=100,
                             is_selfplay=False, temperature=1e-3,
                             board_area=args.board_width * args.board_height)

    app = QApplication(sys.argv)
    w = GobangWidget(env, mcts_player, human_first=True)
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()