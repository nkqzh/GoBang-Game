# ============================
# FILE: gomoku/env.py
# ============================
from __future__ import annotations
import numpy as np
from gym import spaces


class GomokuEnv:
    """简化 AlphaZero 自博弈所需的五子棋环境（无渲染，重在状态/合法动作/胜负判定）。

    状态 shape: [4, W, H]
    - [0]: 当前玩家的落子位置
    - [1]: 对手玩家的落子位置
    - [2]: 上一步的落子位置（one-hot）
    - [3]: 轮到当前执棋方标记（全 1 或 0）
    """
    def __init__(self, board_width=6, board_height=6, n_in_row=4, start_player=0):
        assert board_width >= n_in_row and board_height >= n_in_row
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.players = [1, 2]
        self.start_player = start_player
        self.action_space = spaces.Discrete(board_width * board_height)
        self.observation_space = spaces.Box(0, 1, shape=(4, board_width, board_height))
        self.reset()


    # -------- 基本操作 --------
    def reset(self):
        self.current_player = self.players[self.start_player]
        self.availables = list(range(self.board_width * self.board_height))
        self.states = {} # {flat_index: player_id}
        self.last_move = -1
        return self.current_state()


    def step(self, action: int):
        assert action in self.availables, "非法落子"
        self.states[action] = self.current_player
        self.availables.remove(action)
        self.last_move = action
        done, winner = self.game_end()
        reward = 0
        if done:
            if winner == self.current_player:
                reward = 1
            elif winner == -1: # 和棋
                reward = 0
            else:
                reward = -1
        # 切换执棋方
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        obs = self.current_state()
        return obs, reward, done, {}


    # -------- 判定/编码 --------
    def has_a_winner(self):
        w, h, n = self.board_width, self.board_height, self.n_in_row
        states, moved = self.states, list(set(range(w * h)) - set(self.availables))
        if len(moved) < n * 2 - 1:
            return False, -1
        for m in moved:
            r, c = divmod(m, w)
            p = states[m]
            # 横
            if c in range(w - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n))) == 1:
                return True, p
            # 竖
            if r in range(h - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n * w, w))) == 1:
                return True, p
            # 斜 \ 方向
            if c in range(w - n + 1) and r in range(h - n + 1) and \
                len(set(states.get(i, -1) for i in range(m, m + n * (w + 1), w + 1))) == 1:
                return True, p
            # 斜 / 方向
            if c in range(n - 1, w) and r in range(h - n + 1) and \
                len(set(states.get(i, -1) for i in range(m, m + n * (w - 1), w - 1))) == 1:
                return True, p
        return False, -1


    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        if not len(self.availables):
            return True, -1
        return False, -1


    def current_state(self):
        w, h = self.board_width, self.board_height
        s = np.zeros((4, w, h), dtype=np.float32)
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            cur = moves[players == self.current_player]
            opp = moves[players != self.current_player]
            s[0][cur // w, cur % h] = 1.0
            s[1][opp // w, opp % h] = 1.0
            if self.last_move >= 0:
                s[2][self.last_move // w, self.last_move % h] = 1.0
        if len(self.states) % 2 == 0:
            s[3][:, :] = 1.0
        # 注意：与文章一致，按行翻转以统一坐标系
        return s[:, ::-1, :]


    # -------- 辅助 --------
    def move_to_location(self, move: int):
        r, c = divmod(move, self.board_width)
        return [r, c]


    def location_to_move(self, loc):
        r, c = loc
        move = r * self.board_width + c
        if move not in range(self.board_width * self.board_height):
            return -1
        return move