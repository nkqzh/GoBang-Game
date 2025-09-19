# ============================
# FILE: gomoku/mcts.py
# ============================
from __future__ import annotations
import copy
import numpy as np
from operator import itemgetter

# softmax with stability

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p


    def expand(self, action_priors):
        for a, p in action_priors:
            if a not in self._children:
                self._children[a] = TreeNode(self, p)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=100):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout


    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.step(action)
        action_priors, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_priors)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.current_player else -1.0
        node.update_recursive(-leaf_value)


    def get_move_probs(self, state, temp=1e-3):
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        act_visits = [(a, n._n_visits) for a, n in self._root._children.items()]
        acts, visits = zip(*act_visits) if act_visits else ([], [])
        if not acts:
            return [], []
        probs = softmax((1.0 / temp) * np.log(np.array(visits) + 1e-10))
        return list(acts), probs


    def get_move(self, state):
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        if not self._root._children:
            return None
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]


    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=100, is_selfplay=False, temperature=1.0,
        noise_eps=0.75, dirichlet_alpha=0.3, board_area=36):
        self.mcts = MCTS(policy_value_function, c_puct=c_puct, n_playout=n_playout)
        self._is_selfplay = is_selfplay
        self.temperature = temperature
        self.noise_eps = noise_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.board_area = board_area
        self.player = None


def set_player_ind(self, p):
    self.player = p


def reset_player(self):
    self.mcts.update_with_move(-1)


def get_action(self, env, return_prob=False):
    sensible_moves = env.availables
    move_probs = np.zeros(self.board_area, dtype=np.float32)
    if len(sensible_moves) > 0:
        acts, probs = self.mcts.get_move_probs(env, self.temperature)
        if not acts:
            # fallback: 随机
            a = np.random.choice(sensible_moves)
            if return_prob:
                return a, move_probs
            return a
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # Dirichlet 噪声鼓励自博弈探索
            a = np.random.choice(acts, p=self.noise_eps * probs + (1 - self.noise_eps) *
            np.random.dirichlet(self.dirichlet_alpha * np.ones(len(probs))))
            self.mcts.update_with_move(a)
        else:
            a = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        if return_prob:
            return a, move_probs
        return a
    else:
        return None