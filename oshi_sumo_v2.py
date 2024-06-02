import random
import math
import numpy as np
from typing import NamedTuple
from functools import cache
from scipy.optimize import linprog
from enum import Enum



def solve_game(payoff, max):
    """
    Find a mixed strategy for the zero-sum game described by the input matrix.
    It assumes that MAX-player chooses the rows.

    Parameters
    ----------
    * V: (n, m)-array; payoff matrix of zero-sum game;
    * row: bool (default True);
        True:  Compute strategy for MAX-player.
        False: Compute strategy for MIN-player.

    Returns
    -------
    * s: (n,)-array; probability vector of MAX-player (MIN-player if row=False).
    * v: float; Expected value of the mixed strategy.
    """
    # MAX-player choses the rows
    payoff = np.transpose(payoff)

    # MIN-player choses the columns
    if not max:
        payoff = -payoff.T

    if min(payoff.shape) == 1:
        # pure strategy
        redux = payoff.min(0)
        i = redux.argmax()
        value = redux[i]
        probs = np.zeros(len(redux))
        probs[i] = 1

    else:

        # ensure positive
        skew = 1 - payoff.min()
        Vpos = payoff + skew

        # solve linear program
        m, n = payoff.shape
        result = linprog(np.ones(n), -Vpos, -np.ones(m), bounds=[(0, 1)], method='highs')

        if result.status:
            raise Exception(result.message)

        # compute strategy and value
        value = 1 / result.x.sum()
        probs = result.x * value

        # correct skew
        value -= skew

    # Adjust payoff to MAX-player
    if not max:
        value = -value

    return value, probs


class RewardMethodEnum(Enum):
    stepReward = 0
    endReward = 1
    winTieLoseReward = 2

class OpponentPolicyEnum(Enum):
    Optimal = 0
    Random = 1

class OshiZumoState(NamedTuple):
    wrestler: int
    p1_coins: int
    p2_coins: int
    board_size: int
    both0: int

    def is_wrestler_off_board(self):
        return self.wrestler < 0 or self.wrestler > self.board_size

    def no_coins(self):
        return self.p1_coins == 0 and self.p2_coins == 0

    def is_draw(self):
        center = self.board_size // 2
        return self.wrestler == center

    def print_state(self, comment=""):
        print(comment, 'OshiZumoState(', self.wrestler, ',', self.p1_coins, ',', self.p2_coins, ',', self.board_size, ',', self.both0, ')')

class OshiZumoGame:
    def __init__(self, N, M, K, rewardMethod):
        self.N = N # Nb coins
        self.M = M # Minimal bid
        self.K = K # Field size
        self.TrackLengthK=2*K
        self.rewardMethod = rewardMethod

    def start_state(self):
        return OshiZumoState(self.K, self.N, self.N, self.TrackLengthK, 0)

    def is_terminal(self, state: OshiZumoState):
        """if state.is_wrestler_off_board() :
            print("off_board :", state.is_wrestler_off_board())
        if state.no_coins():
            print('no coins : ', state.no_coins())
        if state.p1_coins < self.M : 
            print('player 1 moins que M : ', state.p1_coins < self.M)
        if state.p2_coins < self.M:
            print('player 2 moins que M : ', state.p2_coins < self.M)
        if state.both0 == 1:
            print('both 0 : ', state.both0 == 1)"""
        return state.is_wrestler_off_board() or state.no_coins() or state.p1_coins < self.M or state.p2_coins < self.M or state.both0 == 1

    def max_actions(self, state: OshiZumoState):
        if state.both0 == 1:
            return []
        else:
            return range(state.p1_coins + 1)

    def min_actions(self, state: OshiZumoState):
        if state.both0 == 1:
            return []
        else:
            return range(state.p2_coins +1)

    def successors(self, state: OshiZumoState, p1_action: int, p2_action: int):

        next_p1_coins = state.p1_coins - p1_action
        next_p2_coins = state.p2_coins - p2_action
        move = 0
        if p1_action < p2_action:
            move = -1
        elif p1_action > p2_action:
            move = 1

        next_wrestler = state.wrestler + move
        reward = 0
        if self.rewardMethod == RewardMethodEnum.stepReward:
            reward = move

        next_state = OshiZumoState(next_wrestler, next_p1_coins, next_p2_coins, state.board_size, state.both0)
        return next_state, reward

def evaluation_reward(state: OshiZumoState, game: OshiZumoGame):
    b = 0
    if (state.p1_coins > state.p2_coins and state.wrestler >= game.K) or (
        state.p1_coins >= state.p2_coins and state.wrestler > game.K):
        b = 1
    if (state.p1_coins < state.p2_coins and state.wrestler <= game.K) or (
        state.p1_coins <= state.p2_coins and state.wrestler < game.K):
        b = -1
    reward = math.tanh(b / 2 + 1 / 3 * ((state.p1_coins - state.p2_coins) / game.M + state.wrestler - game.K))
    return reward

@cache
def minmax_value(state: OshiZumoState, game: OshiZumoGame, max):

    if game.is_terminal(state):
        reward_terminal = 0

        if game.rewardMethod == RewardMethodEnum.endReward:
            if state.wrestler > game.K:
                reward_terminal = 1
            elif state.wrestler < game.K:
                reward_terminal = -1
            else:
                reward_terminal = 0
        if game.rewardMethod == RewardMethodEnum.winTieLoseReward:
            reward_terminal = evaluation_reward(state, game)
        return reward_terminal, []

    max_actions = game.max_actions(state)
    min_actions = game.min_actions(state)

    max_actions = [a for a in max_actions if a >= game.M]
    min_actions = [a for a in min_actions if a >= game.M]

    Q = np.zeros([len(max_actions), len(min_actions)])
    for i, a1 in enumerate(max_actions):
        for j, a2 in enumerate(min_actions):
            new_state = OshiZumoState(state.wrestler, state.p1_coins, state.p2_coins, state.board_size, 1 if a1 == 0 and a2 == 0 else 0)
            next_state, reward = game.successors(new_state, a1, a2)
            value, probs = minmax_value(next_state, game, max)
            Q[i,j] = reward + value

    return solve_game(Q, max)
win = 0
lose = 0
tie = 0

def play_game(game: OshiZumoGame, policy: OpponentPolicyEnum):
    global win, lose, tie
    state = game.start_state()
#    state.print_state('Start: ')
    test = True
    while test == True:
        value, probs = minmax_value(state, game, True)
#        print (probs)
        index_max = np.argmax(probs) + game.M
        #print('probs max: ', probs)

        if policy == OpponentPolicyEnum.Optimal:
            value, probs = minmax_value(state, game, False)
            index_min = np.argmax(probs) + game.M
        else:
#            opponent_bid = random.randint(0, index_max + 1)
            opponent_bid = random.randint(0, state.p2_coins)
            if state.p2_coins < opponent_bid:
                opponent_bid = state.p2_coins
            if opponent_bid < game.M:
                opponent_bid = game.M
            index_min = opponent_bid

        next_state, reward = game.successors(state, index_max, index_min)

        #print('probs min: ', index_min)
        state = next_state
        #print('next_state : ', next_state)
        if (game.is_terminal(state) or (index_max == 0 and index_min == 0)):
            #print("game.is_terminal(state) : ", game.is_terminal(state))
            #print('index_max == 0 and index_min == 0', index_max == 0 and index_min == 0)
            test = False
#        state.print_state('Mid')
#    state.print_state('End: ')

    status = 0
    if game.rewardMethod == RewardMethodEnum.stepReward:
        rewardmethod = "stepReward"
        status = state.wrestler - game.K # position 3 : 0, position 1 = -2, position 4 : +1,..
    elif game.rewardMethod == RewardMethodEnum.endReward:
        rewardmethod = "endReward"
        status = state.wrestler - game.K
    elif game.rewardMethod == RewardMethodEnum.winTieLoseReward:
        rewardmethod = "winTieLoseReward"
        status = evaluation_reward(state, game)
    if status > 0:
        result = "player A > player B"
        win += 1
    elif status < 0:
        result = "player A < player B"
        lose += 1
    else:
        result = "player A = player B"
        tie +=1
    #print('game(', game.N, ',', game.M, ',', game.K, ')-', rewardmethod, ':', result)





"""
for _ in range(20):
    game = OshiZumoGame(20, 1, 3, RewardMethodEnum.winTieLoseReward)
    play_game(game, OpponentPolicyEnum.Optimal)

print(f"win : {win}, lose : {lose}, tie : {tie}")
"""
game = OshiZumoGame(20, 0, 3, RewardMethodEnum.stepReward)
play_game(game, OpponentPolicyEnum.Random)

print()
print()
print()

game = OshiZumoGame(20, 0, 3, RewardMethodEnum.endReward)
play_game(game, OpponentPolicyEnum.Random)
print()
print()
print()

game = OshiZumoGame(20, 1, 3, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)
print()
print()
print()

"""game = OshiZumoGame(20, 2, 3, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)
game = OshiZumoGame(20, 3, 3, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)

game = OshiZumoGame(30, 1, 3, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)
game = OshiZumoGame(40, 1, 3, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)
game = OshiZumoGame(50, 1, 3, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)

game = OshiZumoGame(20, 1, 4, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)
game = OshiZumoGame(20, 1, 5, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)
game = OshiZumoGame(20, 1, 6, RewardMethodEnum.winTieLoseReward)
play_game(game, OpponentPolicyEnum.Random)"""

#print('Optimal Opponent stepReward reward')
#game = OshiZumoGame(40, 3, 3, RewardMethodEnum.winTieLoseReward)
#play_game(game, OpponentPolicyEnum.Random)
#state=OshiZumoState(3,5,7,6,0)
#value, probs = minmax_value(state, game, True)

exit(0)