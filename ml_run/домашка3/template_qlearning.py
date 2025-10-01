import numpy as np
import random
from collections import defaultdict


def my_softmax(values: np.ndarray, T=1.):
    shifted_values = values - np.max(values)
    exp_values = np.exp(shifted_values / T)
    probas = exp_values / np.sum(exp_values)
    assert probas is not None
    return probas


class QLearningAgent:
    def __init__(self, alpha, discount, get_legal_actions, temp=1.):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.discount = discount
        self.temp = temp

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value

    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        
        if len(possible_actions) == 0:
            return 0.0
        
        q_values = [self.get_qvalue(state, action) for action in possible_actions]
        value = np.max(q_values)
        
        return value

    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        
        current_q = self.get_qvalue(state, action)
        next_value = self.get_value(next_state)
        qvalue = (1 - learning_rate) * current_q + learning_rate * (reward + gamma * next_value)
        
        self.set_qvalue(state, action, qvalue)

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        
        if len(possible_actions) == 0:
            return None
        
        q_values = [self.get_qvalue(state, action) for action in possible_actions]
        best_action = possible_actions[np.argmax(q_values)]
        
        return best_action

    def get_softmax_policy(self, state):
        possible_actions = self.get_legal_actions(state)
        
        if len(possible_actions) == 0:
            return None
        
        q_values = np.array([self.get_qvalue(state, action) for action in possible_actions])
        probabilities = my_softmax(q_values, self.temp)
        
        return probabilities

    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        
        if len(possible_actions) == 0:
            return None
        
        action_probs = self.get_softmax_policy(state)
        chosen_action = np.random.choice(possible_actions, p=action_probs)
        
        return chosen_action

class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of q-learning functions to implement Expected Value SARSA.
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    If it doesn't, please add
        def update(self, state, action, reward, next_state):
            and implement it for Expected Value SARSA's V(s')
    """

    def get_value(self, state):
        """
        Returns Vpi for current state under the softmax policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        possible_actions = self.get_legal_actions(state)
        
        if len(possible_actions) == 0:
            return 0.0

        action_probs = self.get_softmax_policy(state)
        value = 0.0
        for i, action in enumerate(possible_actions):
            value += action_probs[i] * self.get_qvalue(state, action)

            
        assert value is not None

        return value