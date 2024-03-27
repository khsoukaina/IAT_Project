import random
from collections import defaultdict
from mdp import MDP

class q_agent:
    def __init__(self, mdp):
        self.mdp = mdp
        self.q_values = defaultdict(lambda: defaultdict(float))  # Nested defaultdict for storing Q values

    def greedy(self, state):
        possible_actions = self.mdp.get_actions(state)
        best_action = max(possible_actions, key=lambda action: self.q_values[state][action])
        return best_action

    def solve(self, episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        for episode in range(episodes):
            state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(state):
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(self.mdp.get_actions(state))
                else:
                    action = self.greedy(state)
                
                next_state, reward = self.mdp.execute(state, action)
                delta = self.get_delta(reward, self.q_values[state][action], state, next_state)
                self.q_values[state][action] += learning_rate * delta
                state = next_state

    def get_delta(self, reward, q_value, state, next_state):
        best_next_action = self.greedy(next_state)
        best_future_q = self.q_values[next_state][best_next_action]
        return reward + self.mdp.get_discount_factor() * best_future_q - q_value

    def state_value(self, state):
        return max(self.q_values[state].values())
