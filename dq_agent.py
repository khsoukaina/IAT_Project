from mdp import MDP

class dp_agent():
    def __init__(self, mdp):
        self.mdp = mdp
        self.value_function = {state: 0 for state in self.mdp.get_states()}

    def get_value(self, state):
        return self.value_function[state]
    
    def get_width(self, v, v_bis):
        return max(abs(v[s] - v_bis[s]) for s in self.mdp.get_states())

    def solve(self):
        while True:
            v_bis = self.value_function.copy()
            delta = 0
            for state in self.mdp.get_states():
                if not self.mdp.is_terminal(state):
                    max_value = max(self.mdp.get_reward(state, action, None) + self.mdp.get_discount_factor() * sum(prob * v_bis[next_state] for (next_state, prob) in self.mdp.get_transitions(state, action)) for action in self.mdp.get_actions(state))
                    self.value_function[state] = max_value
                    delta = max(delta, abs(max_value - v_bis[state]))
            if delta < 1e-4:  # A threshold for stopping, should be adjusted according to your specific needs
                break

    def update(self, state):
        best_value = float('-inf')
        for action in self.mdp.get_actions(state):
            total = sum(prob * (self.mdp.get_reward(state, action, next_state) + self.mdp.get_discount_factor() * self.value_function[next_state]) for next_state, prob in self.mdp.get_transitions(state, action))
            if total > best_value:
                best_value = total
        self.value_function[state] = best_value
