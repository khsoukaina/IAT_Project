from mdp import *

class dp_agent():
    def __init__(self, mdp):
        self.mdp = mdp
        self.value_function = {state: 0 for state in mdp.get_states()}  # Initialiser toutes les valeurs à 0
        self.policy = {state: None for state in mdp.get_states()}  # Initialiser la politique à None

    def get_value(self, s):
        # Retourne la valeur de l'état spécifique s
        return self.value_function[s]

    def get_width(self, v_bis):
        # Retourne la norme absolue entre la fonction de valeur actuelle et v_bis
        return max(abs(self.value_function[state] - v_bis[state]) for state in self.mdp.get_states())

    def solve(self):
        # Boucle principale de résolution
        threshold = 0.001  # Seuil de convergence
        while True:
            delta = 0
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    continue
                
                v = self.get_value(state)
                max_value = float('-inf')
                
                for action in self.mdp.get_actions(state):
                    total = sum(prob * (self.mdp.get_reward(state, action, next_state) + 
                                        self.mdp.get_discount_factor() * self.get_value(next_state))
                                for next_state, prob in self.mdp.get_transitions(state, action))
                    max_value = max(max_value, total)
                
                self.value_function[state] = max_value
                delta = max(delta, abs(v - self.value_function[state]))
            
            if delta < threshold:
                break

    def update(self, s):
        # Met à jour la valeur de l'état spécifique s
        max_value = float('-inf')
        for action in self.mdp.get_actions(s):
            total = sum(prob * (self.mdp.get_reward(s, action, next_state) + 
                                self.mdp.get_discount_factor() * self.get_value(next_state))
                        for next_state, prob in self.mdp.get_transitions(s, action))
            max_value = max(max_value, total)
        
        self.value_function[s] = max_value
