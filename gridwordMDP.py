class GridWorldMDP:
    def __init__(self, width, height, start_state, terminal_states, transition_probs, rewards):
        self.width = width
        self.height = height
        self.start_state = start_state
        self.terminal_states = terminal_states
        self.transition_probs = transition_probs  # Probabilities of transitions for each action.
        self.rewards = rewards  # Rewards for transitions.

    def get_initial_state(self):
        return self.start_state

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_actions(self, state):
        if self.is_terminal(state):
            return []
        return ['up', 'down', 'left', 'right']  # Example actions

    def execute(self, state, action):
        if state in self.terminal_states:
            return state, 0  # No reward for terminal states.

        # Example deterministic transition function for a grid world.
        next_state = {
            'up': (state[0], state[1] + 1),
            'down': (state[0], state[1] - 1),
            'left': (state[0] - 1, state[1]),
            'right': (state[0] + 1, state[1]),
        }.get(action, state)  # Default to current state if action is unknown.

        # Ensure next_state is within grid bounds.
        next_state = (
            max(0, min(next_state[0], self.width - 1)),
            max(0, min(next_state[1], self.height - 1))
        )

        # Get reward for transition.
        reward = self.rewards.get((state, action, next_state), 0)

        return next_state, reward

# Define the grid size, start state, terminal states, and rewards.
width, height = 5, 5
start_state = (0, 0)
terminal_states = [(4, 4)]
transition_probs = {
    # Assuming deterministic environment for simplicity.
}
rewards = {
    ((1, 1), 'up', (1, 2)): 1,  # Just an example reward.
    # Add other rewards as needed.
}

# Create an instance of the MDP.
grid_world_mdp = GridWorldMDP(width, height, start_state, terminal_states, transition_probs, rewards)
