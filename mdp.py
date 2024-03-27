import random

class MDP:
    """ Return all states of this MDP """
    def __init__(self, width, height, start, goals, walls, negative):
        self.width = width
        self.height = height
        self.start = start
        self.goals = goals
        self.walls = walls
        self.negative = negative
        self.discount_factor = 0.9  # example discount factor
    def get_states(self):
        return [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.walls]

    """ Return all actions with non-zero probability from this state """

    def get_actions(self, state):
        actions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = state[0] + dx, state[1] + dy
            if (new_x, new_y) not in self.walls and 0 <= new_x < self.width and 0 <= new_y < self.height:
                actions.append((dx, dy))
        return actions

    """ Return all non-zero probability transitions for this action
        from this state, as a list of (state, probability) pairs
    """

    def get_transitions(self, state, action):
        return [(self.move(state, action), 1.0)]
    """ Return the reward for transitioning from state to
        nextState via action
    """
    def move(self, state, action):
        # Applies an action to a state
        new_state = (state[0] + action[0], state[1] + action[1])
        if new_state in self.walls or new_state[0] not in range(self.width) or new_state[1] not in range(self.height):
            return state  # if hits a wall or goes out of bounds, stay in same state
        return new_state

    def get_reward(self, state, action, next_state):
        if next_state in self.goals:
            return self.goals[next_state]
        if next_state in self.negative:
            return -1  # or some negative value
        return -0.01  # cost of living

    """ Return true if and only if state is a terminal state of this MDP """

    def is_terminal(self, state):
        return state in self.goals or state in self.negative

    """ Return the discount factor for this MDP """

    def get_discount_factor(self):
        return self.discount_factor


    """ Return the initial state of this MDP """

    def get_initial_state(self):
        return self.start
    """ Return all goal states of this MDP """

    def get_goal_states(self):
        return list(self.goals.keys())

    """ Return a new state and a reward for executing action in state,
    based on the underlying probability. This can be used for
    model-free learning methods, but requires a model to operate.
    Override for simulation-based learning
    """

    def execute(self, state, action):
        rand = random.random()
        cumulative_probability = 0.0
        for (new_state, probability) in self.get_transitions(state, action):
            if cumulative_probability <= rand <= probability + cumulative_probability:
                return (new_state, self.get_reward(state, action, new_state))
            cumulative_probability += probability
            if cumulative_probability >= 1.0:
                raise (
                    "Cumulative probability >= 1.0 for action "
                    + str(action)
                    + " from "
                    + str(state)
                )

        raise (
            "No outcome state in simulation for action"
            + str(action)
            + " from "
            + str(state)
        )

    """ Execute a policy on this mdp for a number of episodes """

    def execute_policy(self, policy, episodes=100):
        for _ in range(episodes):
            state = self.get_initial_state()
            while not self.is_terminal(state):
                action = policy.select_action(state)
                (next_state, reward) = self.execute(state, action)
                state = next_state

width = 3
height = 3
start = (0, 0)
goals = {(2, 2): 1}
walls = []
negative = {(1, 1): -1}
grid_world_mdp = MDP(width, height, start, goals, walls, negative)