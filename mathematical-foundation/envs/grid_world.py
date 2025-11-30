import gymnasium as gym
from gymnasium import spaces

class GridWorld(gym.Env):
    def __init__(self, height=7, width=7):
        self.height = height
        self.width = width
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple((spaces.Discrete(height), spaces.Discrete(width)))
        
        # action space: 0-up, 1-down, 2-left, 3-right, 4-stay
        self._action_to_direction = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0)
        }
        
        # reward configuration
        self.target_reward = 1.0
        self.forbidden_reward = -10.0
        self.boundary_reward = -1.0
        self.step_reward = 0.0
        
        # grid configuration
        self._agent_location = (0, 0)
        self.target_state = (3, 2)
        self.forbidden_states = [(1,1), (1,2), (2,2), (2,6), (3,1), (3,3), (4,1), (4,3), (4,5), (5,1), (5,4), (6,2)]

        # define the reward map
        self.rewards = {}
        self.rewards[self.target_state] = self.target_reward
        for s in self.forbidden_states:
            self.rewards[s] = self.forbidden_reward

    def get_all_states(self):
        states = []
        for i in range(self.height):
            for j in range(self.width):
                states.append((i, j))
        return states

    def get_transition(self, state, action_idx):
        """ put in state and action, return next_state and reward """
        row, col = state
        d_row, d_col = self._action_to_direction[action_idx]
        
        next_row = row + d_row
        next_col = col + d_col
        
        # check the boundary
        if next_row < 0 or next_row >= self.height or next_col < 0 or next_col >= self.width:
            next_state = state
            reward = self.boundary_reward 
        else:
            next_state = (next_row, next_col)
            reward = self.rewards.get(next_state, self.step_reward)
        
        return next_state, reward