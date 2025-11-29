import numpy as np

class MCExploringStarts:
    """
    MC Exploring Starts algorithm, refering to chapter 5.3, algorithm 5.2
    """

    def __init__(self, env, gamma=0.9, epsilon=1e-8, num_episodes=100, max_episode_steps=50):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        
        self.states = self.env.get_all_states()
        self.actions = range(self.env.action_space.n)
        
        self.q_value = {(s, a): 0.0 for s in self.states for a in self.actions}
        self.visits = {(s, a): 0 for s in self.states for a in self.actions}
        self.policy = {s: np.random.choice(self.actions) for s in self.states}
        self.value = {s: 0.0 for s in self.states}

    def _generate_episode(self, start_state, start_action):
        """ generate one episode starting explicitly from (start_state, start_action) """
        episode = []
        
        # force the first step i.e. exploring start
        cur_state = start_state
        next_state, reward = self.env.get_transition(cur_state, start_action)
        episode.append((cur_state, start_action, reward))
        
        cur_state = next_state
        
        # follow current policy for the rest of the episode
        for _ in range(self.max_episode_steps):
            # select action based on current deterministic policy
            action = self.policy[cur_state]
            next_state, reward = self.env.get_transition(cur_state, action)
            
            episode.append((cur_state, action, reward))
            cur_state = next_state
            
        return episode

    def train(self):
        """ 
        apply MC Exploring Starts.
        process: generate episode -> calculate returns -> evaluate policy i.e. update q -> improve policy
        """
        for epoch in range(self.num_episodes):
            error = 0
            # loop through all state-action pairs i.e. satisfy exploring starts condition
            for start_state in self.states:
                for start_action in self.actions:
                    # generate an episode starting from (start_state, start_action)
                    episode = self._generate_episode(start_state, start_action)
                    
                    # iterate backwards to calculate returns g
                    g = 0
                    for t in range(len(episode)-1, -1, -1):
                        s, a, r = episode[t]
                        g = self.gamma * g + r

                        self.visits[(s, a)] += 1
                        n = self.visits[(s, a)]
                        old_q = self.q_value[(s, a)]
                        
                        # policy evaluation: update q-value
                        new_q = old_q + (g - old_q) / n
                        self.q_value[(s, a)] = new_q
                        error = max(error, abs(new_q - old_q))
                        
                        # policy improvement: update policy for state s immediately
                        best_action = self.actions[0]
                        best_q = -float('inf')
                        
                        for action_idx in self.actions:
                            q = self.q_value[(s, action_idx)]
                            if q > best_q:
                                best_q = q
                                best_action = action_idx
                        
                        self.policy[s] = best_action
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"MC Exploring Starts: Epoch {epoch+1}/{self.num_episodes} completed.")

            if error < self.epsilon:
                print(f"MC Exploring Starts converged at epoch {epoch+1} with error {error:.6f}")
                break

        # calculate state values for visualization based on final q
        for s in self.states:
            self.value[s] = self.q_value[(s, self.policy[s])]

    def get_policy(self):
        return self.policy