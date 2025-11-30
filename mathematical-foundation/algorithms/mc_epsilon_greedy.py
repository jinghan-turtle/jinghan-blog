import numpy as np
import random

class MCEpsilonGreedy:
    """
    MC Epsilon-Greedy algorithm (On-policy Control) with GLIE.
    Reference: Chapter 5.4 & Monte_Carlo.py
    
    Correction:
    - Actions MUST be selected based on the current policy probabilities (On-Policy), 
      not uniformly random.
    - Q-values are updated based on the current trajectory (batch), ignoring stale history from old policies.
    """

    def __init__(self, env, gamma=0.9, epsilon=0.1, min_epsilon=0.1, decay_rate=0.999, num_episodes=1000, max_episode_steps=1000):
        self.env = env
        self.gamma = gamma
        
        # Epsilon decay parameters
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        
        self.states = self.env.get_all_states()
        self.n_actions = self.env.action_space.n
        self.actions = range(self.n_actions)
        
        # Initialize Q(s,a)
        self.q_value = {(s, a): 0.0 for s in self.states for a in self.actions}
        
        # Initialize Policy pi(a|s) - Uniform probability initially
        self.policy = {s: [1.0 / self.n_actions] * self.n_actions for s in self.states}
        
        # Initialize V(s) for visualization
        self.value = {s: 0.0 for s in self.states}
        
        # Global visits (for logging/debugging, similar to reference)
        self.visits = {(s, a): 0 for s in self.states for a in self.actions}

        self.returns_sum = {(s, a): 0.0 for s in self.states for a in self.actions}
        self.returns_count = {(s, a): 0 for s in self.states for a in self.actions}

    def _generate_episode(self, start_state):
        """ 
        Generate one episode starting from start_state following the current stochastic policy.
        """
        episode = []
        curr_state = start_state
        
        for _ in range(self.max_episode_steps):
            # [CORRECTION] On-Policy Sampling:
            # Must choose action according to the current policy distribution
            action_probs = self.policy[curr_state]
            action = np.random.choice(self.actions, p=action_probs)
            
            # Record global visits
            self.visits[(curr_state, action)] += 1
            
            # Simulate step
            next_state, reward = self.env.get_transition(curr_state, action)
            
            episode.append((curr_state, action, reward))
            curr_state = next_state
            
        return episode

    def train(self):
        """ 
        Apply MC Epsilon-Greedy with epsilon decay.
        Matching Logic from Monte_Carlo.py: e_greedy_MC
        """
        for epoch in range(self.num_episodes):
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            
            # Start from (0,0) as in Monte_Carlo.py (implied s=0)
            start_state = random.choice(self.states)
            
            # 1. Generate an episode (On-Policy)
            episode = self._generate_episode(start_state)
            
            G = 0
            # 2. Iterate backwards
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = self.gamma * G + r
                
                self.returns_sum[(s, a)] += G
                self.returns_count[(s, a)] += 1
                
                # 更新 Q 值 = 总回报 / 总次数
                self.q_value[(s, a)] = self.returns_sum[(s, a)] / self.returns_count[(s, a)]
                
                # --- Policy Improvement (Epsilon-Greedy) ---
                # Find best action based on updated Q
                best_action = self.actions[0]
                best_q = -float('inf')
                
                for action_idx in self.actions:
                    q = self.q_value.get((s, action_idx), 0.0)
                    if q > best_q:
                        best_q = q
                        best_action = action_idx
                
                # Update probabilities
                # Greedy action: 1 - epsilon + (epsilon / |A|)
                # Non-greedy actions: epsilon / |A|
                for action_idx in self.actions:
                    if action_idx == best_action:
                        self.policy[s][action_idx] = 1 - self.epsilon + (self.epsilon / self.n_actions)
                    else:
                        self.policy[s][action_idx] = self.epsilon / self.n_actions
                
                # Update state value for visualization
                self.value[s] = best_q
            
            if (epoch + 1) % 50 == 0:
                print(f"MC Epsilon-Greedy: Epoch {epoch+1}/{self.num_episodes} completed. Epsilon: {self.epsilon:.4f}")

    def get_policy(self):
        return self.policy