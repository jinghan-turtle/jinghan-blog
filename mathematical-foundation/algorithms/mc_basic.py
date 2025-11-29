import numpy as np

class MCBasic:
    """
    MC Basic algorithm: model-free variant of policy iteration
    reference: Chapter 5.2, Algorithm 5.1
    As can be seen, it is very similar to the policy iteration algorithm. The only difference is that 
    it calculates action values directly from experience samples, whereas policy iteration calculates 
    state values first and then calculates the action values based on the system model.
    """

    def __init__(self, env, gamma=0.9, epsilon=1e-8, episodes_per_eval=50, max_episode_steps=30):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes_per_eval = episodes_per_eval
        self.max_episode_steps = max_episode_steps
        
        self.states = self.env.get_all_states()
        self.actions = range(self.env.action_space.n)
        
        # initialize policy with uniform probability i.e. 0.2 for each of the 5 actions
        n_actions = self.env.action_space.n
        self.policy = {s: [1.0 / n_actions] * n_actions for s in self.states}
        
        # initialize q-value and state-value
        self.q_value = {(s, a): 0.0 for s in self.states for a in self.actions}
        self.value = {s: 0.0 for s in self.states}

    def _generate_episode_return(self, start_state, start_action):
        """
        generate one episode starting explicitly from (start_state, start_action)
        and calculate the discounted return gain in this episode
        """
        # execute the exploring start i.e. forced first step
        curr_state, reward = self.env.get_transition(start_state, start_action)
        gain = reward
        current_gamma = self.gamma
        
        # follow current policy for the rest of the episode
        for _ in range(self.max_episode_steps):
            # select action according to current policy
            policy_val = self.policy[curr_state]
            
            # Check if policy is stochastic (list of probs) or deterministic (int action)
            if isinstance(policy_val, list):
                action = np.random.choice(self.actions, p=policy_val)
            else:
                action = policy_val
                
            next_state, reward = self.env.get_transition(curr_state, action)
            
            # accumulate discounted return
            gain += current_gamma * reward
            current_gamma *= self.gamma
            
            curr_state = next_state
            
        return gain

    def policy_evaluation(self):
        """ 
        model-free policy evaluation:
        estimate q-value_{pi_k}(s,a) by averaging returns from multiple episodes starting from (s,a)
        """
        max_error = 0
        
        # exploring starts condition: loop through all state-action pairs
        for state in self.states:
            for action in self.actions:

                # collect sufficiently many episodes cause the law of large numbers
                returns = []
                for _ in range(self.episodes_per_eval):
                    gain = self._generate_episode_return(state, action)
                    returns.append(gain)
                
                # average return
                new_q = sum(returns) / len(returns)
                
                # update max error for convergence check
                old_q = self.q_value[(state, action)]
                max_error = max(max_error, abs(old_q - new_q))
                
                # save the q-value estimate
                self.q_value[(state, action)] = new_q
        
        # update value(s) for visualization: value(s) = max_{a} q-value(s,a)
        for s in self.states:
            self.value[s] = max([self.q_value[(s, a)] for a in self.actions])

        return max_error

    def policy_improvement(self):
        """ 
        policy improvement: update the policy according to the value(pi_{k}) obtained in policy evaluation
        pi_{k+1} = argmax_{pi} q-value
        return: whether the policy is stable or not (True/False)
        """
        policy_stable = True
        actions = range(self.env.action_space.n)
        
        for state in self.env.get_all_states():
            old_action = self.policy[state]
            
            # seek the action = argmax q-value under this state
            best_action = None
            max_q_value = float('-inf')
            
            for action in actions:
                q_value = self.q_value[(state, action)]
                
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            
            # update the policy by pi_{k+1} = argmax_{pi} q-value, this turns the policy from stochastic to deterministic
            self.policy[state] = best_action
            
            # check whether we already get the best policy
            if old_action != best_action:
                policy_stable = False
                
        return policy_stable

    def train(self):
        """ 
        apply the MC Basic algorithm
        loop is similar to policy iteration: MC policy evaluation -> policy improvement
        """
        iteration = 0
        while True:
            error = self.policy_evaluation()
            policy_stable = self.policy_improvement()
            
            print(f"iteration {iteration}: max q-value change = {error:.4f}")
            iteration += 1

            # convergence criteria: policy is stable and values haven't changed much
            if policy_stable and error < self.epsilon:
                print(f"MC Basic converged in {iteration} rounds")
                break

    def get_policy(self):
        return self.policy