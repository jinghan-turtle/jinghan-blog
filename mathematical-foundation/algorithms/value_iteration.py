class ValueIteration:
    def __init__(self, env, gamma=0.9, epsilon=1e-8):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.value = {s: 0.0 for s in self.env.get_all_states()}
        
    def train(self):
        """ apply the value iteration algorithm """
        actions = range(self.env.action_space.n) # 0, 1, 2, 3, 4
        
        iteration = 0
        while True:
            error = 0
            # the process below can be concluded that construct the q-table and selcet a_k*(s) = argmax_a q_k(s,a)
            for state in self.env.get_all_states():
                new_v_candidates = []
                old_value = self.value[state]
                
                # given the state, for every action, calculate the q-value
                for action in actions:
                    next_state, reward = self.env.get_transition(state, action)
                    q_value = reward + self.gamma * self.value[next_state]
                    new_v_candidates.append(q_value)
                
                # select the action according to the max q-value
                self.value[state] = max(new_v_candidates)

                # Iteration stop criterion: the infinite norm of state values ||v_{k+1}-v_{k}|| < epsilon
                error = max(error, abs(old_value - self.value[state]))
            
            iteration += 1
            if error < self.epsilon:
                print(f"The value iteration converges in the {iteration} round.")
                break

    def get_policy(self):
        policy = {}
        actions = range(self.env.action_space.n)
        
        # construct the q-table again
        for state in self.env.get_all_states():
            best_action = None
            max_q_value = float('-inf')
            
            # update the policy as pi_{k+1}(a|s)=1, if a=a_k*, otherwise pi_{k+1}(a|s)=0
            for action in actions:
                next_state, reward = self.env.get_transition(state, action)
                q_value = reward + self.gamma * self.value[next_state]
                
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            
            policy[state] = best_action

        return policy