class PolicyIteration:
    def __init__(self, env, gamma=0.9, epsilon=1e-8):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # initialize state value and policy
        self.value = {s: 0.0 for s in self.env.get_all_states()}
        self.policy = {s: 0 for s in self.env.get_all_states()}

    def policy_evaluation(self):
        """ 
        policy evaluation: solve the Bellman equation to get the value(state) corresponding to pi_{k}
        this leverages the property that f(v) on the right-hand side of the Bellman equation is a contraction mapping
        """
        while True:
            error = 0
            for state in self.env.get_all_states():
                old_value = self.value[state]
                
                # get needed elements to calculate the Bellman equation
                action = self.policy[state]
                next_state, reward = self.env.get_transition(state, action)
                
                # repeat calculating the right-hand side of the Bellman equation to update the value(state)
                new_value = reward + self.gamma * self.value[next_state]
                self.value[state] = new_value

                # the iteration error is defined as ||v(pi_{k})^{j+1}-v(pi_{k})^{j}||
                error = max(error, abs(old_value - new_value))
            
            if error < self.epsilon:
                break

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
                next_state, reward = self.env.get_transition(state, action)
                # refer to the element-wise expansion of the policy iteration algorithm
                q_value = reward + self.gamma * self.value[next_state]
                
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            
            # update the policy by pi_{k+1} = argmax_{pi} q-value
            self.policy[state] = best_action
            
            # check whether we already get the best policy
            if old_action != best_action:
                policy_stable = False
                
        return policy_stable

    def train(self):
        """ 
        apply the policy iteration algorithm 
        the train process is evaluation -> improvement -> check stability
        """
        iteration = 0
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            
            iteration += 1
            if policy_stable:
                print(f"The policy iteration converges in the {iteration} round.")
                break

    def get_policy(self):
        return self.policy