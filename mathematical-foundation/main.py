from envs import GridWorld
from algorithms import ValueIteration, PolicyIteration, TruncatedPolicyIteration, MCBasic, MCExploringStarts, MCEpsilonGreedy
from visualization import Visualizer

def run_pipeline():
    # create the grid world
    env = GridWorld()
    
    # train the agent
    agent = MCEpsilonGreedy(env)
    agent.train()
    
    # get the optimal policy
    optimal_policy = agent.get_policy()
    
    # visualize the policy and state value
    Visualizer.plot_grid_policy(env, optimal_policy, agent.value)

if __name__ == "__main__":
    run_pipeline()