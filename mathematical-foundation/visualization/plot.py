import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Visualizer:
    @staticmethod
    def plot_grid_policy(env, policy, values):
        """
        Visualizes the grid world with the policy and state values.
        Supports both deterministic policy (int) and stochastic policy (list of probs).
        """
        # configuration: pastel color palette
        COLOR_NORMAL = '#FFFFFF'
        COLOR_FORBIDDEN = '#FFF9C4'
        COLOR_GOAL = '#E1F5FE'
        COLOR_TEXT = '#546E7A'
        COLOR_ARROW = '#FFAB91'
        BORDER_COLOR = '#B0BEC5'
        
        rows, cols = env.height, env.width
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Setup plot limits and hide axes
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_aspect('equal')
        ax.axis('off')

        start_state = env._agent_location 

        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                center_x, center_y = c + 0.5, r + 0.5
                
                # determine background color
                face_color = COLOR_NORMAL
                if state == env.target_state or state == start_state:
                    face_color = COLOR_GOAL
                elif state in env.forbidden_states:
                    face_color = COLOR_FORBIDDEN
                
                # draw cell: background & border
                rect = patches.Rectangle((c, r), 1, 1, 
                                         facecolor=face_color, 
                                         edgecolor=BORDER_COLOR, 
                                         linewidth=2)
                ax.add_patch(rect)
                
                # draw state value
                if state in values:
                    ax.text(center_x, center_y + 0.25, f"{values[state]:.2f}", 
                            ha='center', va='center', 
                            fontsize=10, color=COLOR_TEXT, fontweight='normal')

                # Draw Policy (Actions)
                policy_data = policy.get(state)
                
                # Check if policy is stochastic (list) or deterministic (int/None)
                actions_to_draw = []
                
                if policy_data is None:
                    continue
                elif isinstance(policy_data, (list, np.ndarray)):
                    # Stochastic: policy_data is [prob_0, prob_1, ...]
                    for action_idx, prob in enumerate(policy_data):
                        if prob > 0.01: # Only draw actions with meaningful probability
                            actions_to_draw.append((action_idx, prob))
                else:
                    # Deterministic: policy_data is action_idx
                    actions_to_draw.append((policy_data, 1.0))

                for action, prob in actions_to_draw:
                    # Scale size by probability (min size for visibility)
                    scale = 0.5 + 0.5 * prob 
                    alpha = 0.3 + 0.7 * prob # Transparency based on prob
                    
                    # action 4: stay (draw circle)
                    if action == 4:
                        radius = 0.1 * scale
                        circle = patches.Circle((center_x, center_y - 0.1), radius, 
                                                facecolor=COLOR_ARROW, edgecolor=COLOR_ARROW, 
                                                linewidth=2, alpha=alpha)
                        ax.add_patch(circle)
                    
                    # actions 0-3: move (draw arrow)
                    else:
                        dx, dy = 0, 0
                        if action == 0: dy = -0.2   # Up
                        elif action == 1: dy = 0.2  # Down
                        elif action == 2: dx = -0.2 # Left
                        elif action == 3: dx = 0.2  # Right
                        
                        if dx != 0 or dy != 0:
                            # Scale arrow width and head based on probability
                            width = 0.015 * scale
                            h_width = 0.08 * scale
                            h_length = 0.08 * scale
                            
                            ax.arrow(center_x - dx * 0.5, center_y - dy * 0.5 - 0.1, dx, dy, 
                                     head_width=h_width, head_length=h_length, 
                                     fc=COLOR_ARROW, ec=COLOR_ARROW, 
                                     width=width, alpha=alpha)

        ax.set_title("Epsilon-Greedy Policy & State Values", fontsize=15, color='#455A64', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()