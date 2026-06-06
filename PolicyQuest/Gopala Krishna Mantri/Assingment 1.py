import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def evaluate_policy(env, policy, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            a = policy[s] 

            v_new = 0
            for prob, next_state, reward, done in env.P[s][a]:
                v_new += prob * (reward + gamma * V[next_state] * (not done))
            
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break
            
    return V

def improve_policy(env, V, gamma=0.99):

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    new_policy = np.zeros(n_states, dtype=int)
    
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_state] * (not done))
                
        new_policy[s] = np.argmax(q_values)
        
    return new_policy

def policy_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    
    policy = np.zeros(n_states, dtype=int) 
    
    while True:

        V = evaluate_policy(env, policy, gamma, theta)
        
        new_policy = improve_policy(env, V, gamma)

        if np.array_equal(policy, new_policy):
            break
            
        policy = new_policy
        
    return policy, V

def plot_policy_on_frozen_lake(env, policy, title="FrozenLake policy"):
	desc = np.asarray(env.unwrapped.desc, dtype=str)
	policy_grid = np.asarray(policy).reshape(desc.shape)
	arrows = np.array(["<", "v", ">", "^"])
	colors = {
		"S": "#9be7a1",
		"F": "#dceefb",
		"H": "#3a3a3a",
		"G": "#ffd54f",
	}

	fig, ax = plt.subplots(figsize=(8, 8))
	for r in range(desc.shape[0]):
		for c in range(desc.shape[1]):
			tile = desc[r, c]
			rect = plt.Rectangle((c, desc.shape[0] - 1 - r), 1, 1, facecolor=colors[tile], edgecolor="black", linewidth=1.5)
			ax.add_patch(rect)

			if tile == "H":
				label = "H"
			elif tile == "G":
				label = "G"
			elif tile == "S":
				label = f"S{arrows[policy_grid[r, c]]}"
			else:
				label = arrows[policy_grid[r, c]]

			ax.text(c + 0.5, desc.shape[0] - 1 - r + 0.5, label, ha="center", va="center", fontsize=16, fontweight="bold", color="black")

	ax.set_xlim(0, desc.shape[1])
	ax.set_ylim(0, desc.shape[0])
	ax.set_xticks(np.arange(desc.shape[1] + 1))
	ax.set_yticks(np.arange(desc.shape[0] + 1))
	ax.grid(True, color="black", linewidth=1.0)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_aspect("equal")
	ax.set_title(title)
	plt.tight_layout()
	plt.show()

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)

optimal_policy, optimal_values = policy_iteration(env.unwrapped)



plot_policy_on_frozen_lake(env, optimal_policy, title="Policy Iteration on FrozenLake 8x8")
#print(optimal_policy.reshape((8, 8)))
