import numpy.random as rng
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

states = ['a', 'b']

rewards = {}
rewards['a'] = -1.0
rewards['b'] = 1.0

sigma = 0.3

policy = {}
policy[('a','b')] = 0.0
policy[('a','a')] = 0.0
policy[('b','a')] = 0.0
policy[('b','b')] = 0.0

edges = {}

edges['a'] = {}
edges['a']['a'] = True
edges['a']['b'] = True
edges['b'] = {}
edges['b']['a'] = True
edges['b']['b'] = True

def sample(policy,start_state):
    probs = []
    edge_states = []

    for state in edges[start_state].keys():
        probs.append(policy[(start_state,state)])
        edge_states.append(state)

    probs = np.exp(np.array(probs))
    probs = probs/probs.sum()

    samp = rng.multinomial(n=1, pvals=probs).argmax()

    return edge_states[samp]

'''
On each iteration, keep track of the current state.  
Sample an epsilon value for each state.  
Also sample the opposite epsilon value.  

Select a new state using the modified policy.  

Then figure out an update vector, and remove the random perturbation.  

'''

num_steps = 20
reward_lst = []

use_antithetic = False

eps = {}

for iteration in range(0,200000):

    for trans in policy:
        if use_antithetic and iteration % 2 == 1:
            eps[trans] = -1.0 * eps[trans]
        else:
            eps[trans] = random.gauss(0,1)*sigma
        policy[trans] += eps[trans]

    current_state = 'a'

    reward_total = 0.0
    for step in range(0,num_steps):
        current_state = sample(policy,current_state)
        reward_total += rewards[current_state]

    reward_lst.append(reward_total)
    #UPDATE VALUES FOR EACH TRANSITION

    for trans in policy: 
        policy[trans] -= eps[trans]
        policy[trans] += 0.01 * reward_total * eps[trans]

    if iteration % 200 == 0:
        print iteration, reward_total
        print policy

 #   current_state = sample(policy, current_state)

  #  for state 


plt.plot(reward_lst)
plt.title("Rewards on the simple 2 state MDP using Evolution Strategies")
plt.xlabel("Iterations")
plt.ylabel("Reward")
plt.savefig('plots_rewards.png')



