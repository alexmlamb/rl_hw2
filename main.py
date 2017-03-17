import numpy.random as rng
import numpy as np


states = ['a', 'b']

rewards = {}
rewards['a'] = -1.0
rewards['b'] = 1.0

sigma = 0.1

policy = {}
policy[('a','b')] = 0.0
policy[('a','a')] = 0.0
policy[('b','a')] = 0.0
policy[('b','b')] = 0.0

allowed_transitions = {}
allowed_transitions[('a','a')] = True
allowed_transitions[('a','b')] = True
allowed_transitions[('b','a')] = True
allowed_transitions[('b','b')] = True

def sample(policy,start_state):
    probs = []

    for state in states:
        probs.append(policy[(start_state,state)])

    probs = np.exp(np.array(probs))
    probs = probs/probs.sum()

    samp = rng.multinomial(n=1, pvals=probs).argmax()

    return states[samp]

'''
On each iteration, keep track of the current state.  
Sample an epsilon value for each state.  
Also sample the opposite epsilon value.  

Select a new state using the current policy.  

'''
current_state = 'a'

eps = {}

for iteration in range(0,200):
    

    current_state = sample(policy, current_state)

    for state 




