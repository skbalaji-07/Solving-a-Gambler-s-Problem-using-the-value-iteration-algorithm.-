# Solving Gambler’s Problem using Value Iteration

## Date :

## Aim
To implement the Value Iteration Algorithm for solving the Gambler’s Problem in Reinforcement Learning and determine the optimal policy that maximizes the probability of reaching the goal.

## Algorithm

1. Initialize the environment:
   - Define goal amount.
   - Define probability of winning.
   - Initialize value function and policy.

2. Initialize state values:
   - Set value of goal state to 1.
   - Set all other state values to 0.

3. Perform Value Iteration:
   - For each state:
     - Evaluate all possible stakes.
     - Compute expected returns for each action.
     - Update the value function with the maximum expected return.

4. Repeat the iteration until the value function converges.

5. Derive the optimal policy:
   - Select the action with the highest value for each state.

6. Display the optimal value function and policy.

---

## Program

```
#Solving a Gambler’s Problem using the value iteration algorithm. 

import numpy as np

GOAL = 100

# Probability of winning
p_h = 0.4

# Discount factor
gamma = 1.0

# State values
V = np.zeros(GOAL + 1)

# Goal state
V[GOAL] = 1

# Policy
policy = np.zeros(GOAL + 1, dtype=int)

theta = 0.0001


# -------------------------------
# VALUE ITERATION
# Bellman Optimality Update
# -------------------------------

while True:

    delta = 0

    for s in range(1, GOAL):

        old_value = V[s]

        action_returns = []

        # Possible actions
        actions = range(1, min(s, GOAL - s) + 1)

        for a in actions:

            win_state = s + a
            lose_state = s - a

            reward = 0

            if win_state == GOAL:
                reward = 1

            # V(s) = max_a [ p_h(r+γV(s+a))
            #               +(1-p_h)(r+γV(s-a)) ]

            value = (
                p_h * (reward + gamma * V[win_state])
                + (1 - p_h) * (reward + gamma * V[lose_state])
            )

            action_returns.append(value)

        V[s] = max(action_returns)

        delta = max(delta, abs(old_value - V[s]))

    if delta < theta:
        break


# -------------------------------
# EXTRACT OPTIMAL POLICY
# -------------------------------

for s in range(1, GOAL):

    actions = range(1, min(s, GOAL - s) + 1)

    best_action = 0
    best_value = -1

    for a in actions:

        win_state = s + a
        lose_state = s - a

        reward = 0

        if win_state == GOAL:
            reward = 1

        val = (
            p_h * (reward + gamma * V[win_state])
            + (1 - p_h) * (reward + gamma * V[lose_state])
        )

        if val > best_value:
            best_value = val
            best_action = a

    policy[s] = best_action


print("Optimal State Values:\n")
print(V)

print("\nOptimal Betting Policy:\n")
print(policy)
        
```

---

## Output

<img width="749" height="551" alt="image" src="https://github.com/user-attachments/assets/e6497021-1202-458b-b7a8-9f185a3b7975" />


## Result

The Value Iteration Algorithm was successfully implemented for the Gambler’s Problem. The optimal value function and policy were obtained, maximizing the probability of reaching the goal state through optimal betting decisions.
