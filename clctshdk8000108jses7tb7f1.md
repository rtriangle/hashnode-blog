# On-policy and Off-policy algorithms in Reinforcement Learning

Reinforcement learning (RL) is a type of machine learning that enables agents to learn how to make decisions by interacting with an environment. The agent learns to maximize a reward signal by trying different actions and observing the resulting rewards. RL has been used in a wide range of applications, including robotics, game-playing, navigation, control, and recommendation systems.

## On-policy and Off-policy RL

There are several types of RL methods, each with its own advantages and disadvantages depending on the specific problem and environment.

On-policy methods involve learning from the current policy that the agent is following. These methods are well-suited for tasks where the agent needs to learn how to improve its current behavior, based on the feedback it receives from the environment. On-policy methods will be more stable because they will not deviate too much from the current policy, making them less likely to experience a performance drop. However, it will take longer to reach the optimal solution if the current policy is not already optimal. In the context of the Exploration vs Exploitation dilemma, on-policy methods solve it by introducing randomness in a policy which means that non-greedy actions are selected with some probability.

Off-policy methods, on the other hand, involve learning from actions taken by a different policy than the one the agent is currently following. These methods are well-suited for tasks where the agent needs to learn a good policy based on historical data, regardless of the actions it's currently taking. Off-policy methods can learn from more diverse experiences, making it more likely to find the optimal solution quickly. However, there is a chance that it will deviate too much from the current policy, and this could lead to poor performance. Off-policy methods provide a different solution to the exploration vs. exploitation problem. They include 2 policies: behavior policy and target policy. The behavioral policy is used for exploration and episode generation, and the target or goal policy is used for function estimation and improvement.

[***On-policy methods attempt to evaluate or improve the policy that is used to make decisions, whereas off-policy methods evaluate or improve a policy different from that used to generate the data.***](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

## On-policy algorithms examples

There are several on-policy reinforcement learning algorithms. Here are some examples of them:

1. **SARSA (State-Action-Reward-State-Action)** is an on-policy version of Q-learning. It updates the action-value function based on the current policy, rather than the optimal policy.
    
2. **REINFORCE** - a policy gradient algorithm that updates the policy directly. It uses a Monte Carlo approach to estimate the gradient of the expected total reward with respect to the policy parameters.
    
3. **A2C (Advantage Actor-Critic)** is an actor-critic algorithm that uses the policy gradient method to update both the actor and the critic. It's an on-policy method because it uses the current policy to generate experiences to update the actor and critic.
    
4. **TRPO (Trust Region Policy Optimization)** - the optimization-based algorithm that updates the policy by finding the step that maximizes an objective function defined using both the new and old policies, making it on-policy.
    

We will focus mainly on **SARSA** and **REINFORCE** today.

### SARSA

**SARSA (State-Action-Reward-State-Action)** is an on-policy algorithm that updates the action-value function (Q-function) based on the current policy. The Q-function represents the expected future reward for taking a certain action in a given state and following the current policy thereafter. The main formula used to update the Q-function in SARSA is the following:

$$Q(s,a) = Q(s,a) + α * (r + γ * Q(s',a') - Q(s,a))$$

In this equation, Q(s,a) is the current estimate of the action-value function for state s and action a, α is the learning rate, r is the immediate reward received after taking action a in state s, γ is the discount factor, s' is the next state, and a' is the next action chosen according to the current policy. It's important to note that SARSA will always use the current policy to choose the next action, this could make it converge slower to the optimal policy compared to off-policy methods, but it will be more stable and less likely to experience a performance drop. Also depending on the problem and the environment, the step-size parameter α and discount factor γ, will have to be adjusted accordingly to get the best performance.

### REINFORCE

**REINFORCE** is a policy gradient algorithm that updates the policy directly. It is an on-policy algorithm that uses a Monte Carlo approach to estimate the gradient of the expected total reward with respect to the policy parameters. The main formula used to update the policy in REINFORCE is the following:

1. Collect a set of trajectories {(s\_t,a\_t,r\_t)} by following the current policy π.
    
2. For each time step t, compute the gradient of the policy log-likelihood with respect to the policy parameters θ:
    
    $$∇_θ log π(a_t|s_t; θ)$$
    
3. Use these gradients to update the policy parameters θ using a chosen optimization algorithm, such as stochastic gradient ascent:
    
    $$θ \leftarrow θ + α * ∇_θ J(θ)$$
    

Where:

* J(θ) is the expected cumulative reward of the policy.
    
* α is the learning rate.
    

The REINFORCE method calculates the gradient of the expected total reward with respect to the policy parameters and updates the policy parameters in the direction of increasing the expected reward.

## Off-policy examples and applications

There are several examples of off-policy reinforcement learning algorithms. Here are a few:

1. **Q-learning** - algorithm updates the action-value function based on the maximum expected future reward, which is computed using the optimal policy. Because it's using a different policy (the optimal policy) than the one the agent is currently following, it is considered an off-policy algorithm.
    
2. **DQN (Deep Q-Network)** is an extension of Q-learning that uses neural networks to approximate the action-value function. DQN is able to handle high-dimensional and continuous state spaces and is considered off-policy because it still uses the Q-learning approach.
    
3. **DDPG (Deep Deterministic Policy Gradient)** is a variant of the actor-critic method that uses off-policy data to update both the actor and the critic. It uses a replay buffer to store experiences from a behavior policy and learns from them.
    
4. **TD3 (Twin Delayed DDPG)** - an extension of DDPG that improves the stability of the algorithm by using two separate critics and delaying the policy update. It's also considered off-policy as it uses a replay buffer and the experiences are collected from a behavior policy.
    

We will dive deeper into **Q-learning** and **DQN** in this article.

### Q-Learning

Q-learning is an off-policy algorithm that updates the action-value function (Q-function) based on the maximum expected future reward, which is computed using the optimal policy. The Q-function represents the expected future reward for taking a certain action in a given state and following the optimal policy thereafter.

The main formula used to update the Q-function in Q-learning is the following:

Q(s,a) = Q(s,a) + α *(r + γ* max\_a'(Q(s',a')) - Q(s,a))

In this equation, Q(s,a) is the current estimate of the action-value function for state s and action a, α is the learning rate, r is the immediate reward received after taking action a in state s, γ is the discount factor, s' is the next state, and max\_a'(Q(s',a')) is the maximum expected future reward for all possible actions in the next state s' computed using the optimal policy.

### DQN (Deep Q-Network)

DQN (Deep Q-Network) is an extension of Q-learning that uses neural networks to approximate the action-value function. DQN is able to handle high-dimensional and continuous state spaces. It uses a replay buffer to store experiences, and then randomly sample from this buffer to train the Q-Network.

The main formula used to update the Q-Network in DQN is the following:

$$L = (y_j - Q(s,a;θ))^2$$

where y\_j is the target value, calculated as:

$$y_j = r + γ * max_a'(Q(s',a';θ_j))$$

In this equation, Q(s,a;θ) is the output of the Q-network for the state s and action a, and θ are the network parameters. L is the loss function, typically mean-squared error is used. The target value y\_j is the Q-value of the next state s' plus the immediate reward r, discounted by γ, the discount factor. The parameters θ\_j are the network parameters obtained after j-th iteration.

It's important to note that DQN uses an off-policy approach, which means it uses experience replay, a target network, and fixed Q-targets in order to stabilize the learning process. Furthermore, DQN has been extended to many different variations like DDQN (Double DQN) which uses two networks to improve stability and more, and generally, many advanced techniques have been used along with DQN to improve its performance.

### Summary

On-policy and off-policy are two types of reinforcement learning algorithms that differ in how they use the data they collect. On-policy algorithms are more sample-efficient as they update the policy as soon as new data is collected. However, they are more sensitive to the initial policy and may get stuck in a suboptimal policy. Off-policy algorithms are less sensitive to the initial policy, but they require more data to converge and can be less sample-efficient.