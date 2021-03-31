# Mountaincar-v0
Mountaincar is a simulation featuring a car on a one-dimensional track, positioned between two “mountains”. The. Goal is to drive up the mountain on the right; however, the car’s engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
This was solved using Q-learning; a model-free reinforcement learning algorithm which allows the agent to derive an optimal policy directly from interactions with the environment without building a model.
Agent found a reliable and efficient policy for completing the task, reaching peak performance after approximately 23,000 games. This process was visualized by plotting the reward function as a function of number of games played.
