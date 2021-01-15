import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle

pos_space = np.linspace(-1.2, 0.6, 20)
vel_space = np.linspace(-0.07, 0.07, 20)

def get_state(observation):
    pos, vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)

    return (pos_bin, vel_bin)

def max_action(Q, state, action=[0, 1, 2]):
    values = np.array([Q[state,a] for a in action])
    action = np.argmax(values)

    return action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    n_games = 25000
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos,vel))

    Q = {}
    for state in states:
        for action in [0, 1, 2]:
            Q[state, action] = 0

    
    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs)
        if i % 100 == 0 and i > 0:
            print('episode', i, 'score', score, 'epsilon %.3f' % eps)
        score = 0

        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < eps \
                else max_action(Q, state)
            obs_, reward, done, info = env.step(action)
            state_ = get_state(obs_)
            score += reward
            action_ = max_action(Q, state_)
            Q[state, action] = Q[state, action] + \
                alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
            state = state_
        
        total_rewards[i] = score
        eps = eps - 2/n_games if eps > 0.01 else 0.01

    filename = 'new_Q_mtnc'
    outfile = open(filename, 'wb')
    pickle.dump(Q, outfile)
    outfile.close()

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t - 50):(t + 1)])
    plt.plot(mean_rewards)
    plt.savefig('mountaincar.png')