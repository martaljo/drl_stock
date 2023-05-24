import numpy as np
import pandas as pd
from random import randint
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_rate=0.001):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table =
    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return randint(0, self.action_size - 1)
        else:
            return np.argmax(self.get_q(state))

    def get_q(self, state):
        return self.q_table.get(state, np.zeros(self.action_size))
      
    def update_q_table(self, state, action, reward, next_state):
        current_q = self.get_q(state)[action]
        max_next_q = np.max(self.get_q(next_state))
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        new_q_values = self.get_q(state)
        new_q_values[action] = new_q
        self.q_table[state] = new_q_values

    def decay_exploration_rate(self):
        self.exploration_rate = self.exploration_rate * (1 - self.exploration_decay_rate)


class TradingEnvironment:
    def __init__(self, data):
        self.data = data.values
        self.total_steps = len(data)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        return tuple(self.data[self.current_step])

    def take_action(self, action, current_price, next_price):
        if action == 0: 
            return next_price - current_price
        elif action == 1:
            return current_price - next_price
        else:
            return 0

    def step(self, action):
        current_price = self.data[self.current_step][1]
        self.current_step += 1
        done = self.current_step >= self.total_steps
        if done:
            next_state = tuple(self.data[self.current_step - 1])
            reward = 0
        else:
            next_price = self.data[self.current_step][1]
            reward = self.take_action(action, current_price, next_price)
            next_state = self.get_state()
        return next_state, reward, done

csv_file = 'drl_stock/data/AAPL.csv'
data = pd.read_csv(csv_file)
data['Close/Last'] = data['Close/Last'].str.replace('$', '').astype(float)
data = data.iloc[:, 1:]  # Excluding Date

action_size = 3
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
total_episodes = 2000

env = TradingEnvironment(data)
agent = QLearningAgent(action_size, learning_rate, discount_factor, exploration_rate)

total_rewards = []

for episode in range(total_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        total_reward += reward
        state = next_state

    agent.decay_exploration_rate()
    total_rewards.append(total_reward)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate}")

plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()