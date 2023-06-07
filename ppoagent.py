import numpy as np
import pandas as pd
import gym
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras import backend as K

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.reset()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.data.columns) + 2,)
        )

    def reset(self):
        self.current_step = 0
        self.account_balance = 100000.0
        self.shares_held = 0
        self.prev_net_worth = 0.0
        self.net_worth = self.account_balance
        self.done = False
        return self._next_observation()

    def step(self, action):
        action_type, amount = action
        self._take_action(action_type, amount)
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        reward = self.net_worth - self.prev_net_worth
        return self._next_observation(), reward, self.done, {}

    def _next_observation(self):
        obs = np.concatenate(
            (
                self.data.iloc[self.current_step].values,
                np.array([self.account_balance, self.shares_held]),
            )
        )
        return obs

    def _take_action(self, action_type, amount):
        current_price = self.data.iloc[self.current_step]["Close/Last"]#.str.replace('$', '').astype(float)
        if action_type == 0:
            shares_to_buy = int(self.account_balance / current_price)
            cost = shares_to_buy * current_price

            self.account_balance -= cost
            self.shares_held += shares_to_buy
        else:
            #shares_to_sell = int(self.shares_held * amount)
            sale = amount * current_price
            self.account_balance += sale
            self.shares_held -= amount
        self.prev_net_worth = self.net_worth
        self.net_worth = self.account_balance + self.shares_held * current_price

data = pd.read_csv("data/AAPL.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)
data.drop(["Volume", "Open", "High", "Low"], axis=1, inplace=True)

epsilon = 1e-8
data['Close/Last'] = data['Close/Last'].str.replace('$', '').astype(float)
# data = (data - data.mean()) / (data.std() + epsilon)
# print(f'JOOOO{data}')

state_dim = len(data.columns)
action_dim = 3
clip_ratio = 0.2
critic_coef = 0.5
learning_rate = 0.001

def build_actor_critic():
    inputs = Input(shape=(3,))
    dense = Dense(64, activation='relu')(inputs)
    actor_output = Dense(action_dim, activation='softmax')(dense)
    critic_output = Dense(1)(dense)
    actor_model = Model(inputs=inputs, outputs=actor_output)
    critic_model = Model(inputs=inputs, outputs=critic_output)
    return actor_model, critic_model

actor_model, critic_model = build_actor_critic()

def ppo_loss(old_policy_probs, advantages, clipped_values):
    def loss(y_true, y_pred):
        policy_ratio = K.exp(K.log(y_pred + 1e-10) - K.log(y_true + 1e-10))
        unclipped_loss = policy_ratio * advantages
        clipped_loss = K.clip(policy_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -K.mean(K.minimum(unclipped_loss, clipped_loss))
        critic_loss = K.mean(K.square(y_true - y_pred))
        total_loss = actor_loss + critic_coef * critic_loss
        return total_loss
    return loss

actor_model.compile(optimizer=Adam(learning_rate=learning_rate), loss=ppo_loss)
critic_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")

env = StockTradingEnv(data)

num_episodes = 10000
for episode in range(num_episodes):

    obs = env.reset()
    done = False

    while not done:
        states = []
        actions = []
        rewards = []
        dones = []

        while not done:
            states.append(obs)
            obs_tensor = tf.convert_to_tensor(np.expand_dims(obs, axis=0), dtype=tf.float32)

            action_probs = actor_model.predict(obs_tensor)[0]# tf.nn.softmax(np.nan_to_num(actor_model.predict(obs_tensor)[0]))
            if np.isnan(action_probs).any():
                action = (np.random.choice(np.arange(action_dim)), 1)
            else:
                action = (np.random.choice(np.arange(action_dim), p=action_probs), 1)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            dones.append(done)
            if done:
                break
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        running_return = 0
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * (1 - dones[t])
            running_tderror = rewards[t] + gamma * critic_model.predict(np.expand_dims(states[t], axis=0))[0][0] * (
                        1 - dones[t]) - critic_model.predict(np.expand_dims(states[t - 1], axis=0))[0][0]
            running_advantage = running_tderror + gamma * gae_lambda * running_advantage * (1 - dones[t])
            returns[t] = running_return
            advantages[t] = running_advantage
        actor_model.fit(
            states,
            actions,
            sample_weight=advantages,
            epochs=1,
            verbose=0
        )
        critic_model.fit(
            states,
            returns,
            epochs=1,
            verbose=0
        )
    if episode % 100 == 0:
        print(episode)

obs = env.reset()
done = False
rewards = []

while not done:
 #   print(obs)
    obs_array = np.expand_dims(obs, axis=0).astype(np.float32)
  #  print(obs_array)
    action_probs = actor_model.predict(obs_array)[0]
    action = np.argmax(action_probs)
  #  print(f'action:{action}')
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.show()
