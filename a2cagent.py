import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import gym
from elegantrl.agents.AgentA2C import AgentA2C
import matplotlib.pyplot as plt

TICKERS = ["AAPL", "MSFT", "NFLX", "QCOM", "TSLA"]

BUY = 0
SELL = 1
SELL_ALL = -1

class TradingEnvironment:

    def __init__(self, data):
        self.data = data.values
        self.total_steps = len(data)
        self.current_step = 0
        # self.inventory = 0 # Track of agent's holdings
        self.inventory = {"AAPL": (0, 0.0)}
        self.capital = 10000  # Initial capital, adjust to your preference
        self.buy_price = 0

    def reset(self):
        self.current_step = 0
        # self.inventory = 0
        self.inventory = {"AAPL": (0, 0.0)}
        self.capital = 10000
        self.buy_price = 0
        return self.get_state()

    def get_state(self):
        return tuple(self.data[self.current_step])
        # return (self.capital, self.inventory)

    def get_stock_value(self):
        return self.inventory["AAPL"][0] * self.inventory["AAPL"][1]

    def sell_stocks(self, ticker, quantity, current_price):
        curr_inventory = self.inventory[ticker]
        sell_amount = curr_inventory[1] * quantity
        
        self.inventory[ticker] = (curr_inventory - quantity, curr_inventory[1])
        self.capital += sell_amount

        return sell_amount

    def buy_stocks(self, ticker, quantity, current_price):
        curr_ticker_inv = self.inventory[ticker]
        current_amount = curr_ticker_inv[0] * curr_ticker_inv[1]
        buy_amount = current_price * quantity

        new_quantity = curr_ticker_inv[0] + quantity
        new_mean_price = (current_amount + buy_amount) / new_quantity

        self.inventory[ticker] = (new_quantity, new_mean_price)
        self.capital -= buy_amount

        return buy_amount

    def take_action(self, action, current_price, next_price):
        reward = 0
        hold = False
        if action == BUY:  # Buy
            if self.capital > current_price:
                self.buy_stocks("AAPL", 1, current_price)
            else:
                hold=True
        elif action == SELL:  # Sell
            if self.inventory["AAPL"][0] > 0:
                current = self.inventory["AAPL"]
                reward = (next_price - current[1])
                self.capital += next_price
                self.inventory["AAPL"] = (current[0] - 1, current[1])

                # self.sell_stocks("AAPL", 1, current_price)
            else:
                hold = True
        elif action == SELL_ALL:
            current = self.inventory["AAPL"]
            reward = (next_price - current[1]) * current[0]
            self.capital += current[0] * next_price
            self.inventory["AAPL"] = (0, 0.0)
            
            # self.sell_stocks("AAPL", current[0], current_price)
        else:  # Hold
            if self.inventory["AAPL"][0] > 0:
                reward = (current_price - self.inventory["AAPL"][1]) * self.inventory["AAPL"][0]

        if hold:
            if self.inventory["AAPL"][0] > 0:
                reward = (current_price - self.inventory["AAPL"][1]) * self.inventory["AAPL"][0]
        return reward

    def step(self, action):
        current_price = self.data[-self.current_step][0]
        self.current_step += 1
        done = self.current_step >= self.total_steps
        if done:
            next_state = tuple(self.data[self.current_step - 1])
            reward = 0
        elif self.current_step == self.total_steps -1:
            next_price = self.data[-self.current_step - 1][2]
            reward = self.take_action(SELL_ALL, current_price, next_price)
            next_state = self.get_state()
        else:
            next_price = self.data[-self.current_step - 1][2]
            reward = self.take_action(action, current_price, next_price)
            next_state = self.get_state()
        return next_state, reward, done



# class TradingEnvironment:

#     def __init__(self, data):
#         self.data = data
#         self.total_steps = len(data[TICKERS[0]])
#         self.current_step = 0
#         # self.inventory = 0 # Track of agent's holdings
#         self.inventory = {"AAPL": (0, 0.0), "MSFT": (0, 0.0), "NFLX": (0, 0.0), "QCOM": (0,0.0), "TSLA": (0, 0.0)}
#         self.capital = 10000  # Initial capital, adjust to your preference
#         self.buy_price = 0

#     def reset(self):
#         self.current_step = 0
#         # self.inventory = 0
#         self.inventory = {"AAPL": (0, 0.0), "MSFT": (0, 0.0), "NFLX": (0, 0.0), "QCOM": (0,0.0), "TSLA": (0, 0.0)}
#         self.capital = 10000
#         self.buy_price = 0
#         return self.get_state()

#     def get_state(self):
#         state_info = []

#         for stock in self.data.keys():
#             state_info.append(tuple(self.data[stock][-self.current_step - 1]))

#         return tuple(state_info)
#         # return (self.capital, self.inventory)
#         # come back to this

#     def get_stock_value(self):
#         total = 0.0
#         for stock in TICKERS:
#             total += self.inventory[stock][0] * self.inventory[stock][1]
#         return total

#     def sell_stocks(self, ticker, quantity, current_price):
#         curr_inventory = self.inventory[ticker]
#         sell_amount = curr_inventory[1] * quantity
        
#         self.inventory[ticker] = (curr_inventory - quantity, curr_inventory[1])
#         self.capital += sell_amount

#         return sell_amount

#     def buy_stocks(self, ticker, quantity, current_price):
#         curr_ticker_inv = self.inventory[ticker]
#         current_amount = curr_ticker_inv[0] * curr_ticker_inv[1]
#         buy_amount = current_price * quantity

#         new_quantity = curr_ticker_inv[0] + quantity
#         new_mean_price = (current_amount + buy_amount) / new_quantity

#         self.inventory[ticker] = (new_quantity, new_mean_price)
#         self.capital -= buy_amount

#         return buy_amount

#     def take_action(self, action, current_price, next_price, ticker):
#         reward = 0
#         hold = False
#         if action[0] == BUY:  # Buy
#             if self.capital > current_price:
#                 self.buy_stocks(ticker, 1, current_price)
#             else:
#                 hold=True
#         elif action[0] == SELL:  # Sell
#             if self.inventory[ticker][0] > 0:
#                 current = self.inventory[ticker]
#                 reward = (next_price - current[1])
#                 self.capital += next_price
#                 self.inventory[ticker] = (current[0] - 1, current[1])

#                 # self.sell_stocks("AAPL", 1, current_price)
#             else:
#                 hold = True
#         elif action[0] == SELL_ALL:
#             reward = 0.0
#             self.capital += self.get_stock_value()

#             for ticks in TICKERS: 
#                 current = self.inventory[ticks]
#                 next_price = self.data[ticks][-self.current_step - 1][2]
#                 reward += (next_price - current[1]) * current[0]
#                 self.inventory[ticks] = (0, 0.0)
#             # current = self.inventory[ticker]
#             # reward = (next_price - current[1]) * current[0]
#             # self.capital += current[0] * next_price
#             # self.inventory[ticker] = (0, 0.0)

#             # self.sell_stocks("AAPL", current[0], current_price)
#         else:  # Hold
#             if self.inventory[ticker][0] > 0:
#                 reward = (current_price - self.inventory[ticker][1]) * self.inventory[ticker][0]

#         if hold:
#             if self.inventory["AAPL"][0] > 0:
#                 reward = (current_price - self.inventory["AAPL"][1]) * self.inventory[ticker][0]
#         return reward

#     def step(self, action):
#         current_price = self.data[action[1]][-self.current_step][0]
#         self.current_step += 1
#         done = self.current_step >= self.total_steps
#         if done:
#             stattte = []
#             for stock in self.data.keys():
#                 stattte.append(tuple(self.data[stock][0]))
#             next_state = tuple(stattte)
#             reward = 0
#         elif self.current_step == self.total_steps -1:
#             next_price = self.data[action[1]][-self.current_step - 1][2]
#             reward = self.take_action((SELL_ALL, None), current_price, next_price, action[1])
#             next_state = self.get_state()
#         else:
#             next_price = self.data[action[1]][-self.current_step - 1][2]
#             reward = self.take_action(action, current_price, next_price, action[1])
#             next_state = self.get_state()
#         return next_state, reward, done

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        self.fc_pi = nn.Linear(32, action_dim)
        self.fc_v = nn.Linear(32, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        return pi, v

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.to(self.device)
        self.prev_action = 0

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        pi, _ = self.actor_critic(state)

        action_probs = F.softmax(pi, dim=0)
        action = np.random.choice(range(action_probs.shape[0]), p=action_probs.detach().cpu().numpy())
        self.prev_action = action
        return action

    def learn(self, state, action, reward, next_state, discount_factor):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        # # done = torch.FloatTensor(done).to(self.device)

        _, value = self.actor_critic(state)
        _, next_value = self.actor_critic(next_state)

        delta = reward + 0.99 * (1 - discount_factor) * next_value - value
        temp = self.actor_critic(state)[0]

        actor_loss = -torch.log(temp[action]) * delta.detach()
        critic_loss = delta.pow(2)

        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

csv_file = 'data/AAPL.csv'
data = pd.read_csv(csv_file)
data['Close/Last'] = data['Close/Last'].str.replace('$', '').astype(float)
data['Open'] = data['Open'].str.replace('$', '').astype(float)
data['High'] = data['High'].str.replace('$', '').astype(float)
data['Low'] = data['Low'].str.replace('$', '').astype(float)
data = data.iloc[:, 1:]  # Excluding Date

# data = {}

# for tickers in TICKERS:
#     csv_file = 'data/' + tickers + '.csv'
#     temp = pd.read_csv(csv_file)
#     temp['Close/Last'] = temp['Close/Last'].str.replace('$', '').astype(float)
#     temp['Open'] = temp['Open'].str.replace('$', '').astype(float)
#     temp['High'] = temp['High'].str.replace('$', '').astype(float)
#     temp['Low'] = temp['Low'].str.replace('$', '').astype(float)
#     temp = temp.iloc[:,1:]

#     data[tickers] = temp.values

env = TradingEnvironment(data)
state_dim = len(env.get_state())
action_dim = 3
discount_factor=0.9
total_rewards = []

agent = A2CAgent(state_dim, action_dim)

for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        if action != agent.prev_action:
            print(action)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, discount_factor)
        state = next_state
        total_reward += reward

    total_rewards.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

print(total_rewards)
state = env.reset()
done = False

while not done:
    action = agent.choose_action(state)
    state, reward, done = env.step(action)
    env.render()



env.close()
