import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from elegantrl.agents.AgentA2C import AgentA2C

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

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        pi, _ = self.actor_critic(state)
        action_probs = F.softmax(pi, dim=-1)
        action = np.random.choice(range(action_probs.shape[0]), p=action_probs.detach().cpu().numpy())
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        _, value = self.actor_critic(state)
        _, next_value = self.actor_critic(next_state)

        delta = reward + 0.99 * (1 - done) * next_value - value
        actor_loss = -torch.log(self.actor_critic.pi(state)[action]) * delta.detach()
        critic_loss = delta.pow(2)

        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

env = gym.make('StockTrading-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = A2CAgent(state_dim, action_dim)

for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {env.account.cumulative_reward}")

state = env.reset()
done = False

while not done:
    action = agent.choose_action(state)
    state, reward, done, _ = env.step(action)
    env.render()

env.close()
