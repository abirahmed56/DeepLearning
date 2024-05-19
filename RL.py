import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CustomEnvironment:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.state_dim = 2  # Dimension of state space
        self.action_dim = 3  # Dimension of action space
        self.max_steps = 10  # Maximum number of steps per episode
        self.reset()

    def reset(self):
        self.steps = 0
        self.agents_state = torch.zeros((self.num_agents, self.state_dim))
        self.agents_done = torch.zeros(self.num_agents, dtype=torch.bool)
        self.global_state = torch.zeros(self.state_dim)  # Global state

    def step(self, actions):
        self.steps += 1
        rewards = torch.zeros(self.num_agents)
        dones = torch.zeros(self.num_agents, dtype=torch.bool)

        # Update agent states based on actions
        for i in range(self.num_agents):
            if not self.agents_done[i]:
                self.agents_state[i] += torch.tensor(actions[i])
                if torch.all(self.agents_state[i] >= self.global_state):
                    rewards[i] = 1.0
                    self.agents_done[i] = True
                    dones[i] = True

        # Update global state
        self.global_state += torch.mean(self.agents_state, dim=0)

        # Check if max steps reached
        if self.steps >= self.max_steps:
            dones[:] = True

        return self.global_state, rewards, dones


class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimplePolicy, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return torch.softmax(self.fc(state), dim=-1)


def train(env, agents, num_episodes=1000, lr=0.001):
    optimizer = optim.Adam(agents[0].parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for episode in range(num_episodes):
        env.reset()
        episode_rewards = torch.zeros(len(agents))

        while not torch.all(env.agents_done):
            actions = [agent(env.global_state.unsqueeze(0)) for agent in agents]
            actions = [torch.multinomial(action, 1).item() for action in actions]
            next_state, rewards, dones = env.step(actions)
            episode_rewards += rewards

            # Update policy
            optimizer.zero_grad()
            loss = sum([criterion(agent(env.global_state.unsqueeze(0)), torch.tensor([action])) for agent, action in
                        zip(agents, actions)])
            loss.backward()
            optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Rewards: {episode_rewards}")


# Main
num_agents = 3
env = CustomEnvironment(num_agents)
agents = [SimplePolicy(env.state_dim, env.action_dim) for _ in range(num_agents)]
train(env, agents)
