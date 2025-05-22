import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import gym 
from torch.distributions import Categorical

# Hyperparameters 
learning_rate = 0.02 
gamma = 0.99 

class Policy(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Policy, self).__init__()
        self.data = []  # Stores (reward, log_prob)
        self.network = nn.Sequential(
            nn.Linear(n_states, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, n_actions), 
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.network(x)

    def put_data(self, data):
        self.data.append(data)

    def train_net(self):
        
        R = 0
        self.optimizer.zero_grad()
        for r, log_prob in reversed(self.data):
            R = r + gamma * R
            loss = -log_prob * R  # Higher R → lower loss → reinforce good action
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    policy = Policy(env.observation_space.shape[0], env.action_space.n)

    score = 0.0
    print_interval = 20

    for episode in range(10000):
        state, _ = env.reset()
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = torch.log(probs[action])
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            policy.put_data((reward, log_prob))
            state = next_state
            score += reward
        # print("training")
        policy.train_net()

        if episode % print_interval == 0 and episode != 0:
            print(f"# Episode: {episode}, avg score: {score / print_interval}")
            score = 0.0

    # Saving  model weights
    torch.save(policy.state_dict(), "reinforce_cartpole.pth")
    print("Model saved as reinforce_cartpole.pth ✅")
    env.close()

if __name__ == '__main__':
    main()
