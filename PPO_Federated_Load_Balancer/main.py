import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from cloudlet_env import CloudletEnv

def train():
    num_servers = 3
    env = CloudletEnv(num_servers)
    agent = PPOAgent(num_servers)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    gamma = 0.99
    epsilon = 0.2
    episodes = 25
    steps_per_episode = 150  # Limit steps per episode for faster execution

    # Lists to store metrics for plotting
    episode_rewards = []
    total_tasks_list = []
    total_delay_list = []
    total_migrations_list = []
    total_migration_time_list = []  # Added migration time list

    for episode in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        total_reward = 0
        total_tasks = 0
        total_delay = 0
        total_migrations = 0
        total_migration_time = 0

        for step in range(steps_per_episode):
            action, log_prob = agent.get_action(state)
            next_state, reward, done, info, _ = env.step(action.item())

            # Compute advantage
            next_state = torch.tensor(next_state, dtype=torch.float32)
            value = agent.forward(state)
            next_value = agent.forward(next_state)
            advantage = reward + gamma * next_value - value

            # Surrogate objective
            ratio = torch.exp(log_prob - agent.get_action(state)[1])
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            surrogate_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

            # Value loss
            value_loss = 0.5 * (reward + gamma * next_value - value) ** 2

            # Total loss
            loss = surrogate_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            total_tasks += info['completed_tasks']
            total_delay += info['delay']
            total_migrations += info['migrations']
            total_migration_time += info.get('migration_time', 0)  # Track migration time

            state = next_state

            if done:
                break

        # Store metrics for plotting
        episode_rewards.append(total_reward)
        total_tasks_list.append(total_tasks)
        total_delay_list.append(total_delay)
        total_migrations_list.append(total_migrations)
        total_migration_time_list.append(total_migration_time)  # Store migration time

        # Print episode summary
        print(f"Episode {episode + 1:04d} | Reward: {total_reward:.2f} | Tasks: {total_tasks} | "
              f"Delay: {total_delay:.2f} | Migrations: {total_migrations} | Migration Time: {total_migration_time:.2f} ms")

    # Save the model
    torch.save(agent.state_dict(), 'ppo_model.pth')
    print("\n✅ Training complete. Model saved as 'ppo_model.pth'.")

    # Plot the results using matplotlib
    plt.figure(figsize=(12, 8))

    # Plot reward over episodes
    plt.subplot(2, 3, 1)
    plt.plot(range(episodes), episode_rewards, label="Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward over Episodes")
    plt.grid(True)

    # Plot total tasks completed over episodes
    plt.subplot(2, 3, 2)
    plt.plot(range(episodes), total_tasks_list, label="Completed Tasks", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Completed Tasks")
    plt.title("Tasks Completed over Episodes")
    plt.grid(True)

    # Plot total delay over episodes
    plt.subplot(2, 3, 3)
    plt.plot(range(episodes), total_delay_list, label="Delay", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Delay (ms)")
    plt.title("Delay over Episodes")
    plt.grid(True)

    # Plot total migrations over episodes
    plt.subplot(2, 3, 4)
    plt.plot(range(episodes), total_migrations_list, label="Migrations", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Migrations")
    plt.title("Migrations over Episodes")
    plt.grid(True)

    # Plot migration time over episodes (New graph)
    plt.subplot(2, 3, 5)
    plt.plot(range(episodes), total_migration_time_list, label="Migration Time", color="purple")
    plt.xlabel("Episode")
    plt.ylabel("Migration Time (ms)")
    plt.title("Migration Time over Episodes")
    plt.grid(True)

    # Display the plots
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train()
