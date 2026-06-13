import numpy as np
from cloudlet_env import CloudletEnv  # Import your CloudletEnv class
from ppo_agent import PPOAgent  # Import the PPOAgent class

# Initialize the environment and agent
env = CloudletEnv(num_servers=3, max_steps=1000)
agent = PPOAgent(num_servers=env.num_servers, alpha=0.001)

# Start training the PPO agent
num_episodes = 1000  # Number of episodes to run

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Agent selects an action (which server to pick)
        action = agent.select_action(state)

        # Take the action in the environment and get the feedback
        next_state, reward, done, info = env.step(action)
        
        # Store the experience for later training
        agent.store_experience(state, action, reward, next_state, done)

        # Train the agent after each episode or after a batch of experiences
        if done:
            agent.train()
        
        # Update the state
        state = next_state
        total_reward += reward

    # Print performance (optional)
    print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

