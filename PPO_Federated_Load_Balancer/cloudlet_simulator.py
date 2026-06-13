import numpy as np

class CloudletSimulator:
    def __init__(self, num_servers=3, num_tasks=3000):
        self.num_servers = num_servers
        self.num_tasks = num_tasks
        self.remaining_tasks = num_tasks
        self.completed_tasks = 0
        self.server_utilization = np.zeros(self.num_servers)
        self.previous_utilization = np.zeros(self.num_servers)  # For trend tracking

    def reset(self):
        self.remaining_tasks = self.num_tasks
        self.completed_tasks = 0
        self.server_utilization = np.zeros(self.num_servers)
        self.previous_utilization = np.zeros(self.num_servers)
        return self.server_utilization.copy()

    def step(self, action):
        tasks_completed = np.random.randint(5, 15)
        if np.random.random() < 0.05:
            tasks_completed = 0  # Random chance of no tasks completed

        tasks_completed = min(tasks_completed, self.remaining_tasks)
        self.completed_tasks += tasks_completed
        self.remaining_tasks -= tasks_completed

        delay = np.random.randint(100, 300)
        delay += self.server_utilization[action] * 0.1  # Delay based on server load

        migrations = np.random.randint(0, 5)

        self.previous_utilization = self.server_utilization.copy()

        # Gradual update of server utilization rather than re-randomizing
        self.server_utilization += np.random.random(self.num_servers) * 10
        self.server_utilization = np.clip(self.server_utilization, 0, 100)

        trend_penalty = 0
        if self.server_utilization[action] > 90 and self.previous_utilization[action] > 85:
            trend_penalty = 0.5

        done = self.remaining_tasks <= 0
        next_state = self.server_utilization.copy()

        info = {
            'delay': delay,
            'migrations': migrations,
            'completed_tasks': tasks_completed,
            'total_tasks': self.num_tasks,
            'server_utilization': self.server_utilization
        }

        reward = tasks_completed * 2 - delay - migrations * 5 - trend_penalty
        return next_state, reward, done, info
