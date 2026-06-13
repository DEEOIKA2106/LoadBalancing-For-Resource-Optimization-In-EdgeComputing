import numpy as np

class CloudletEnv:
    def __init__(self, num_servers=3, num_tasks=200):
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
        # Simulate task completion based on server utilization
        tasks_completed = np.random.randint(5, 15) * (1 - self.server_utilization[action] / 100)
        tasks_completed = min(tasks_completed, self.remaining_tasks)
        self.completed_tasks += tasks_completed
        self.remaining_tasks -= tasks_completed

        # Simulate delay
        delay = np.random.randint(100, 300) + self.server_utilization[action] * 0.1

        # Simulate migrations
        migrations = np.random.randint(0, 5) * (self.server_utilization[action] / 100)
        migrations = int(migrations)

        # Calculate migration time
        migration_time = migrations * 10  # ms

        # Update utilization
        self.previous_utilization = self.server_utilization.copy()
        self.server_utilization += np.random.random(self.num_servers) * 10
        self.server_utilization = np.clip(self.server_utilization, 0, 100)

        # Trend penalty
        trend_penalty = 0
        if self.server_utilization[action] > 90 and self.previous_utilization[action] > 85:
            trend_penalty = 5  # Slightly increased to match new scale

        done = self.remaining_tasks <= 0
        next_state = self.server_utilization.copy()

        # Info
        info = {
            'delay': delay,
            'migrations': migrations,
            'completed_tasks': tasks_completed,
            'total_tasks': self.num_tasks,
            'server_utilization': self.server_utilization,
            'migration_time': migration_time
        }

        # ✅ NEW POSITIVE REWARD FUNCTION
        reward = (
            tasks_completed * 10         # Task bonus
            - delay * 0.3                # Delay penalty scaled
            - migrations * 5            # Migration penalty
            - trend_penalty             # Utilization penalty
        )
        reward = max(0, reward)  # Keep reward non-negative

        return next_state, reward, done, info, trend_penalty
