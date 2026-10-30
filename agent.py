import numpy as np
import random

class RLAgent:
    def __init__(self, state_size=16, action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Simple Q-table for demonstration
        self.q_table = {}
        
    def get_state(self, gpus):
        """Extract state from GPU cluster"""
        state = []
        for gpu in gpus[:4]:  # Limit to 4 GPUs for fixed state size
            state.extend([
                gpu.get_memory_usage_percentage() / 100.0,
                gpu.temperature / 100.0,
                len(gpu.task_queue) / 10.0,
                len(gpu.running_tasks) / 3.0
            ])
        
        # Pad or truncate to fixed size
        while len(state) < self.state_size:
            state.append(0.0)
        return tuple(state[:self.state_size])
    
    def choose_action(self, state, num_actions):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randint(0, num_actions - 1)
        
        # Get Q-values for current state
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size
        
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, num_actions):
        """Update Q-table"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.action_size
        
        # Q-learning update
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        new_value = old_value + self.learning_rate * (reward + 0.95 * next_max - old_value)
        self.q_table[state][action] = new_value
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def snapshot_q_table(self):
        return {state: list(values) for state, values in self.q_table.items()}

    def restore_q_table(self, snapshot):
        if not snapshot:
            self.q_table = {}
            return
        self.q_table = {state: list(values) for state, values in snapshot.items()}
