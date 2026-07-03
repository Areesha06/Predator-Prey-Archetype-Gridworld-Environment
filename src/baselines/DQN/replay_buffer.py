# src/baselines/DQN/replay_buffer.py
"""
Fixed-size circular buffer of past transitions, with random batch sampling (training does not use expereinces in order).
"""

from collections import deque
import numpy as np

# each agent has its own replay memory
class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = None): #seed makes random sampling reproducible 
        if capacity <= 0:
            raise ValueError(f"buffer capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._buffer = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed) # random number geenrator

    # state -> action -> reward -> next_state -> done == one step for agent
    def push(self, state, action, reward, next_state, done) -> None: #push that into buffer
        self._buffer.append((
            np.asarray(state, dtype=np.float64),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float64),
            bool(done),
        ))

    def sample(self, batch_size: int):
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions; buffer only "
                f"has {len(self._buffer)} stored so far."
            )
        indices = self.rng.choice(len(self._buffer), size=batch_size, replace=False) #we never pick the same transition twice within one batch
        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float64),
            np.stack(next_states),
            np.array(dones, dtype=bool),
        )

    def __len__(self):
        return len(self._buffer)