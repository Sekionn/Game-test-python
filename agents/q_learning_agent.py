import json
import random
from copy import deepcopy
from pathlib import Path


class QLearningAgent:
    def __init__(
        self,
        actions=6,
        learning_rate=0.1,
        discount=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.2,
        state_precision=0,
    ):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.initial_min_epsilon = min_epsilon
        self.min_epsilon = min_epsilon
        self.state_precision = state_precision
        self.q_table = {}

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.actions)

        return self.best_action(state)

    def best_action(self, state):
        state_key = self._state_key(state)
        action_values = self.q_table.get(state_key, [0.0] * self.actions)
        best_value = max(action_values)
        best_actions = [
            action
            for action, value in enumerate(action_values)
            if value == best_value
        ]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        self.q_table.setdefault(state_key, [0.0] * self.actions)
        self.q_table.setdefault(next_state_key, [0.0] * self.actions)

        old_value = self.q_table[state_key][action]
        best_future_value = max(self.q_table[next_state_key])
        target = reward if done else reward + self.discount * best_future_value

        self.q_table[state_key][action] = old_value + self.learning_rate * (
            target - old_value
        )

    def finish_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset_exploration(self, epsilon=None):
        self.epsilon = self.initial_epsilon if epsilon is None else epsilon
        self.min_epsilon = self.initial_min_epsilon

    def clone(self):
        agent = QLearningAgent(
            actions=self.actions,
            learning_rate=self.learning_rate,
            discount=self.discount,
            epsilon=self.epsilon,
            epsilon_decay=self.epsilon_decay,
            min_epsilon=self.min_epsilon,
            state_precision=self.state_precision,
        )
        agent.q_table = deepcopy(self.q_table)
        return agent

    def save(self, path):
        data = {
            "actions": self.actions,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "initial_epsilon": self.initial_epsilon,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "initial_min_epsilon": self.initial_min_epsilon,
            "min_epsilon": self.min_epsilon,
            "state_precision": self.state_precision,
            "q_table": self.q_table,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            actions=data["actions"],
            learning_rate=data["learning_rate"],
            discount=data["discount"],
            epsilon=0.0,
            epsilon_decay=data["epsilon_decay"],
            min_epsilon=data["min_epsilon"],
            state_precision=data["state_precision"],
        )
        agent.initial_epsilon = data.get("initial_epsilon", data.get("epsilon", 1.0))
        agent.initial_min_epsilon = data.get("initial_min_epsilon", data["min_epsilon"])
        agent.q_table = data["q_table"]
        return agent

    def _state_key(self, state):
        return "|".join(
            str(round(float(value), self.state_precision))
            for value in state
        )
