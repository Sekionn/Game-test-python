from agents.q_learning_agent import QLearningAgent
from env.platformer_env import PlatformerEnv
from main import LEVELS

EPISODES_PER_LEVEL = 100
Q_TABLE_PATH = "q_table.json"


def train():
    agent = QLearningAgent()

    for level_index, level in enumerate(LEVELS):
        level_name = f"level_{level_index + 1}"
        env = PlatformerEnv(
            level,
            render=False,
            run_label="q_learning_train",
            level_name=level_name,
            log_results=True,
        )

        print(f"Training {level_name}")

        for episode in range(1, EPISODES_PER_LEVEL + 1):
            state, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            while not terminated and not truncated:
                action = agent.act(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.learn(state, action, reward, next_state, terminated or truncated)

                state = next_state
                total_reward += reward

            agent.finish_episode()
            status = "complete" if terminated else "timeout"
            print(
                f"{level_name} episode {episode:03d}: "
                f"{status}, ticks={info['ticks']}, "
                f"inputs={info['inputs']}, "
                f"reward={total_reward:.2f}, "
                f"epsilon={agent.epsilon:.3f}"
            )

        env.close()

    agent.save(Q_TABLE_PATH)
    print(f"Saved learned Q-table to {Q_TABLE_PATH}")


if __name__ == "__main__":
    train()
