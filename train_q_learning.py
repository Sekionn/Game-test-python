from agents.q_learning_agent import QLearningAgent
from env.platformer_env import MAX_EPISODE_TICKS, PlatformerEnv
from main import LEVELS

GENERATIONS_PER_LEVEL = 100
ATTEMPTS_PER_GENERATION = 200
SUCCESS_REPLAY_PASSES = 8
Q_TABLE_PATH = "q_table.json"
BEST_Q_TABLE_PATH = "best_q_table.json"
EVALUATION_ATTEMPTS = 5
SHOW_PREVIEW_AFTER_GENERATION = True
PREVIEW_MAX_TICKS = MAX_EPISODE_TICKS


def train():
    """Train Q-learning in batches called generations.

    A generation is a group of episodes that are stepped together. The game still
    uses Gymnasium environments, but rendering is disabled so the episodes in
    a generation finish much faster than watching them one at a time.
    """
    agent = QLearningAgent()
    best_agent = None
    best_score = None
    best_details = None

    for level_index, level in enumerate(LEVELS):
        level_name = f"level_{level_index + 1}"
        agent.reset_exploration()
        print(f"Training {level_name}")

        for generation in range(1, GENERATIONS_PER_LEVEL + 1):
            envs = [
                PlatformerEnv(
                    level,
                    render=False,
                    run_label="q_learning_train",
                    level_name=level_name,
                    log_results=True,
                    generation=generation,
                    episode=episode,
                )
                for episode in range(1, ATTEMPTS_PER_GENERATION + 1)
            ]

            states = []
            active = []
            rewards = []
            outcomes = []
            infos = []
            histories = []

            for env in envs:
                state, info = env.reset()
                states.append(state)
                active.append(True)
                rewards.append(0)
                outcomes.append("running")
                infos.append(info)
                histories.append([])

            while any(active):
                for index, env in enumerate(envs):
                    if not active[index]:
                        continue

                    action = agent.act(states[index], training=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    histories[index].append((states[index], action, reward, next_state, done))
                    agent.learn(
                        states[index],
                        action,
                        reward,
                        next_state,
                        done,
                    )

                    states[index] = next_state
                    rewards[index] += reward
                    infos[index] = info

                    if done:
                        active[index] = False
                        outcomes[index] = "complete" if terminated else "timeout"
                        if terminated:
                            replay_success(agent, histories[index])
                        agent.finish_episode()

            completed = outcomes.count("complete")
            average_ticks = sum(info["ticks"] for info in infos) / len(infos)
            average_inputs = sum(info["inputs"] for info in infos) / len(infos)
            average_reward = sum(rewards) / len(rewards)

            print(
                f"{level_name} generation {generation:02d}: "
                f"{completed}/{ATTEMPTS_PER_GENERATION} complete, "
                f"avg ticks={average_ticks:.1f}, "
                f"avg inputs={average_inputs:.1f}, "
                f"avg reward={average_reward:.2f}, "
                f"epsilon={agent.epsilon:.3f}"
            )

            for env in envs:
                env.close()

            evaluation = evaluate_agent(agent, level, level_name, generation)
            print(
                f"{level_name} generation {generation:02d} evaluation: "
                f"{evaluation['completed']}/{EVALUATION_ATTEMPTS} complete, "
                f"avg ticks={evaluation['average_ticks']:.1f}, "
                f"avg reward={evaluation['average_reward']:.2f}, "
                f"score={evaluation['score']:.2f}"
            )

            if best_score is None or evaluation["score"] > best_score:
                best_score = evaluation["score"]
                best_agent = agent.clone()
                best_details = {
                    "level": level_name,
                    "generation": generation,
                    **evaluation,
                }
                best_agent.save(BEST_Q_TABLE_PATH)
                print(
                    f"New best Q-table saved from {level_name} "
                    f"generation {generation:02d}"
                )

            if SHOW_PREVIEW_AFTER_GENERATION:
                preview_training_progress(agent, level, level_name, generation)

    agent.save(Q_TABLE_PATH)
    print(f"Saved learned Q-table to {Q_TABLE_PATH}")
    if best_agent is not None:
        print(
            f"Best Q-table saved to {BEST_Q_TABLE_PATH}: "
            f"{best_details['level']} generation {best_details['generation']}, "
            f"score={best_details['score']:.2f}"
        )


def replay_success(agent, history):
    for _ in range(SUCCESS_REPLAY_PASSES):
        for state, action, reward, next_state, done in reversed(history):
            agent.learn(state, action, reward, next_state, done)


def evaluate_agent(agent, level, level_name, generation):
    completed = 0
    total_ticks = 0
    total_reward = 0

    for attempt in range(1, EVALUATION_ATTEMPTS + 1):
        env = PlatformerEnv(
            level,
            render=False,
            run_label="q_learning_evaluation",
            level_name=level_name,
            log_results=True,
            max_ticks=MAX_EPISODE_TICKS,
            generation=generation,
            episode=attempt,
        )
        state, info = env.reset()
        terminated = False
        truncated = False
        attempt_reward = 0

        while not terminated and not truncated:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            attempt_reward += reward

        if terminated:
            completed += 1

        total_ticks += info["ticks"]
        total_reward += attempt_reward
        env.close()

    average_ticks = total_ticks / EVALUATION_ATTEMPTS
    average_reward = total_reward / EVALUATION_ATTEMPTS
    score = completed * 10000 - average_ticks + average_reward

    return {
        "completed": completed,
        "average_ticks": average_ticks,
        "average_reward": average_reward,
        "score": score,
    }


def preview_training_progress(agent, level, level_name, generation):
    """Render one short greedy attempt so training progress is visible.

    The batch training itself runs headless and fast. This preview shows what
    the current Q-table has learned after a generation, without forcing you to
    watch every training episode.
    """
    env = PlatformerEnv(
        level,
        render=True,
        run_label="q_learning_preview",
        level_name=level_name,
        log_results=True,
        max_ticks=PREVIEW_MAX_TICKS,
        generation=generation,
        episode=0,
    )
    state, info = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.act(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)

    env.close()

    print(
        f"{level_name} generation {generation:02d} preview: "
        f"{'complete' if terminated else 'timeout'}, "
        f"ticks={info['ticks']}, "
        f"inputs={info['inputs']}, "
        f"distance={info['distance_to_door']:.2f}"
    )


if __name__ == "__main__":
    train()
