from agents.q_learning_agent import QLearningAgent
from env.platformer_env import MAX_EPISODE_TICKS, PlatformerEnv
from main import LEVELS

GENERATIONS_PER_LEVEL = 100
ATTEMPTS_PER_GENERATION = 300
STARTING_EPSILON = 1.0
TARGET_EPSILON = 0.05
SUCCESS_REPLAY_PASSES = 12
Q_TABLE_PATH = "q_table.json"
BEST_Q_TABLE_TEMPLATE = "best_q_table_{level_name}.json"
EVALUATION_ATTEMPTS = 5
PERFECT_EVALUATION_STREAK_TO_ADVANCE = 5
SHOW_PREVIEW_AFTER_GENERATION = True
PREVIEW_MAX_TICKS = MAX_EPISODE_TICKS


def train():
    """Train Q-learning in batches called generations.

    A generation is a group of episodes that are stepped together. The game still
    uses Gymnasium environments, but rendering is disabled so the episodes in
    a generation finish much faster than watching them one at a time.
    """
    agent = QLearningAgent(
        epsilon=STARTING_EPSILON,
        epsilon_decay=calculate_epsilon_decay(
            STARTING_EPSILON,
            TARGET_EPSILON,
            GENERATIONS_PER_LEVEL,
            ATTEMPTS_PER_GENERATION,
        ),
        min_epsilon=TARGET_EPSILON,
    )
    all_best_details = []

    for level_index, level in enumerate(LEVELS):
        level_name = f"level_{level_index + 1}"
        best_agent = None
        best_score = None
        best_details = None
        best_q_table_path = best_q_table_path_for(level_name)
        print(f"Training {level_name}")
        perfect_evaluation_streak = 0

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
            training_candidate = summarize_training_candidate(outcomes, infos, rewards)

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
                f"avg inputs={evaluation['average_inputs']:.1f}, "
                f"avg reward={evaluation['average_reward']:.2f}, "
                f"score={evaluation['score']:.2f}"
            )
            if training_candidate is not None:
                print(
                    f"{level_name} generation {generation:02d} best training attempts: "
                    f"{training_candidate['completed']}/{EVALUATION_ATTEMPTS} complete, "
                    f"avg ticks={training_candidate['average_ticks']:.1f}, "
                    f"avg inputs={training_candidate['average_inputs']:.1f}, "
                    f"avg reward={training_candidate['average_reward']:.2f}, "
                    f"score={training_candidate['score']:.2f}"
                )

            perfect_generation = completed == ATTEMPTS_PER_GENERATION
            perfect_evaluation = evaluation["completed"] == EVALUATION_ATTEMPTS

            if perfect_generation and perfect_evaluation:
                perfect_evaluation_streak += 1
            else:
                perfect_evaluation_streak = 0

            print(
                f"{level_name} perfect streak: "
                f"{perfect_evaluation_streak}/"
                f"{PERFECT_EVALUATION_STREAK_TO_ADVANCE}"
            )

            best_score, best_agent, best_details = update_best_agent(
                agent,
                best_score,
                best_agent,
                best_details,
                level_name,
                generation,
                "training",
                training_candidate,
                best_q_table_path,
            )

            if SHOW_PREVIEW_AFTER_GENERATION:
                preview_training_progress(agent if best_agent is None else best_agent, level, level_name, generation)

            if perfect_evaluation_streak >= PERFECT_EVALUATION_STREAK_TO_ADVANCE:
                print(
                    f"{level_name} reached "
                    f"{ATTEMPTS_PER_GENERATION}/{ATTEMPTS_PER_GENERATION} "
                    f"training completions and "
                    f"{EVALUATION_ATTEMPTS}/{EVALUATION_ATTEMPTS} evaluation "
                    f"completions for "
                    f"{PERFECT_EVALUATION_STREAK_TO_ADVANCE} generations in a row; "
                    "moving to the next level"
                )
                break

        if best_details is not None:
            all_best_details.append(best_details)
        
        agent.reset_exploration(epsilon=0.6)
        agent.reset_decay(decay=calculate_epsilon_decay(
            0.6,
            TARGET_EPSILON,
            GENERATIONS_PER_LEVEL,
            ATTEMPTS_PER_GENERATION,
        ))
        
    agent.save(Q_TABLE_PATH)
    print(f"Saved learned Q-table to {Q_TABLE_PATH}")
    for details in all_best_details:
        print(
            f"Best Q-table saved to {details['path']}: "
            f"{details['level']} generation {details['generation']}, "
            f"source={details['source']}, "
            f"score={details['score']:.2f}"
        )


def calculate_epsilon_decay(
    starting_epsilon,
    target_epsilon,
    generations,
    attempts_per_generation,
):
    episodes = generations * attempts_per_generation
    if episodes <= 0:
        raise ValueError("Training must include at least one episode.")

    return (target_epsilon / starting_epsilon) ** (1 / episodes)


def replay_success(agent, history):
    for _ in range(SUCCESS_REPLAY_PASSES):
        for state, action, reward, next_state, done in reversed(history):
            agent.learn(state, action, reward, next_state, done)


def best_q_table_path_for(level_name):
    return BEST_Q_TABLE_TEMPLATE.format(level_name=level_name)


def clone_for_level(agent, level_name):
    level_number = float(level_name.removeprefix("level_"))
    level_prefix = f"{level_number}|"
    level_agent = agent.clone()
    level_agent.q_table = {
        state: values
        for state, values in agent.q_table.items()
        if state.startswith(level_prefix)
    }
    return level_agent


def summarize_training_candidate(outcomes, infos, rewards):
    completed_attempts = [
        (infos[index], rewards[index])
        for index, outcome in enumerate(outcomes)
        if outcome == "complete"
    ]

    if not completed_attempts:
        return None

    completed_attempts.sort(
        key=lambda attempt: (
            attempt[0]["ticks"],
            attempt[0]["inputs"],
            -attempt[1],
        )
    )
    best_attempts = completed_attempts[:EVALUATION_ATTEMPTS]
    completed = len(best_attempts)
    average_ticks = sum(info["ticks"] for info, _ in best_attempts) / completed
    average_inputs = sum(info["inputs"] for info, _ in best_attempts) / completed
    average_reward = sum(reward for _, reward in best_attempts) / completed

    return {
        "completed": completed,
        "average_ticks": average_ticks,
        "average_inputs": average_inputs,
        "average_reward": average_reward,
        "score": score_attempt_group(
            completed,
            average_ticks,
            average_inputs,
            average_reward,
        ),
    }


def score_attempt_group(completed, average_ticks, average_inputs, average_reward):
    return completed * 100000 - average_ticks - average_inputs + average_reward


def update_best_agent(
    agent,
    best_score,
    best_agent,
    best_details,
    level_name,
    generation,
    source,
    candidate,
    path,
):
    if candidate is None:
        return best_score, best_agent, best_details

    if best_score is not None and candidate["score"] <= best_score:
        return best_score, best_agent, best_details

    best_score = candidate["score"]
    best_agent = clone_for_level(agent, level_name)
    best_details = {
        "level": level_name,
        "generation": generation,
        "source": source,
        "path": path,
        **candidate,
    }
    best_agent.save(path)
    print(
        f"New best Q-table saved from {level_name} "
        f"generation {generation:02d} ({source}) to {path}"
    )
    return best_score, best_agent, best_details


def evaluate_agent(agent, level, level_name, generation):
    completed = 0
    total_ticks = 0
    total_inputs = 0
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
        total_inputs += info["inputs"]
        total_reward += attempt_reward
        env.close()

    average_ticks = total_ticks / EVALUATION_ATTEMPTS
    average_inputs = total_inputs / EVALUATION_ATTEMPTS
    average_reward = total_reward / EVALUATION_ATTEMPTS
    score = score_attempt_group(
        completed,
        average_ticks,
        average_inputs,
        average_reward,
    )

    return {
        "completed": completed,
        "average_ticks": average_ticks,
        "average_inputs": average_inputs,
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
