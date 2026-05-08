# Game test python

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the game as a human:

```bash
python main.py human
```

Train the Q-learning AI:

```bash
python train_q_learning.py
```

Watch several training attempts run at the same time:

```bash
python visual_train_q_learning.py
```

Training uses generations. One generation is a batch of episodes that are run
without rendering, so the AI can test many attempts quickly.

Current training settings:

```python
GENERATIONS_PER_LEVEL = 5
ATTEMPTS_PER_GENERATION = 100
```

That gives the AI `100` attempts in each generation. With the current settings,
each level gets `500` training attempts total.

After each generation, training shows one short preview attempt using the current
Q-table. This lets you see whether the AI is improving without watching every
episode in the generation.

Watch the trained Q-learning AI:

```bash
python watch_q_learning.py
```

`train_q_learning.py` saves both the latest policy and the best evaluated policy:

```text
q_table.json
best_q_table.json
```

`watch_q_learning.py` watches `best_q_table.json` by default.

Summarize recorded timing and input results:

```bash
python summarize_results.py
```

Episode results are appended to `game_results.txt`. The file records the run
type, level, generation, episode, outcome, ticks, inputs, elapsed seconds, and
final reward.
