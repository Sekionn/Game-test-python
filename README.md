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

Watch the trained Q-learning AI:

```bash
python watch_q_learning.py
```

Summarize recorded timing and input results:

```bash
python summarize_results.py
```

Episode results are appended to `game_results.txt`. The file records the run
type, level, outcome, ticks, inputs, elapsed seconds, and final reward.
