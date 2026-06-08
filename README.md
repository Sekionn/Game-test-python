# Spillets regler
1. Spilleren har 5 inputmuligheder: op-, ned-, venstre- og højrepiletasterne samt `R`.
   - Venstre og højre bevæger spilleren på x-aksen og kontrollerer instantieringer af `Extender`-klassen, som er instantieret med x-aksen som deres `axis`.
   - Op og ned kontrollerer `Extender`-klassen, som udvider eller skrumper afhængigt af den instantierede retning, når den er instantieret med y-aksen som dens `axis`-parameter. Hvis `Player`-klassen står oven på en extender, der udvider sig opad, bliver `Player`-klassen løftet med op ad y-aksen.
   - `R`-knappen udfører et "soft reboot" af den bane, der spilles på det givne tidspunkt. Dette giver spilleren mulighed for at starte banen forfra uden at påvirke dataene fra spilsessionen.

2. For at gennemføre en bane skal spilleren kollidere med "døren", som er den grønne klods på banen.

3. Der skal være mulighed for at have flere spillerobjekter i banen på én gang, og disse skal kunne stables oven på hinanden.

4. Der skal være mulighed for at tilføje en modsat spiller. Den eneste forskel er, at den skal bevæge sig modsat på venstre- og højreinput sammenlignet med det normale `Player`-objekt.

5. Hver bane skal gennemspilles 5 gange af den menneskelige spiller, før vedkommende går videre til næste bane.


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
GENERATIONS_PER_LEVEL = 100
ATTEMPTS_PER_GENERATION = 300
```

That gives the AI `300` attempts in each generation. With the current settings,
each level gets `30000` training attempts total.

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
