# Evolution Field Lab

An interactive ecosystem for exploring natural selection. Creatures search for food, spend energy, reproduce, inherit traits with occasional mutation, and die from starvation or age. The live charts show how population, resources, speed, and sight change over time.

**Live app:** [pelnystak.github.io/evolution](https://pelnystak.github.io/evolution/)

**Kudo Lab:** [paweljedrzejewski.pl/evolution](https://paweljedrzejewski.pl/evolution/)

## Explore

1. Start the simulation or advance one step at a time.
2. Change food supply and mutation chance while it runs.
3. Change the starting population or random seed, then reset.
4. Compare the population and trait charts across runs. The same seed and settings produce the same sequence of outcomes.

The browser app has no runtime dependencies. Serve the repository locally because ES modules do not load consistently from a `file://` URL:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Model

The world advances in fixed steps. Each creature has a speed and a sight trait. It moves toward visible food or wanders, spends energy on movement and sensing, and gains energy when it eats. Adults with enough energy can reproduce; each newborn inherits its parent's traits with a configurable chance of mutation. Creatures die when energy reaches zero or they reach old age.

Food regrows at a configurable rate. Population and food have explicit caps so long runs stay responsive. These rules make selection visible; they are a simplified educational model, not a biological prediction.

| File | Purpose |
| --- | --- |
| `index.html` | Interface and explanation |
| `style.css` | Kudo Lab visual design and responsive layout |
| `simulation.mjs` | Seeded, deterministic ecosystem model |
| `app.js` | Canvas rendering, charts, and controls |

## Python experiments

The repository also contains a separate Python genetic algorithm explorer. It demonstrates optimization problems such as string matching and function optimization. It is not the engine behind the browser ecosystem.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The Python example can also run from the command line with `python example.py`.
