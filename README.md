# Evolution Algorithm Explorer 🧬

An interactive implementation of genetic/evolution algorithms with a user-friendly interface, inspired by Andrzej Dragan's "Quo vAIdis".

## Features

- **Interactive UI**: Streamlit-based interface for easy parameter configuration
- **Multiple Problem Types**:
  - String Matching: Evolve a string to match a target
  - Function Optimization: Minimize mathematical functions (Sphere, Rastrigin)
  - Binary Optimization: Solve the OneMax problem
- **Real-time Visualization**: Watch evolution progress with live charts
- **Configurable Parameters**: Control all aspects of the evolution process
- **Statistical Analysis**: Track fitness, diversity, and convergence

## Installation

1. Clone or navigate to this repository:
```bash
cd evolution
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

### Using the Interface

1. **Select Problem Type**: Choose from String Matching, Function Optimization, or Binary Optimization
2. **Configure Parameters**:
   - **Population Size**: Number of individuals per generation (10-500)
   - **Max Generations**: How many generations to evolve (10-1000)
   - **Mutation Rate**: Probability of random changes (0.0-1.0)
   - **Crossover Rate**: Probability of combining parents (0.0-1.0)
   - **Selection Method**: Tournament or Roulette wheel selection
   - **Elitism Count**: Number of best individuals preserved each generation
3. **Configure Problem**: Set problem-specific parameters
4. **Click "Run Evolution"**: Watch the algorithm work in real-time!

## Algorithm Parameters Explained

### Core Parameters

- **Population Size**: Larger populations explore more solutions but are slower
  - Small (10-50): Fast but may miss optimal solutions
  - Medium (50-200): Good balance
  - Large (200-500): Thorough exploration

- **Mutation Rate**: Controls exploration vs exploitation
  - Low (0.01-0.05): Fine-tuning existing solutions
  - Medium (0.05-0.2): Balanced exploration
  - High (0.2+): High exploration, may be unstable

- **Crossover Rate**: Probability of combining parent genes
  - Typical range: 0.6-0.9
  - Higher values emphasize recombination

- **Elitism**: Preserving best individuals
  - 0: Pure evolution (may lose good solutions)
  - 1-5: Recommended for most problems
  - Higher: More exploitation, less exploration

### Selection Methods

- **Tournament Selection**: Randomly select N individuals, choose the best
  - Tournament Size 2-3: More diversity
  - Tournament Size 5+: Stronger selection pressure

- **Roulette Wheel Selection**: Probability proportional to fitness
  - Good for problems with varying fitness scales
  - May converge faster but less diverse

## Problem Types

### 1. String Matching
Evolve a random string to match a target string.

**Use Case**: Demonstrates basic genetic algorithm concepts

**Parameters**:
- Target String: The string to evolve towards

**Tips**:
- Longer strings require more generations
- High mutation rate (0.1-0.2) works well

### 2. Function Optimization (Sphere)
Minimize the sphere function: f(x) = x₁² + x₂² + ... + xₙ²

**Use Case**: Simple convex optimization

**Parameters**:
- Dimension: Number of variables (2-10)
- Bounds: Search space range

**Tips**:
- Global optimum is at origin (0, 0, ..., 0)
- Converges quickly with proper parameters

### 3. Function Optimization (Rastrigin)
Minimize the Rastrigin function (many local minima)

**Use Case**: Complex optimization with many local optima

**Parameters**:
- Dimension: Number of variables
- Bounds: Typically [-5.12, 5.12]

**Tips**:
- Harder problem with many local minima
- Requires larger population and more generations
- Higher mutation rate helps escape local minima

### 4. Binary Optimization (OneMax)
Maximize the number of 1s in a binary string

**Use Case**: Classic benchmark problem

**Parameters**:
- Binary String Length: Size of the problem

**Tips**:
- Easy problem, converges quickly
- Good for testing algorithm parameters

## Code Structure

```
evolution/
├── evolution.py         # Core algorithm implementation
│   ├── EvolutionConfig  # Configuration dataclass
│   ├── Individual       # Represents a single solution
│   ├── EvolutionAlgorithm # Main algorithm class
│   └── Problem classes  # Example problems
├── app.py              # Streamlit UI application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Customizing for Your Own Problems

To solve your own optimization problem:

1. Create a problem class with these methods:
```python
class MyProblem:
    def create_genome(self):
        # Return a random solution
        pass

    def fitness(self, genome):
        # Return fitness score (higher is better)
        pass

    def mutate(self, genome):
        # Return mutated version of genome
        pass

    def crossover(self, genome1, genome2):
        # Return two child genomes
        pass
```

2. Add your problem to the app.py interface

3. Configure parameters and run!

## Examples

### Example 1: Evolving "Hello World"
- Problem: String Matching
- Target: "Hello World"
- Population: 100
- Generations: 50
- Mutation Rate: 0.1
- Result: Usually converges in 30-40 generations

### Example 2: Finding Minimum of Sphere Function
- Problem: Function Optimization (Sphere)
- Dimension: 2
- Bounds: [-10, 10]
- Population: 50
- Generations: 100
- Mutation Rate: 0.05
- Result: Converges to near [0, 0]

## Theory

### How Evolution Algorithms Work

1. **Initialization**: Create random population of solutions
2. **Evaluation**: Calculate fitness for each individual
3. **Selection**: Choose better individuals as parents
4. **Crossover**: Combine parent solutions to create offspring
5. **Mutation**: Randomly modify offspring
6. **Replacement**: Form new generation
7. **Repeat**: Until convergence or max generations

### When to Use Evolution Algorithms

✅ Good for:
- Complex search spaces
- No gradient information available
- Multiple local optima
- Discrete/combinatorial problems
- Black-box optimization

❌ Not ideal for:
- Simple convex optimization
- When gradient-based methods work
- Very high-dimensional problems (>100 variables)
- Real-time applications (can be slow)

## Performance Tips

- **Start small**: Test with small populations and few generations
- **Monitor diversity**: If diversity drops too quickly, increase mutation rate
- **Adjust elitism**: Too much elitism = premature convergence
- **Population size**: Roughly 10-100x the problem dimension
- **Generations**: Start with 50-100, increase if not converging

## Credits

Inspired by the evolution algorithm described in **"Quo vAIdis"** by Andrzej Dragan, a book exploring artificial intelligence and computational methods.

## License

This implementation is provided for educational purposes.

## Further Reading

- "Quo vAIdis" by Andrzej Dragan
- "Introduction to Evolutionary Computing" by Eiben & Smith
- "Genetic Algorithms in Search, Optimization, and Machine Learning" by Goldberg
