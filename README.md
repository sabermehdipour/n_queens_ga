# 🧬 N-Queens Solver Using Genetic Algorithm

A clean, modular, and object-oriented implementation of the Genetic Algorithm (GA) to solve the classic **N-Queens Problem** — a fundamental combinatorial optimization problem in computer science and artificial intelligence.

---

## 🧠 What is the N-Queens Problem?

The **N-Queens problem** asks how to place N queens on an N×N chessboard so that no two queens threaten each other. That is, no two queens may share the same row, column, or diagonal.

This project uses a **Genetic Algorithm (GA)** to evolve a population of candidate solutions until a valid (or near-optimal) configuration is found.

---

## 🚀 Features

- Object-Oriented Python Implementation
- Genetic Algorithm components:
  - Selection (roulette wheel)
  - Crossover (Partially Mapped Crossover — PMX)
  - Mutation (swap mutation)
- Plots fitness progression over generations
- Clean separation of logic and visualization
- Easily extensible for different board sizes and hyperparameters

---

## 🧪 Example Output
```bash
Best solution: [1 5 8 6 3 7 2 4]
Fitness score: 999999.000001
```
---
## 📦 Installation

### ✅ Prerequisites

- Python 
- pip

### 🔧 Create and Activate a Virtual Environment


### Clone the repository
```bash
git clone https://github.com/your-username/n-queens-ga.git
cd n-queens-ga
```
### (Optional but recommended) Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
or manually install
```bash
pip install numpy matplotlib
```

---

# ▶️ Usage
To run the solver:
```bash
python n_queens_ga.py
```
The script will:
- Evolve solutions to the N-Queens problem using GA
- Plot best fitness per generation
- Print the best solution and its fitness

---

# ⚙️ Configuration
You can easily configure GA hyperparameters in n_queens_ga.py:
```bash
ga = QueenGA(
    n=8,               # Size of the board (e.g. 8x8)
    pop_size=50,       # Number of individuals in population
    mutation_rate=0.1, # Swap mutation probability
    max_iter=1000      # Number of generations
)
```
