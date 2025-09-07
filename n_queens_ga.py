import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class Individual:
    """
    Represents a single candidate solution (chromosome) for the N-Queens problem.

    Attributes:
        genes (np.ndarray): A permutation representing queen positions.
        fitness_score (float): Fitness value based on number of non-conflicting queens.
    """

    def __init__(self, genes: np.ndarray) -> None:
        """
        Initialize an individual with a given gene sequence.

        Args:
            genes (np.ndarray): A permutation of queen positions.
        """
        self.genes: np.ndarray = genes
        self.fitness_score: float = self._evaluate_fitness()

    def _evaluate_fitness(self) -> float:
        """
        Compute fitness based on the number of diagonal conflicts.

        Returns:
            float: Fitness score (max_conflicts - actual conflicts).
        """
        conflicts: int = 0
        for i in range(len(self.genes) - 1):
            for j in range(i + 1, len(self.genes)):
                if abs(i - j) == abs(self.genes[i] - self.genes[j]):
                    conflicts += 1
        max_conflicts: int = len(self.genes) * (len(self.genes) - 1) // 2
        return max_conflicts - conflicts

    def mutate(self, mutation_rate: float) -> 'Individual':
        """
        Perform swap mutation on the individual's genes.

        Args:
            mutation_rate (float): Probability of mutation occurring.

        Returns:
            Individual: A new individual with possible mutation.
        """
        genes = self.genes.copy()
        if np.random.rand() < mutation_rate:
            i, j = np.random.randint(len(genes)), np.random.randint(len(genes))
            genes[i], genes[j] = genes[j], genes[i]
        return Individual(genes)

    @staticmethod
    def crossover(parent1: 'Individual', parent2: 'Individual') -> Tuple['Individual', 'Individual']:
        """
        Perform Order Crossover (OX) to produce offspring.

        Args:
            parent1 (Individual): First parent.
            parent2 (Individual): Second parent.

        Returns:
            Tuple[Individual, Individual]: Two offspring from the parents.
        """
        size: int = len(parent1.genes)
        start, end = sorted(np.random.choice(size, 2, replace=False))

        def ox(p1: Individual, p2: Individual) -> Individual:
            child = np.full(size, -1)
            child[start:end + 1] = p1.genes[start:end + 1]

            pos = (end + 1) % size
            for gene in p2.genes:
                if gene not in child:
                    child[pos] = gene
                    pos = (pos + 1) % size
            return Individual(child)

        return ox(parent1, parent2), ox(parent2, parent1)


class QueenGA:
    """
    Genetic Algorithm for solving the N-Queens problem.

    Attributes:
        n (int): Number of queens (board size).
        pop_size (int): Population size.
        mutation_rate (float): Probability of mutation.
        max_iter (int): Maximum number of generations.
        tournament_size (int): Number of individuals in tournament selection.
        population (List[Individual]): Current population.
    """

    def __init__(
        self,
        n: int = 8,
        pop_size: int = 50,
        mutation_rate: float = 0.1,
        max_iter: int = 1000,
        seed: Optional[int] = None,
        tournament_size: int = 3
    ) -> None:
        """
        Initialize the Genetic Algorithm.

        Args:
            n (int): Number of queens.
            pop_size (int): Population size.
            mutation_rate (float): Probability of mutation.
            max_iter (int): Maximum number of generations.
            seed (Optional[int]): Seed for reproducibility.
            tournament_size (int): Number of candidates in tournament selection.
        """
        if seed is not None:
            np.random.seed(seed)

        self.n: int = n
        self.pop_size: int = pop_size
        self.mutation_rate: float = mutation_rate
        self.max_iter: int = max_iter
        self.tournament_size: int = tournament_size
        self.population: List[Individual] = self._initialize_population()

    def _initialize_population(self) -> List[Individual]:
        """
        Create the initial population of random permutations.

        Returns:
            List[Individual]: List of individuals.
        """
        return [Individual(np.random.permutation(self.n) + 1) for _ in range(self.pop_size)]

    def _select_parents(self) -> List[Individual]:
        """
        Select individuals for reproduction using tournament selection.

        Returns:
            List[Individual]: Selected parent individuals.
        """
        selected: List[Individual] = []
        for _ in range(self.pop_size):
            tournament = np.random.choice(
                self.population, self.tournament_size, replace=False)
            winner = max(tournament, key=lambda ind: ind.fitness_score)
            selected.append(winner)
        return selected

    def _recombine(self, parents: List[Individual]) -> List[Individual]:
        """
        Recombine parent individuals to produce offspring.

        Args:
            parents (List[Individual]): List of selected parents.

        Returns:
            List[Individual]: Offspring population.
        """
        offsprings: List[Individual] = []
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = Individual.crossover(parents[i], parents[i + 1])
            offsprings.extend([child1, child2])
        if len(offsprings) < self.pop_size:
            offsprings.append(parents[-1])
        return offsprings

    def _mutate_population(self, population: List[Individual]) -> List[Individual]:
        """
        Apply mutation across the population.

        Args:
            population (List[Individual]): List of individuals.

        Returns:
            List[Individual]: Mutated individuals.
        """
        return [ind.mutate(self.mutation_rate) for ind in population]

    def run(self) -> Tuple[List[float], Individual]:
        """
        Execute the genetic algorithm loop.

        Returns:
            Tuple[List[float], Individual]: Best fitness over generations and the best individual.
        """
        best_fitness_history: List[float] = []
        best_individual: Individual = max(
            self.population, key=lambda x: x.fitness_score)

        for _ in range(self.max_iter):
            elite: Individual = best_individual

            parents: List[Individual] = self._select_parents()
            offsprings: List[Individual] = self._recombine(parents)
            self.population = self._mutate_population(offsprings)

            # Elitism: Replace worst with elite
            worst_index: int = min(
                range(self.pop_size), key=lambda i: self.population[i].fitness_score)
            self.population[worst_index] = elite

            current_best: Individual = max(
                self.population, key=lambda x: x.fitness_score)
            best_fitness_history.append(current_best.fitness_score)

            if current_best.fitness_score > best_individual.fitness_score:
                best_individual = current_best

            # Stop if perfect solution found
            if best_individual.fitness_score == (self.n * (self.n - 1)) // 2:
                break

        return best_fitness_history, best_individual

    def plot_fitness(self, fitness_history: List[float]) -> None:
        """
        Plot the best fitness score over generations.

        Args:
            fitness_history (List[float]): Fitness scores by generation.
        """
        plt.plot(fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('N-Queens Genetic Algorithm')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ga = QueenGA(n=8, pop_size=50, mutation_rate=0.1, max_iter=1000, seed=42)
    history, best = ga.run()
    ga.plot_fitness(history)

    print("Best solution (queen positions):", best.genes)
    print("Fitness score:", best.fitness_score)
