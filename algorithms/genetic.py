from random import choices, randint, randrange, random, sample
from typing import List, Callable, Tuple

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


def generate_genome(length):
    return choices([0, 1], k=length)


def generate_population(size, genome_length):
    return [generate_genome(genome_length) for i in range(size)]


def single_point_crossover(a, b):
    length = len(a)
    if length < 2:
        return a, b
    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome, num, probability):
    for i in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def population_fitness(population, fitness_func):
    return sum([fitness_func(genome) for genome in population])


def selection_pair(population, fitness_func):
    return sample(population=generate_weighted_distribution(population, fitness_func), k=2)


def generate_weighted_distribution(population, fitness_func):
    result = []
    for gene in population:
        result += [gene] * int(fitness_func(gene) + 1)
    return result


def sort_population(population, fitness_func):
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome):
    return "".join(map(str, genome))


def run_evolution(populate_func, fitness_func, fitness_limit, selection_func=selection_pair,
                  crossover_func=single_point_crossover, mutation_func=mutation, generation_limit=100,
                  printer=None):
    population = populate_func()

    i = 0
    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
        population = next_generation
    return population, i
