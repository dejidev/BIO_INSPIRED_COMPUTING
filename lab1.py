import numpy as np
import random
import matplotlib.pyplot as plt

# Define the fitness function
def fitness_function(x1, x2, x3):
    return 2*x1*x2*x3 - 4*x1*x3 - 2*x2*x3 + x1**2 + x2**2 + x3**2 - 2*x1 - 4*x2 + 4*x3

# Decode binary string to real value within bounds
def decode(chromosome, bounds):
    decoded = []
    split = len(chromosome) // 3
    for i, (lower, upper) in enumerate(bounds):
        substring = chromosome[i*split:(i+1)*split]
        integer_value = int("".join(str(bit) for bit in substring), 2)
        max_int = 2**split - 1
        value = lower + (upper - lower) * integer_value / max_int
        decoded.append(value)
    return decoded

# Initialize a random population
def initialize_population(pop_size, chromosome_length):
    return [np.random.randint(0, 2, chromosome_length).tolist() for _ in range(pop_size)]

# Roulette Wheel Selection (handles negative fitnesses)
def roulette_wheel_selection(population, fitnesses):
    min_fit = min(fitnesses)
    if min_fit < 0:
        adjusted_fitnesses = [f - min_fit + 1e-6 for f in fitnesses]
    else:
        adjusted_fitnesses = fitnesses[:]

    total_fitness = sum(adjusted_fitnesses)
    probs = [f / total_fitness for f in adjusted_fitnesses]
    return population[np.random.choice(len(population), p=probs)]

# One-point crossover
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Mutation function (bit-flip)
def mutate(chromosome, mutation_rate):
    return [bit if random.random() > mutation_rate else 1 - bit for bit in chromosome]

# Genetic Algorithm
def genetic_algorithm(bounds, chromosome_bits, pop_size=50, generations=50, crossover_prob=0.8, mutation_prob=0.2, threshold=0.05):
    chromosome_length = chromosome_bits * 3
    population = initialize_population(pop_size, chromosome_length)
    best_fitness_progress = []

    best_solution = None
    best_fitness = -float('inf')

    for gen in range(generations):
        decoded_population = [decode(individual, bounds) for individual in population]
        fitnesses = [fitness_function(*individual) for individual in decoded_population]

        gen_best_fitness = max(fitnesses)
        best_fitness_progress.append(gen_best_fitness)

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = decoded_population[fitnesses.index(gen_best_fitness)]

        if gen > 0 and abs(best_fitness_progress[-1] - best_fitness_progress[-2]) < threshold:
            break

        new_population = []
        for _ in range(pop_size // 2):
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)

            if random.random() < crossover_prob:
                offspring1, offspring2 = one_point_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]

            new_population.append(mutate(offspring1, mutation_prob))
            new_population.append(mutate(offspring2, mutation_prob))

        population = new_population

    return best_solution, best_fitness, best_fitness_progress

# Set the bounds for x1, x2, x3
bounds = [(10, 90), (0, 90), (-20, 60)]
chromosome_bits = 8  # Number of bits for each variable can be 8, 12, or 15
# Run the GA
best_sol, best_fit, fitness_progress = genetic_algorithm(bounds, chromosome_bits)

# Output results
print("Best Solution (x1, x2, x3):", best_sol)
print("Best Fitness:", best_fit)

# Optional: plot the progress
plt.plot(fitness_progress)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title(f"Fitness vs Generation ({chromosome_bits}-bit)")
plt.grid(True)
plt.show()