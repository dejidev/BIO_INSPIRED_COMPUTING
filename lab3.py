import numpy as np
import random
import matplotlib.pyplot as plt

# Distance matrix for cities A to H
distance_matrix = [
    [0, 20, 30, 25, 12, 33, 44, 57],  # A
    [22, 0, 19, 20, 20, 29, 43, 45],  # B
    [28, 19, 0, 17, 38, 48, 55, 60],  # C
    [25, 20, 19, 0, 28, 35, 40, 55],  # D
    [12, 18, 34, 25, 0, 21, 30, 40],  # E
    [35, 25, 45, 30, 20, 0, 25, 39],  # F
    [47, 39, 50, 35, 28, 20, 0, 28],  # G
    [60, 38, 54, 50, 33, 40, 25, 0]   # H
]

city_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Calculate total tour distance
def calculate_fitness(tour):
    total_distance = 0
    tour = [0] + tour + [0]  # Start and end at city A (index 0)
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i]][tour[i+1]]
    return -total_distance  # We maximize negative distance => minimize actual distance

# Order Crossover (OX)
def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))

    # Copy segment from parent1
    child[start:end+1] = parent1[start:end+1]

    # Fill remaining positions with genes from parent2
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    return child

# Swap Mutation
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Roulette Wheel Selection
def roulette_wheel_selection(population, fitnesses):
    max_fitness = max(fitnesses)
    adjusted_fitness = [f - min(fitnesses) + 1e-6 for f in fitnesses]
    total = sum(adjusted_fitness)
    probs = [f / total for f in adjusted_fitness]
    return population[np.random.choice(len(population), p=probs)]

# Initialize population
def initialize_population(pop_size):
    base = list(range(1, 8))  # Cities B to H (indices 1 to 7)
    return [random.sample(base, len(base)) for _ in range(pop_size)]

# Genetic Algorithm
def genetic_algorithm(pop_size=40, generations=40, crossover_prob=0.8, mutation_prob=0.25, threshold=0.05):
    population = initialize_population(pop_size)
    best_fitness_progress = []
    best_individual = None
    best_fitness = float('-inf')

    for gen in range(generations):
        fitnesses = [calculate_fitness(ind) for ind in population]
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        best_fitness_progress.append(-gen_best_fitness)  # Convert back to distance

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx]

        if gen > 0 and abs(best_fitness_progress[-1] - best_fitness_progress[-2]) < threshold:
            break

        new_population = []
        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)

            if random.random() < crossover_prob:
                offspring1 = order_crossover(parent1, parent2)
                offspring2 = order_crossover(parent2, parent1)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]

            new_population.append(mutate(offspring1, mutation_prob))
            if len(new_population) < pop_size:
                new_population.append(mutate(offspring2, mutation_prob))

        population = new_population

    # Format final tour
    final_tour = [city_names[i] for i in best_individual]
    full_tour = ['A'] + final_tour + ['A']
    return full_tour, best_fitness_progress

# Task iii–iv: Run GA with default population size 40
default_tour, default_fitness = genetic_algorithm()

print("Best Tour (Population Size 40):", " → ".join(default_tour))
print("Total Distance:", default_fitness[-1])

# Plot Fitness vs Generation
plt.plot(default_fitness, label="Population Size 40")
plt.title("Fitness vs Generation (Population Size 40)")
plt.xlabel("Generation")
plt.ylabel("Total Distance")
plt.grid(True)
plt.legend()
plt.show()

# Task v: Repeat with different population sizes
population_sizes = [20, 30, 40, 50, 60]
results = {}

for size in population_sizes:
    print(f"Running GA with Population Size {size}...")
    _, progress = genetic_algorithm(pop_size=size)
    results[size] = progress

# Plot all together
plt.figure(figsize=(10, 6))
for size in population_sizes:
    plt.plot(results[size], label=f"Pop Size {size}")
plt.title("Fitness vs Generation (Varying Population Sizes)")
plt.xlabel("Generation")
plt.ylabel("Total Distance")
plt.legend()
plt.grid(True)
plt.show()