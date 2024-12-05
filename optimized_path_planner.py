import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from deap import base, creator, tools
from scipy.interpolate import splprep, splev, make_interp_spline

# Load cone positions from CSV file
df = pd.read_csv('/BrandsHatchLayout.csv') # Replace 'BrandsHatchLayout.csv' with the path to your CSV file
x_coords = df.iloc[:, 0].to_numpy()
y_coords = df.iloc[:, 1].to_numpy()

# Remove NaN values to ensure clean data
valid_indices = ~(np.isnan(x_coords) | np.isnan(y_coords))
x_coords = x_coords[valid_indices]
y_coords = y_coords[valid_indices]

# Split cones into inner and outer boundaries
x_inner = x_coords[157:]
y_inner = y_coords[157:]
x_outer = x_coords[:156]
y_outer = y_coords[:156]

# Define track width to calculate limits
track_width = 8

# Calculate the inner and outer track limits based on the cone positions
x_inner_limit = []
y_inner_limit = []
x_outer_limit = []
y_outer_limit = []

for i in range(len(x_inner)):
    dx = x_outer[i] - x_inner[i]
    dy = y_outer[i] - y_inner[i]
    dist = math.hypot(dx, dy)
    if dist == 0:
        dist = 1e-6  # Avoid division by zero
    scale = track_width / dist
    x_inner_limit.append(x_inner[i] + dx * scale)
    y_inner_limit.append(y_inner[i] + dy * scale)
    x_outer_limit.append(x_outer[i] - dx * scale)
    y_outer_limit.append(y_outer[i] - dy * scale)

# Compute a middle line between the inner and outer boundaries
middle_line_x = []
middle_line_y = []
half_length = int(len(x_coords) / 2)
for i in range(half_length - 1):
    index_outer = i
    index_inner = i + 1 + half_length
    if index_inner < len(x_coords):
        mid_x = (x_coords[index_outer] + x_coords[index_inner]) / 2
        mid_y = (y_coords[index_outer] + y_coords[index_inner]) / 2
        middle_line_x.append(mid_x)
        middle_line_y.append(mid_y)

# Create splines for smoother track representation
t_vals = np.linspace(0, len(middle_line_x) - 1, len(middle_line_x))
spline_x_inner = make_interp_spline(t_vals, x_inner_limit[:len(t_vals)], k=3)
spline_y_inner = make_interp_spline(t_vals, y_inner_limit[:len(t_vals)], k=3)
spline_x_outer = make_interp_spline(t_vals, x_outer_limit[:len(t_vals)], k=3)
spline_y_outer = make_interp_spline(t_vals, y_outer_limit[:len(t_vals)], k=3)
spline_x_middle = make_interp_spline(t_vals, middle_line_x, k=3)
spline_y_middle = make_interp_spline(t_vals, middle_line_y, k=3)

# Generate dense waypoints for the path
num_waypoints = 300
t_dense = np.linspace(0, len(middle_line_x) - 1, num_waypoints)

# Interpolate track limits and middle line at dense waypoints
dense_x = spline_x_middle(t_dense)
dense_y = spline_y_middle(t_dense)
dense_x_inner = spline_x_inner(t_dense)
dense_y_inner = spline_y_inner(t_dense)
dense_x_outer = spline_x_outer(t_dense)
dense_y_outer = spline_y_outer(t_dense)

# Define bounds for each waypoint based on track limits
bounds = []
for i in range(num_waypoints):
    x_min = min(dense_x_inner[i], dense_x_outer[i])
    x_max = max(dense_x_inner[i], dense_x_outer[i])
    y_min = min(dense_y_inner[i], dense_y_outer[i])
    y_max = max(dense_y_inner[i], dense_y_outer[i])
    bounds.append((x_min, x_max, y_min, y_max))

# Evolutionary Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    # Generate a random individual within the track bounds
    individual = []
    for i in range(num_waypoints):
        x_min, x_max, y_min, y_max = bounds[i]
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        individual.extend([x, y])
    return individual

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    # Evaluate path based on total length and boundary constraints
    x_vals = np.array(individual[::2])
    y_vals = np.array(individual[1::2])

    try:
        # Fit a spline to the individual's path
        tck, u = splprep([x_vals, y_vals], s=0)
        unew = np.linspace(0, 1.0, num_waypoints * 10)
        out = splev(unew, tck)

        # Calculate total path length
        dx = np.diff(out[0])
        dy = np.diff(out[1])
        distances = np.hypot(dx, dy)
        total_length = np.sum(distances)

        # Apply penalty for leaving track bounds
        penalty = 0
        for i in range(num_waypoints):
            x = individual[2*i]
            y = individual[2*i + 1]
            x_min, x_max, y_min, y_max = bounds[i]
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                penalty += 1000

        total_fitness = total_length + penalty

    except Exception:
        total_fitness = 1e6  # High fitness for invalid paths

    return (total_fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # Main function for the evolutionary algorithm
    random.seed(42)
    population_size = 1000
    num_generations = 250
    crossover_prob = 0.8
    mutation_prob = 0.25

    population = toolbox.population(n=population_size)

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Run the evolutionary algorithm
    for gen in range(1, num_generations + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                # Enforce bounds after mutation
                for i in range(num_waypoints):
                    x_min, x_max, y_min, y_max = bounds[i]
                    mutant[2 * i] = min(max(mutant[2 * i], x_min), x_max)
                    mutant[2 * i + 1] = min(max(mutant[2 * i + 1], y_min), y_max)
                del mutant.fitness.values

        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        # Log statistics
        record = stats.compile(population)
        print(f"Generation {gen}: {record}")

    # Return the best individual (shortest path)
    best_individual = tools.selBest(population, 1)[0]
    return best_individual

# Find the best path using the evolutionary algorithm
best_individual = main()

# Extract and visualize the shortest path
x_vals = np.array(best_individual[::2])
y_vals = np.array(best_individual[1::2])

# Fit a spline through the best path
tck, u = splprep([x_vals, y_vals], s=0)
unew = np.linspace(0, 1.0, num_waypoints * 10)
out = splev(unew, tck)
best_x_spline = out[0]
best_y_spline = out[1]

# Plot the results
plt.figure(figsize=(10, 10))
plt.plot(dense_x_outer, dense_y_outer, label='Outer Limit', color='black')
plt.plot(dense_x_inner, dense_y_inner, label='Inner Limit', color='black')
plt.scatter(x_coords, y_coords, marker='o', color='gray', label='Cones', s=10)
plt.plot(best_x_spline, best_y_spline, label='Shortest Path', color='red', linewidth=2)
plt.legend()
plt.title('Shortest Path using Evolutionary Algorithm')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()
