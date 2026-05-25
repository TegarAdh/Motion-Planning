import numpy as np
import matplotlib.pyplot as plt
import random

# =========================
# PARAMETER ENVIRONMENT
# =========================
START = np.array([1, 3])
GOAL = np.array([97, 95])

MAP_SIZE = 100
NUM_WAYPOINTS = 5

POPULATION_SIZE = 50
GENERATIONS = 10
MUTATION_RATE = 0.2

# obstacle: (x, y, radius)
OBSTACLES = [
    (30, 40, 10),
    (60, 60, 12),
    (70, 20, 8)
]

# =========================
# FITNESS FUNCTION
# =========================
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def collision(point):
    for ox, oy, r in OBSTACLES:
        if distance(point, np.array([ox, oy])) <= r:
            return True
    return False

def path_length(path):
    total = 0

    for i in range(len(path) - 1):
        total += distance(path[i], path[i + 1])

    return total

def fitness(path):
    penalty = 0

    for point in path:
        if collision(point):
            penalty += 1000

    dist = path_length(path)

    return 1 / (dist + penalty + 1)

# =========================
# CREATE INDIVIDUAL
# =========================
def create_individual():
    path = [START]

    for _ in range(NUM_WAYPOINTS):
        waypoint = np.array([
            random.uniform(0, MAP_SIZE),
            random.uniform(0, MAP_SIZE)
        ])
        path.append(waypoint)

    path.append(GOAL)

    return path

# =========================
# CREATE POPULATION
# =========================
def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

# =========================
# SELECTION
# =========================
def selection(population):
    population = sorted(
        population,
        key=lambda x: fitness(x),
        reverse=True
    )

    return population[:2]

# =========================
# CROSSOVER
# =========================
def crossover(parent1, parent2):
    child = [START]

    split = random.randint(1, NUM_WAYPOINTS)

    child += parent1[1:split]
    child += parent2[split:-1]

    child.append(GOAL)

    return child

# =========================
# MUTATION
# =========================
def mutation(path):
    for i in range(1, len(path) - 1):
        if random.random() < MUTATION_RATE:
            path[i] = np.array([
                random.uniform(0, MAP_SIZE),
                random.uniform(0, MAP_SIZE)
            ])

    return path

# =========================
# GENETIC ALGORITHM
# =========================
population = create_population()

best_path = None
best_fitness = 0

for generation in range(GENERATIONS):

    parent1, parent2 = selection(population)

    new_population = [parent1, parent2]

    while len(new_population) < POPULATION_SIZE:
        child = crossover(parent1, parent2)
        child = mutation(child)

        new_population.append(child)

    population = new_population

    current_best = selection(population)[0]
    current_fitness = fitness(current_best)

    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_path = current_best

    print(
        f"Generation {generation + 1} | "
        f"Fitness: {best_fitness:.6f}"
    )

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(8, 8))

# draw obstacle
for ox, oy, r in OBSTACLES:
    circle = plt.Circle((ox, oy), r, color='red')
    plt.gca().add_patch(circle)

# draw best path
x = [p[0] for p in best_path]
y = [p[1] for p in best_path]

plt.plot(x, y, '-o', linewidth=2, label='Best Path')

# draw start and goal
plt.scatter(START[0], START[1], c='green', s=100, label='Start')
plt.scatter(GOAL[0], GOAL[1], c='blue', s=100, label='Goal')

plt.xlim(0, MAP_SIZE)
plt.ylim(0, MAP_SIZE)

plt.grid(True)
plt.legend()

plt.title("Motion Planning using Genetic Algorithm")

plt.show()