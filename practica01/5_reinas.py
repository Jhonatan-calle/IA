import random
from deap import base, creator, tools, algorithms


BOARD_DEMENTIOS = 10

# --- Crear clases ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore

toolbox = base.Toolbox()

toolbox.register("coordenada", lambda: (random.randint(0, BOARD_DEMENTIOS-1), random.randint(0, BOARD_DEMENTIOS-1)))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.coordenada, BOARD_DEMENTIOS)  # type: ignore
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore


def myFitness(individual):
    conflicts = 0
    n = len(individual)
    for i in range(n):
        x1, y1 = individual[i]
        for j in range(i + 1, n):
            x2, y2 = individual[j]
            if x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2):
                conflicts += 1
    return (conflicts,)


def mutate(individual, board_size= BOARD_DEMENTIOS, indpb=0.1):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = (
                random.randint(0, board_size - 1),
                random.randint(0, board_size - 1),
            )
    return (individual,)


toolbox.register("evaluate", myFitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Registrar estadísticas de fitness
# Función que devuelve el individuo completo

stats = tools.Statistics(lambda ind: ind)  # registramos todo el individuo
stats.register("best", lambda inds: max(inds, key=lambda ind: ind.fitness.values[0]))


# --- Función para imprimir el tablero ---
def print_board(individual):
    board = [["." for _ in range(BOARD_DEMENTIOS)] for _ in range(BOARD_DEMENTIOS)]
    for (x, y) in individual:
        board[x][y] = "♛"
    for row in board:
        print(" ".join(row))
    print()


def main():
    random.seed(42)
    pop = toolbox.population(n=5)  # type: ignore
    hof = tools.HallOfFame(1)  # guarda el mejor individuo
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=6193, halloffame=hof, verbose=False)
    best = hof[0]
    print("Mejor individuo encontrado:", best)
    print("Fitness:", best.fitness.values[0])
    print("\nRepresentación en tablero:\n")
    print_board(best)

if __name__ == "__main__":
    main()
