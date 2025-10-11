import random

from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax) # type: ignore

toolbox = base.Toolbox()

toolbox.register("dijito", random.randint, 0, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.dijito, 5) # type: ignore
toolbox.register("population",tools.initRepeat,list,toolbox.individual) # type: ignore


def distance(individual):
    value = int("".join(map(str, individual))) 
    return abs(value - 2025),


toolbox.register("evaluate",distance)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=3) 

# Estadísticas: mejor individuo por generación
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("best", min)  # Min

def main():
    random.seed(42)
    pop = toolbox.population(n=50) # Población inicial # type: ignore
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=90, stats=stats, verbose=True)

if __name__ == "__main__":
        main()
