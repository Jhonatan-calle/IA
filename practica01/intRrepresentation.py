import random

from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", int, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("individual", tools.initIterate, creator.Individual, 
                 lambda: random.randint(0, 100000))

toolbox.register("population",tools.initRepeat,list,toolbox.individual)


def distance(individual):
    return abs(individual - 2025),


toolbox.register("evaluate",distance)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) 
toolbox.register("select", tools.selTournament, tournsize=3) 



def main():
    random.seed(42)
    pop = toolbox.population(n=50) # Población inicial
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

if __name__ == "__main__":
        main()
