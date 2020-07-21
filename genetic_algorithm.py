"""
Genetic Algorithm
"""
import random


class Individual(object):

    def __init__(self, individual_size, geneSet, optimum, randomBuilder=True):
        self.individual_size = individual_size
        self.geneSet = geneSet
        self.optimum = optimum
        self.chromosome = []

        if randomBuilder:
            self.buildIndividual()
            self.fitness = self.getFitness()

    def buildIndividual(self):
        for _ in range(self.individual_size):
            self.chromosome.append(random.sample(self.geneSet, k=1)[0])

    def getFitness(self):
        fitness = 0
        for i in range(self.individual_size):
            if self.optimum[i] == self.chromosome[i]:
                fitness += 1

        return fitness

    def crossover(self, indv):
        assert isinstance(
            indv, Individual), 'the passed argument should be of type Individual'

        child = Individual(self.individual_size, self.geneSet,
                           self.optimum, randomBuilder=False)

        for gene_parent1, gene_parent2 in zip(self.chromosome, indv.chromosome):
            prob = random.random()

            if prob < 0.50:
                child.chromosome.append(gene_parent1)
            else:
                child.chromosome.append(gene_parent2)

        child.fitness = child.getFitness()

        return child

    def mutation(self, mutation_degree=0.10):
        num_mutations = int(self.individual_size * mutation_degree)

        for _ in range(num_mutations):
            ind = random.randrange(self.individual_size)
            self.chromosome[ind] = random.sample(self.geneSet, k=1)[0]

        self.fitness = self.getFitness()


class GeneticEngine(object):

    def __init__(self, population_size, individual_size, geneSet, optimum):
        assert len(
            optimum) == individual_size, 'individual_size and size of optimum should be same'

        self.population_size = population_size
        self.individual_size = individual_size
        self.geneSet = geneSet
        self.optimum = optimum
        self.population = []

        self.initializePopulation()

    def initializePopulation(self):
        for _ in range(self.population_size):
            individual = Individual(
                self.individual_size, self.geneSet, self.optimum)
            self.population.append(individual)

    def naturalSelection(self, max_generations=int(1e3), elitism_ratio=0.1, mutation_degree=0.1, verbose=True):

        for generation in range(max_generations):

            self.population = sorted(
                self.population, key=lambda ind: ind.fitness, reverse=True)

            top_individuals = self.population[:int(
                self.population_size * elitism_ratio)]

            new_population = []
            new_population.extend(top_individuals)

            ind = int(self.population_size * (1. - elitism_ratio))
            mating_boundary = int(self.population_size * 0.5)

            for _ in range(ind):
                parent1 = random.choice(self.population[: mating_boundary])
                parent2 = random.choice(self.population[: mating_boundary])
                child = parent1.crossover(parent2)

                if random.randrange(2) == 1 and mutation_degree > 0.:
                    child.mutation(mutation_degree)

                new_population.append(child)

            self.population = new_population

            if verbose:
                print('Generation {} --> best_indv: {}, fitness: {}'.format(generation +
                                                                            1, ' '.join(self.population[0].chromosome), self.population[0].fitness))

            if self.population[0].chromosome == self.optimum:
                print('\nproblem converged to optimum in {} generations'.format(
                    generation+1))
                break
        else:
            print("\nproblem didn't converged to optimum, try for more generations.")
            print('\nlast generation best individual -> {}'.format(
                ' '.join(self.population[0].chromosome)))
            print('fitness -> {}'.format(self.population[0].fitness))


if __name__ == '__main__':

    GENE_SET = set(
        list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    TARGET = list('Genetic Algorithm')

    POPULATION_SIZE = 100
    INDIVIDUAL_SIZE = len(TARGET)

    print('\ntarget := {}'.format(' '.join(TARGET)))

    engine = GeneticEngine(POPULATION_SIZE, INDIVIDUAL_SIZE, GENE_SET, TARGET)
    engine.naturalSelection()
