

"""
Basic TSP Example
file: Individual.py
"""

import random
import math


class Individual:
    def __init__(self, _size, _data,_choice):
        """
        Parameters and general variables
        """
        self.fitness    = 0
        self.genes      = []
        self.genSize    = _size
        self.data       = _data
        self.choice = _choice



        if self.choice == 1:
            self.genes = list(self.data.keys())

            for i in range(0, self.genSize):
                n1 = random.randint(0, self.genSize - 1)
                n2 = random.randint(0, self.genSize - 1)
                tmp = self.genes[n2]
                self.genes[n2] = self.genes[n1]
                self.genes[n1] = tmp

        elif self.choice == 2:
            solution, cost = self.insertion_heuristic1(self.data)
            self.genes = solution

    def setGene(self, genes):
        """
        Updating current choromosome
        """
        self.genes = []
        for gene_i in genes:
            self.genes.append(gene_i)

    def copy(self):
        """
        Creating a new individual
        """
        ind = Individual(self.genSize, self.data, self.choice)
        for i in range(0, self.genSize):
            ind.genes[i] = self.genes[i]
        ind.fitness = self.getFitness()
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt((d1[0]-d2[0])**2 + (d1[1]-d2[1])**2)

    def euclideanDistane(self, cityA, cityB):
        return round(math.sqrt((cityA[0] - cityB[0]) ** 2 + (cityA[1] - cityB[1]) ** 2))

    def insertion_heuristic1(self, instance):
        cities = list(instance.keys())
        cIndex = random.randint(0, len(instance) - 1)

        tCost = 0

        solution = [cities[cIndex]]

        del cities[cIndex]

        current_city = solution[0]
        while len(cities) > 0:
            bCity = cities[0]
            bCost = self.euclideanDistane(instance[current_city], instance[bCity])
            bIndex = 0
            for city_index in range(1, len(cities)):
                city = cities[city_index]
                cost = self.euclideanDistane(instance[current_city], instance[city])
                if bCost > cost:
                    bCost = cost
                    bCity = city
                    bIndex = city_index
            tCost += bCost
            current_city = bCity
            solution.append(current_city)
            del cities[bIndex]
        tCost += self.euclideanDistane(instance[current_city], instance[solution[0]])
        return solution, tCost

    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness    = self.euclideanDistance(self.genes[0], self.genes[len(self.genes)-1])
        for i in range(0, self.genSize-1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i+1])
