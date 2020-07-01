"""
Author: Phalguni Rathod
Msc AI
R00183770
"""

import random, time
from Individual import *
import sys

myStudentNum = 183770 # Replace 12345 with your student number
random.seed(myStudentNum)


class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, _config):
        """
        Parameters and general variables
        """

        # Student-defined: Added a new parameter _config to take the configuration from user for executing it.
        self.config = _config
        self.choice = 1  # Student-defined: It chooses between Random(1) or Heuristic(2) Initialization
        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        # Student-defined: list to store the new pool of fittest parents among all via SUS
        self.fit_parents_list = []

        # This helps us to choose between Random/Heuristic as per the config given by user
        if self.config in range(1, 7):
            self.choice = 1
            if self.config == 1 or self.config == 2:
                # Over ride the popSize & mutation Rate for config 1 and keep it as given in assignment
                self.popSize = 100
                self.mutationRate = 0.1
        elif self.config == 7 or self.config == 8:
            self.choice = 2


        self.readInstance()
        self.initPopulation()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data, self.choice)  # Passing choice as paramter to choose between Random/Heurostic
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print("Best initial sol: ", self.best.getFitness())

    def updateBest(self, candidate):
        if self.best is None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            resFile.write("Iteration: " + str(self.iteration)+ "Best: "+ str(self.best.getFitness()))
            print ("Iteration: ", self.iteration, "Best: ", self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[random.randint(0, self.popSize-1)]
        indB = self.matingPool[random.randint(0, self.popSize-1)]
        return [indA, indB]

    #Student defined function
    def minimization(self):
        """
        This function creates a dictionary of all individuals with their minimized fitness.
        And keeps it continuous so, it becomes easy to find an individual when marker falls in certain range.
        :return:
        fit_sum: sum of minimized fitness of all individuals
        par_fit{}: dictionary created
        """

        fit_sum = 0  # Stores Sum of All fitness values
        par_fit = {}  # A dictionary to Key: Individuals(obj) & Value: Fitness

        for i in range(self.popSize):
            # Updating K-V pair by inversing fitness & adding to fit_sum to keep it continuous
            inv_fit = 1 / self.population[i].getFitness()
            par_fit[self.population[i]] = fit_sum + inv_fit
            fit_sum += inv_fit  # Summing up the inversed fitness

        #This is done for normalization
        for i in range(self.popSize):
            par_fit[self.population[i]] = par_fit[self.population[i]]/fit_sum

        return [fit_sum, par_fit]

    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        fit_sum, par_fit = self.minimization() # Calling above defined function to get total minimized fitness & par_fit dictionary

        no_of_parents = self.popSize  # It is "N" and I have assumed it to popSize, SO, more population more smaller distance
        dist = par_fit.get(list(par_fit.keys())[-1])/no_of_parents  # dist is "P" = F/N
        marker = random.uniform(0.01, dist)  # Taking a random value between (0,P) as start point of Marker/Ruler
        prev = 0  # to compare if marker value lie in a chromosome

        marker_list = [marker]  # A list to save all the marker pointers
        for i in range(no_of_parents):  # Looping till no_of_parents as we are creating that many marker pointers
            marker += dist  # Creating next marker pointer
            marker_list.append(marker)  # Adding it to list

        # Checking to which individual each marker pointer is pointing to & adding it to fit parent list
        for marker_i in marker_list: # Taking a single element
            for key, val in par_fit.items():  # Take Key(Individual Obj), Value(Fitness) from each item
                if marker_i > prev and marker_i <= val:  # Check if Marker pointer lies in this Individual
                    self.fit_parents_list.append(key)  # If it points to it,then add the Individual to Fit_parent_list
                    break  # Break when we find where the individual to which marker points
                prev = val  # Save the current fitness value to use it form range with next fitness value

        # Select 2 parents at random from Fit Parent List
        indA = self.fit_parents_list[random.randint(0, len(self.fit_parents_list) - 1)]
        indB = self.fit_parents_list[random.randint(0, len(self.fit_parents_list) - 1)]

        return [indA, indB]

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        fixed_pos = random.sample(range(0, indA.genSize), indA.genSize // 2)  # Fixed the random positions to be as it is. Keeping N/2 positions fixed.

        child_1 = [None] * indB.genSize  # Creating a Child List of None to update it with the upcoming expected crossovered list

        for index in fixed_pos:
            child_1[index] = indA.genes[index]  # Preserving same values at fixed positions

        for parent_item in range(len(indB.genes)):  # Taking element position from Parent B for putting in Child 1
            if indB.genes[parent_item] not in child_1:  # Check if element of Parent A is not in Child 1
                for child_place in range(len(child_1)):  # Run loop for each location of Child 1
                    if not child_1[child_place]:  # Check if Child 1 is not None (If it's empty)
                        child_1[child_place] = indB.genes[parent_item]  # Put the element from Parent B in Child 1
                        break  # To Avoid overiding the value
        child_obj = indA.copy()  # create an object copy of any parent
        child_obj.setGene(child_1) # Reset/Change the gene with the new Child
        return child_obj  # Returning the Cross-overed child for further processing

    # Verified
    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation - Done
        """
        # Taking 2 random points within genes
        pointA = random.randint(0, self.genSize - 1)
        pointB = random.randint(0, self.genSize - 1)

        # Finding mini as index 1 (start) & Finding max as index 2 (end)
        indexA = min(pointA, pointB)
        indexB = max(pointA, pointB)

        child_1 = indB.copy()  # Making a copy of parent b in child 1 and this is modified later

        # Creating Child Lists to work
        child_1_list = [None] * indB.genSize
        child_2_list = [None] * indA.genSize

        # Putting a section of Parent B in Child List 1 & of Parent A in Child list 2
        # These slices work as sublists to compare and get corresponding element value if repetitions occur
        child_1_list[indexA:indexB + 1] = indB.genes[indexA:indexB + 1]
        child_2_list[indexA:indexB + 1] = indA.genes[indexA:indexB + 1]

        for i in range(0, indB.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                continue  # Do not do anything if i is in between sublisted/sliced portion
            if indA.genes[i] not in child_1_list:  # Check if element of Parent is in Child 1 or not
                child_1_list[i] = indA.genes[i]  # If not, then put it

        for i in range(0, indA.genSize):
            if child_1_list[i] is not None:
                continue  # If Child 1 is not empty skip it.
            val = indA.genes[i]  # Take current value of Parent A and save for further calculation
            while True:
                if val in child_1_list[indexA:indexB + 1]:  # check if the value exist in sublist
                    ind = child_1_list.index(val)  # If it exist in sublist then find its index
                    if child_2_list[ind] in child_1_list[indexA:indexB + 1]:  # See if the at given index in other sublist(child 2 sublist), the child2 element is again in child 1 sublist or not
                        val = child_2_list[ind]  # If it does exist then save its value & the loop will repeat until it gets a value that is not already in child 1
                        continue
                    else:
                        val = child_2_list[ind]  # Save when it gets a value which it doesn't already exist in Child 1
                        break
            child_1_list[i] = val  # Put the value in Child 1 List
        child_1.setGene(child_1_list)  # Update the Child with new child List calculated above
        return child_1  # return the object for further processing

    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation
        """
        if random.random() > self.mutationRate:  # Chances of having mutation is dependent on mutation Rate
            ind.computeFitness()  # Compute the fitness of passed individual
            self.updateBest(ind)  # Update the best
            return ind  # return the object
        index1, index2 = random.sample(range(0, ind.genSize), 2)  # Choose any 2 index at random from given Individual
        ind.genes[index1], ind.genes[index2] = ind.genes[index2], ind.genes[index1]  # Swap element at given indexes
        ind.computeFitness()  # Compute the fitness
        self.updateBest(ind) # update it if it's the best
        return ind

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        if random.random() > self.mutationRate:  # Chances of having mutation is dependent on mutation Rate
            ind.computeFitness()  # Compute the fitness
            self.updateBest(ind)  # Update it if it's the best
            return ind  # Return the object
        index1, index2 = random.sample(range(0, ind.genSize), 2)  # Select any 2 index at random from given Individual
        start = min(index1, index2) # Take min as start & max as End
        end = max(index1, index2)
        ind.genes[start:end + 1] = ind.genes[start:end + 1][::-1] # Reverse the sliced part of Individual & put it back
        ind.computeFitness()  # Compute the fitness
        self.updateBest(ind)  # Update the best Fitness

        return ind

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux

        individual = indA.copy()
        individual.setGene(child)
        return individual

    def mutation(self, ind):
        """
        Mutate an individual by swapping two cities with certain probability (i.e., mutation rate)
        """

        if random.random() > self.mutationRate:
            ind.computeFitness()
            self.updateBest(ind)
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)
        return ind

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append(ind_i.copy())

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            if self.config == 1:
                indA, indB = self.randomSelection()
                child = self.uniformCrossover(indA, indB)
                child = self.inversionMutation(child)
            if self.config == 2:
                indA, indB = self.randomSelection()
                child = self.pmxCrossover(indA,indB)
                child = self.reciprocalExchangeMutation(child)
            if self.config == 3:
                indA, indB = self.stochasticUniversalSampling()
                child = self.uniformCrossover(indA, indB)
                child = self.reciprocalExchangeMutation(child)
            if self.config == 4 or self.config == 7:
                indA, indB = self.stochasticUniversalSampling()
                child = self.pmxCrossover(indA, indB)
                child = self.reciprocalExchangeMutation(child)
            if self.config == 5:
                indA, indB = self.stochasticUniversalSampling()
                child = self.pmxCrossover(indA, indB)
                child = self.inversionMutation(child)
            if self.config == 6 or self.config == 8:
                indA, indB = self.stochasticUniversalSampling()
                child = self.uniformCrossover(indA, indB)
                child = self.inversionMutation(child)
            if child is None:
                print("None coming from mutation")
                continue
            self.population[i] = child  # Update the new child in population

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        start_time = time.process_time()

        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print("Population: ", self.popSize)
        print("Mutation Rate: ", self.mutationRate)
        print("Total iterations: ", self.iteration)
        print("Best Solution: ", self.best.getFitness())
        print("Time Taken: ", (time.process_time() - start_time)/60)

        resFile.write("Population: " + str(self.popSize) + "\n"+"Total iterations: " + str(self.iteration)+ "\n"+ "Best Solution: "+str(self.best.getFitness()) +
                      "\n" + "Time Taken: " + str((time.process_time() - start_time)/60) + "\n")


if len(sys.argv) < 2:
    print("Error - Incorrect input")
    print("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)


problem_file = sys.argv[1]
config = int(sys.argv[2])
iterate = 500  #default Iterations
pop = 100  # When Random Selection Running
if config ==7 or config == 8: # Keeping pop as 50 for heuristic configs
    pop = 50  # Heuristic
mutRate = 0.1 # default Rate
for i in range(6):
    resFile = open(problem_file+"_Config_"+str(config)+".txt", "a") # Creating a file to save output result
    # arguments = filename, pop_size, mutation_rate, max_iter, config(1-8)
    resFile.write("Times: " + str(i+1)+"\n")
    print("Times: ", i+1) # Shows the ongoing run for current config

    if i == 3: # change the default values to experiment
        pop = 200 # Random
        if config ==7 or config ==8:
            pop = 100  # Heuristic
        iterate = 500
        mutRate = 0.5

    ga = BasicTSP(sys.argv[1], pop, mutRate, iterate, config)
    ga.search()
    if not (config == 7 or config ==8):
        iterate += 100  # It will skip incrementing iteration if Heuristic config is selected
    resFile.close()
