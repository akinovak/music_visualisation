"""
Cilj rada je da se sto pribliznije moguce
vizualizuju slicnosti (odnosno razlike)
muzickih zanrova, koriscenjem genetskog
algoritma.

Podaci korisceni u radu preuzeti su sa:
https://www.researchgate.net/figure/Relations-between-liking-for-musical-genres-in-the-N332-participants-screened-for-the_fig1_330110820
"""
import random
import numpy 
import math 
import pandas as pd
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plot
import warnings

warnings.filterwarnings("ignore")


class Chromosome:

    def __init__(self, gene, fitness):

        self.gene = gene
        self.fitness = fitness
    
    def __str__(self):
        return "{} -> {}".format(self.gene, self.fitness)

class GeneticAlgorithm:

    def __init__(self, matrix):

        self.matrix = matrix       
        self.generation_size = 5000            
        self.chromosome_size = len(matrix)     
        self.reproduction_size = 2000              
        self.max_iterations = 100            
        self.mutation_rate = 0.2          
        self.tournament_size = 50             
        
        self.selection_type = 'tournament'      
    def calculate_fitness(self, gene):

    	sum_fitness = 0

    	for i in range(0, len(self.matrix)-1):

    		for j in range(i+1, len(self.matrix)):

    			degree = gene[i] - gene[j]
    			sum_fitness = sum_fitness + (self.matrix[i][j] - math.cos(degree*math.pi/180))**2

    	return sum_fitness

    def initial_population(self):

    	init_population = []

    	for i in range(self.generation_size):

    		gene = []

    		for j in range(self.chromosome_size):

    			selected_value = random.randint(0,360)
    			gene.append(selected_value)

    		fitness = self.calculate_fitness(gene)
    		new_chromosome = Chromosome(gene, fitness)

    		init_population.append(new_chromosome)

    	return init_population


    def selection(self, chromosomes):

        selected = []
        
        for i in range(self.reproduction_size):
            if self.selection_type == 'roulette':
                selected.append(self.roulette_selection(chromosomes))
            elif self.selection_type == 'tournament':
                selected.append(self.tournament_selection(chromosomes))
          
        return selected
   
    def roulette_selection(self, chromosomes):
        
        total_fitness = sum([chromosome.fitness for chromosome in chromosomes])
         
        selected_value = random.randrange(0, int(total_fitness))
        
        current_sum = 0
        for i in range(self.generation_size):
            current_sum += chromosomes[i].fitness

            if current_sum > selected_value:
                return chromosomes[i]
	
    def tournament_selection(self, chromosomes):
        
        selected = random.sample(chromosomes, self.tournament_size)
        
        winner = min(selected, key = lambda x: x.fitness)
        
        return winner

        
    def mutate(self, gene):

        random_value = random.random()
        
        if random_value < self.mutation_rate:
            
            random_index = random.randrange(self.chromosome_size)
            
            while True:
                
                new_value = random.randint(0, 360)
                
                if gene[random_index] != new_value:
                    break
                    
            gene[random_index] = new_value
            
        return gene

    def create_generation(self, chromosomes):

        generation = []
        generation_size = 0
        
        while generation_size < self.generation_size:
           		
           	[parent1, parent2] = random.sample(chromosomes, 2)

           	child1_code, child2_code = self.crossover(parent1, parent2)

           	child1_code = self.mutate(child1_code)
           	child2_code = self.mutate(child2_code)

           	child1 = Chromosome(child1_code, self.calculate_fitness(child1_code))
           	child2 = Chromosome(child2_code, self.calculate_fitness(child2_code))

           	generation.append(child1)
           	generation.append(child2)

           	generation_size += 2
            
        return generation

    def crossover(self, parent1, parent2):
        
        break_point = random.randrange(1, self.chromosome_size)
        
        child1 = parent1.gene[:break_point] + parent2.gene[break_point:]
        child2 = parent2.gene[:break_point] + parent1.gene[break_point:]
        
        return (child1, child2)
        
    def optimize(self):

    	best_result = 10000

    	population = self.initial_population()

    	br = 0

    	for i in range(0, self.max_iterations):

    		selected = self.selection(population)
    		population = self.create_generation(selected)
    		global_best_chromosome = min(population, key=lambda x: x.fitness)

    		if global_best_chromosome.fitness < best_result:
    			best_result = global_best_chromosome.fitness
    			br = 0
    			print("{} -> {}".format(i, best_result))

    		else:
    			br += 1
    			print(i)

    		if br == 10:
    			break

    		if global_best_chromosome.fitness < 10:
    			break

    	return global_best_chromosome
            
def main():

	matrix = pd.read_csv("musical-genres.csv", sep = ",", header=None)
	matrix = numpy.array(matrix)

	genetic_algorithm = GeneticAlgorithm(matrix)
    
	result = genetic_algorithm.optimize()
	
	print('Result: {}'.format(result))

	radian = numpy.zeros(len(result.gene))
	radius = 1

	labels = ["Metal", "Blues", "Classical", "Contemporary", "Electro", "Folk",
			  "Jazz", "Pop", "Rap", "Religious", "Rock", "Soul", "Variety", "World"]

	for i in range(0, len(labels)):
		labels[i] = labels[i] + ' ' + str(result.gene[i])


	fig = plot.figure()	
	fig.set_size_inches(18.5, 10.5, forward=True)
	fig.canvas.set_window_title('Visual representation of correlation between music genres')

	plot.clf()
	plot.title('Music genres')
	ax = fig.add_subplot(111, projection='polar')
	ax.set_yticklabels([])
	ax.set_theta_zero_location('W')
	ax.set_theta_direction(-1)
	ax.grid(False)

	for r in range(0, len(result.gene)):

		radian[r] = result.gene[r]*math.pi/180

	for i in range(0, len(radian)):

		plot.polar((0,radian[i]), (0,radius), label = labels[i], zorder = 3)

	ax.legend(loc = 'upper center', bbox_to_anchor = (1.45, 0.8), shadow = True, ncol = 1)
	plot.show()


if __name__ == "__main__":
	main()
