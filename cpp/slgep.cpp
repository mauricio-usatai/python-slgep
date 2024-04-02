#include <tuple>
#include <vector>
#include <math.h>
#include <stdexcept>
#include <stdlib.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Constants
#define H	    10		// Head length of the main program
#define T	    (H+1)	// Tail length of the main program   (the maximum arities of all functions is 2)
#define GSIZE 	2		// Number of ADFs
#define GH		3		// Head length of ADFs
#define GT		(GH+1)	// Tail length of ADFs
#define GNVARS	(GH+GT)	
#define NVARS	(H+T + GSIZE *(GH+GT))	// Chromosome length

#define MAX_SIBLING 20  			// The maximum sibling for each node
#define	LINK_LENGTH	(NVARS * 20)	// Add enough to save all necessary node
#define MAXINPUTS	1000			// Maximum input-output pairs of each problem

#define MAXIMUM_ELEMENTS 100		// MAXIMUM_ELEMENTS > function_num && MAXIMUM_ELEMENTS > terminal_num

// Structs
typedef struct {
	int gene[NVARS];
	double fitness;
} CHROMOSOME;

struct LINK_COMP {
	int value;
	int sibling_num;			
	LINK_COMP *siblings[MAX_SIBLING];
};

// Functions
double randval(double a, double b) {
	return a + (b - a) * rand() / (double)RAND_MAX;
}

// Classes
class SLGEP {
public:
	int POPSIZE;					// Size of the chromosome population

	int L_terminal;					// Start value of terminal symbol
	int L_input;					// Start value of input symbol
	int base_function_num;      	// {add, sub, mul, div, sin, cos, exp, log}
	int terminal_num;				// Current number of terminals
	int function_num;				// Total function numbers including the ADFs

	int generation;					// Initialize generation number
	std::vector<std::vector<double>> history; // The training error history

	CHROMOSOME best_chromosome;		// The chromosome with best fitness

	std::vector<std::vector<double>> inputs; // Dataset features
	std::vector<double> targets;			 // Dataset target

	SLGEP(int terminal_num) {
		this->terminal_num = terminal_num; // The number of features

		generation = 0;

		L_terminal = 10000;
		L_input = 20000;							
		base_function_num = 8;						
		function_num = base_function_num + GSIZE; 	
	}

	std::vector<std::vector<double>> fit(int generations) {
		if (this->generation == 0) {
			best_chromosome = population[0];
			// Initialize the history matrix
			history.push_back(std::vector<double>(POPSIZE));

			// Update initial chromosomes fitness
			for (int i = 0; i < POPSIZE; i++) {
				this->compute_chromosome_fitness(&population[i]);
				// Add to history
				history[0][i] = population[i].fitness;
				// Best chromosome
				if (population[i].fitness < best_chromosome.fitness) {
					best_chromosome = population[i];
				}
			}
		}
		
		int cur_generation = this->generation;
		// Start evolution process
		while (this->generation < cur_generation + generations) {
			this->generation++;
			// Generate a new generation
			its_a_new_generation();

			history.push_back(std::vector<double>(POPSIZE));
			// Select by comparing old and new generation fitness values
			for (int i = 0; i < POPSIZE; i++) {
				if (new_population[i].fitness < population[i].fitness) {
					population[i] = new_population[i];
				}
				// Best fitness value for this generation
				if (population[i].fitness < best_chromosome.fitness) {
					best_chromosome = population[i];
				}
				// Store the fitness values of the whole population after mutation and selection
				history[generation][i] = population[i].fitness;
			}
			// Check for termination condition
			// if(best_chromosome.fitness < 1e-4) break;
			// Print stats every 100 generations
			if(this->generation % 100 == 0) {
				printf("generation %d\t%g\n", generation, best_chromosome.fitness);
			}
		}
		return history;
	}

	std::vector<CHROMOSOME> get_best_chromosome() {
		std::vector<CHROMOSOME> best;
		best.push_back(this->best_chromosome);
		return best;
	}

	std::vector<CHROMOSOME> get_population() {
		return this->population;
	}

	int get_generation_number() {
		return this->generation;
	}

    void set_dataset(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
		int _size = X[0].size();
		if (_size != terminal_num) {
			throw std::invalid_argument(
				"Dimensions does not match. Expecting (N, " + std::to_string(terminal_num) + ") "
				"but got (N, " + std::to_string(_size) + ")"
			);
		}
        inputs = X;
        targets = y;
    }

	void generate_random_population(int POPSIZE_) {
		this->POPSIZE = POPSIZE_;

		population.resize(POPSIZE);
		new_population.resize(POPSIZE); // Resize new generations array

		int i, j, k;
		// Delimit main program and ADF headers and tails by
		// using different numbers
		for (i = 0; i < NVARS; i++) {
			if (i < H) gene_type_flag[i] = 0;
			else if (i < H + T)	gene_type_flag[i] = 1;
			else {
				j = i - H - T;
				if (j % (GH + GT) < GH)	gene_type_flag[i] = 2;
				else gene_type_flag[i] = 3;
			}
		}
		// Generate the inital population randomly
		for (i = 0; i < POPSIZE; i++) {
			for (k = 0; k < NVARS; k++) {
				rand_set_value(k, &population[i].gene[k]);
				population[i].fitness = 1e10;
			}
		}
	}

	double predict(std::vector<double> input, int gene[NVARS]) {
		training_input = input;

		decode_gene(gene);
		compute_rule(link_root);

		return current_value;
	}

private:
	std::vector<CHROMOSOME> population; 				 // The local population
	std::vector<CHROMOSOME> new_population; 			 // The generated population
	LINK_COMP *link_root, link_comp[LINK_LENGTH];		 //the whole expression tree
	LINK_COMP *sub_root[GSIZE], sub_comp[GSIZE][GNVARS]; //the sub expression tree

	int gene_type_flag[NVARS];							 // The type of each bit in the chromosome
	double sub_sibling_value[MAX_SIBLING];

	double t2;
	double current_value, sub_current_value;
	std::vector<double> training_input;

	double FQ;										// In the main heads of population, the proportion of bits being function symbols
	double function_freq[MAXIMUM_ELEMENTS];			// In the main parts of population, the frequency of each function symbol
	double terminal_freq[MAXIMUM_ELEMENTS];			// In the main parts of population, the frequency of each terminal symbol
	double terminal_probability[MAXIMUM_ELEMENTS];	// Store the selection probability of each terminal
	double function_probability[MAXIMUM_ELEMENTS];	// Store the selection probability of each function

	void compute_chromosome_fitness(CHROMOSOME *chromosome) {
		double prediction;
		double rmse = 0; // Root Mean Square error
		
		for (int i = 0; i < inputs.size(); i++) {
			prediction = this->predict(inputs[i], chromosome->gene);
			rmse += (targets[i] - prediction) * (targets[i] - prediction);
		}
		rmse = sqrt(rmse/inputs.size());
		if (rmse < 1e-4) rmse = 0;

		// Update chromosome fitness
		chromosome->fitness = rmse;
	}

	void its_a_new_generation() {
		int k, r1, r2;
		double CR, F;
		double change_vector[NVARS];

		update_probability();

		for(int i = 0; i < POPSIZE; i++) {
			new_population[i] = population[i];

			F = randval(0, 1);
			CR = randval(0,1);

			// Randomly chose two vector out of the population for mutation
			do {r1 = rand() % POPSIZE;} while(r1 == i); 			// Dont mutate the vector with itself
			do {r2 = rand() % POPSIZE;} while(r2 == r1 || r2 == i); // Need a third vector

			k = rand() % NVARS;

			for (int j = 0; j < NVARS; j++){
				if(randval(0,1) < CR || k == j) {			
					
					double dd1 = 0;
					double dd2 = 0;

					if (((int)best_chromosome.gene[j]) != ((int) population[i].gene[j])) dd1 = 1;
					if (((int)population[r1].gene[j]) != ((int) population[r2].gene[j])) dd2 = 1;

					change_vector[j] = F * dd1 + F * dd2 - (F * dd1 * F * dd2);

					if (randval(0,1) < change_vector[j]) {
						biasly_set_value(j, &new_population[i].gene[j]);
					} else {
						new_population[i].gene[j] =  population[i].gene[j];
					}
				} else {
					change_vector[j] = 0;
					new_population[i].gene[j] = population[i].gene[j];
				}
				// Compute mutate chromosome fitness
				this->compute_chromosome_fitness(&new_population[i]);
			}
		}
	}

	void rand_set_value(int I, int *x) {
		switch (gene_type_flag[I]) {
		case 0:
			if (randval(0, 1) < 1. / 3)
				*x = rand() % (base_function_num); // note that function_num = base_function_num + GSIZE;
			else if (randval(0, 1) < 0.5)
				*x = base_function_num + rand() % (GSIZE);
			else
				*x = L_terminal + rand() % (terminal_num);
			break;
		case 1:
			*x = L_terminal + rand() % (terminal_num);
			break;
		case 2:
			if (rand() % 2 == 0)
				*x = rand() % (base_function_num);
			else
				*x = L_input + rand() % (2);
			break;
		case 3:
			*x = L_input + rand() % (2);
			break;
		default:
			printf("fds");
		}
	}

	void decode_gene(int gene[NVARS]) {
		int op = -1;
		int i = 0;
		int k = 0, j;

		// Assign NULL to all 20 siblings of node i
		for(i = 0; i < NVARS; i++){  // For each gene in chromosome	
			link_comp[i].value = gene[i];
			for(j = 0; j < MAX_SIBLING; j++)
				link_comp[i].siblings[j] = NULL;
		}
		
		op = -1, i = 1;

		// Build the expression tree for the main program
		link_root = &link_comp[0];
		if(link_root->value < function_num){
			do{
				//find an op type item
				do{
					op++; 
					if (op >= i) {
						break;
					}
				} while(link_comp[op].value >= L_terminal); // Skip terminals

				if(op >= i) break;
				//set its left and right;
				if(link_comp[op].value < L_terminal){
					if(i >= H+T){break;}
					link_comp[op].siblings[0] = &link_comp[i];				
					i++;
					// Make sure that the expression has arity two
					if(link_comp[op].value < 4 || link_comp[op].value >= base_function_num){
						if(i >= H+T){ break;}
						link_comp[op].siblings[1] = &link_comp[i];
						i++;
					}
				}
			} while(true);
			if(op < i  && i >= H+T){ 			
				printf("\nERROR RULE111"); 
				getchar();
			}
		}

		//build sub expression trees of the individual
		for(int g = 0; g < GSIZE; g++){
			k = H+T + g *GNVARS;	// the starting position of the ADF.	
			for(i = 0; i < GNVARS; i++){
				sub_comp[g][i].value = gene[k + i];
				for(j = 0; j < MAX_SIBLING; j++)
					sub_comp[g][i].siblings[j] = NULL;
			}
			op = -1, i = 1;
			sub_root[g] = &sub_comp[g][0];
			if(sub_root[g]->value < L_terminal){  // Note that L_input > L_terminal;
				do{ // Find an op type item
					do{op++; if(op >= i)break;}while(sub_comp[g][op].value >= L_terminal);
					if(op >= i) break;
					// Set its left and right;
					if(sub_comp[g][op].value < base_function_num){
						if(i >= GH+GT-1){ break;}
						sub_comp[g][op].siblings[0] = &sub_comp[g][i];				
						i++;
						if(sub_comp[g][op].value < 4){
							sub_comp[g][op].siblings[1] = &sub_comp[g][i];
							i++;
						}
					}
				}while(true);
				if(op < i  && i >= GH+GT - 1){ printf("SUB ERROR RULE111"); getchar();}
			}
		}
	}

	void compute_rule(const struct LINK_COMP *node) {
		if(node->value >= L_terminal){
			current_value = training_input[node->value - L_terminal];
		}else{
			double t1;
			compute_rule(node->siblings[0]);
			t1 = current_value;
			if(node->value < 4 || node->value >= base_function_num){
				compute_rule(node->siblings[1]);
				t2 = current_value;
			}
			switch(node->value){
			case 0: // + 			
				current_value = t1 + t2; break;
			case 1: // -
				current_value = t1 - t2; break;
			case 2: // *
				current_value = t1 * t2; break;
			case 3: // /
				if(fabs(t2) <  1e-20) current_value = 0;else current_value = t1 / t2; break;
			case 4: // sin
				current_value = sin(t1); break;
			case 5: // cos
				current_value = cos(t1); break;
			case 6: // exp
				if(t1 < 20) current_value = exp(t1); else current_value = exp(20.); break;
			case 7: // log
				if(fabs(t1) <  1e-20) current_value = 0; else current_value = log(fabs(t1)); break;

			default: // GI
				sub_sibling_value[0] = t1;
				sub_sibling_value[1] = t2;
				compute_sub_rule(sub_root[node->value - 8]);
				current_value = sub_current_value;
				break;
			}
		}
	}

	void compute_sub_rule(const struct LINK_COMP * node) {
		if(node->value >= L_input){
			// If the node is an input then read data from the input vector, i.e., sub_sibling_value[...];
			sub_current_value = sub_sibling_value[node->value - L_input];
		}else{
			// First compute the left child of the node.
			double t1;
			compute_sub_rule(node->siblings[0]);

			t1 = sub_current_value;
			//then compute the right child of the node if the node contain right child
			if(node->value < 4){  // note that the first 4 functions have 2 children
				compute_sub_rule(node->siblings[1]);
				t2 = sub_current_value;
			}
			switch(node->value){
			case 0: //+ 			
				sub_current_value = t1 + t2; break;
			case 1: //-
				sub_current_value = t1 - t2; break;
			case 2: //*
				sub_current_value = t1 * t2; break;
			case 3: // /
				 if(fabs(t2) <  1e-20) sub_current_value = 0;else sub_current_value = t1 / t2; break;
			case 4: //sin
				 sub_current_value = sin(t1); break;
			case 5: //cos
				 sub_current_value = cos(t1); break;
			case 6: //exp
				 if(t1 < 20) sub_current_value = exp(t1); else sub_current_value = exp(20.); break;
			case 7: //log
				 if(fabs(t1) <  1e-20) sub_current_value = 0; else sub_current_value = log(fabs(t1)); break;
			default: printf("unknow function\n");
			}
		}
	}

	int choose_a_terminal() {
		int i, j;
		double p = randval(0,1);
		for(i = 0; i < terminal_num - 1; i++){
			if(p < terminal_probability[i])
				break;
		}
		return L_terminal+i;
	}

	int choose_a_function() {
		int i, j, k;
		double p = randval(0,1);
		for(i = 0; i < function_num - 1; i++){
			if(p < function_probability[i])
				break;
		}
		return i;
	}

	void update_probability() {
		double sum = 0;
		int i, j, k;
		// In the main head of population, the proportion of bits being function symbol
		FQ = 0;
		int	CC = 0;
		for(i = 0; i < POPSIZE; i++){
			for(j = 0; j < H; j++){
				if(population[i].gene[j] < L_terminal) FQ++;
				else if(population[i].gene[j] >= L_terminal) CC++;
			}
		}
		FQ = FQ / (double) (POPSIZE * H);

		bool print_flag = false;
		
		// Now compute the frequency of each symbol in the main parts of the current population.
		for(i = 0; i < MAXIMUM_ELEMENTS; i++){
			function_freq[i] = 1;	// Initialize a very small value
			terminal_freq[i] = 1;
		}

		for(i = 0; i < POPSIZE; i++){
			for(j = 0; j < H+T; j++){  // Only consider main parts
				if(population[i].gene[j] < L_terminal){
					function_freq[population[i].gene[j]]++;
				}else
					terminal_freq[population[i].gene[j] - L_terminal]++;
			}
		}
		
		sum = 0;
		for(i = 0; i < function_num; i++){
			sum +=function_freq[i];
		}
		function_probability[0] =  function_freq[0] / sum;
		for(i = 1; i < function_num; i++){
			function_probability[i] = function_freq[i] / sum + function_probability[i - 1];		
		}

		sum = 0;
		for(i = 0; i < terminal_num; i++){
			sum += terminal_freq[i];
			terminal_probability[i] = terminal_freq[i];
		}
		terminal_probability[0] =  terminal_probability[0] / sum;
		for(i = 1; i < terminal_num; i++){
			terminal_probability[i] = terminal_probability[i] / sum + terminal_probability[i - 1];	
		}
	}

	void biasly_set_value(int I, int*x) {
		// Here we only consider the main parts, while the sub-gene part are also randomly setting, so as to import population diversity
		switch(gene_type_flag[I]){
		case 0: 
			if(randval(0, 1) < FQ) *x = choose_a_function();
			else *x = choose_a_terminal();
			break;
		case 1: *x = choose_a_terminal(); break;
		case 2: 
			if(rand()%2==0) *x = rand()%(base_function_num);
			else *x = L_input + rand()%(2); 
			break;
		case 3: *x = L_input + rand()%(2);break;
		default: printf("fds");
		}
	}
};

// Macro to export functions into Python
PYBIND11_MODULE(slgep, handle) {
	handle.doc() = "Dont forget to update the docs";
	// Initialize function
	PYBIND11_NUMPY_DTYPE(CHROMOSOME, gene, fitness);
	// Classes	
	py::class_<SLGEP>(handle, "C_SLGEP")
		// Constructor
		.def(py::init<int>())
		// Properties
		.def_property_readonly("population", [](SLGEP &self) {
			std::vector<CHROMOSOME> population = self.get_population();
			py::array_t<CHROMOSOME> numpy_array(population.size(), population.data());
			return numpy_array;
		})
		.def_property_readonly("best", [](SLGEP &self) {
			std::vector<CHROMOSOME> population = self.get_best_chromosome();
			py::array_t<CHROMOSOME> numpy_array(population.size(), population.data());
			return numpy_array;
		})
		.def_property_readonly("generation", &SLGEP::get_generation_number)
		// Methods
		.def("set_dataset", &SLGEP::set_dataset)
		.def("generate_random_population", [](SLGEP &self, int POPSIZE) {
			self.generate_random_population(POPSIZE);
		})
		.def("fit", [](SLGEP &self, int generations) {
			py::array out = py::cast(self.fit(generations));
			return out;
		})
		.def("predict", [](SLGEP &self, std::vector<double> input, py::array_t<int> arr) {
			auto buffer = arr.request();
			int *gene = static_cast<int *>(buffer.ptr);
			double prediction = self.predict(input, gene);
			return prediction;
		});
}
