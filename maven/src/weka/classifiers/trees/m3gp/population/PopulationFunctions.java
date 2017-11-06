package weka.classifiers.trees.m3gp.population;

import weka.classifiers.trees.m3gp.tree.Tree;
import weka.classifiers.trees.m3gp.tree.TreeCrossoverHandler;
import weka.classifiers.trees.m3gp.tree.TreeMutationHandler;
import weka.classifiers.trees.m3gp.util.Mat;

class PopulationFunctions {
	/**
	 * Picks as many trees from population as the size of the tournament
	 * and return the one with the lower fitness, assuming the population
	 * is already sorted
	 * @param population Tree population
	 * @return The winner tree
	 */
	static boolean smallerIsBetter = false;
	static Tree tournament(Tree [] population, int tournamentSize) {
		int pick = Mat.random(tournamentSize);
		for(int i = 1; i < tournamentSize; i ++){
			if(smallerIsBetter)
				pick = Math.min(pick, Mat.random(population.length));
			else
				pick = Math.max(pick, Mat.random(population.length));
		}
		return population[pick];
	}
	
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents of p1 and p2 through crossover and returns
	 * the smallest one
	 * @param p1 Parent 1
	 * @param p2 Parent 2
	 * @return Descendent of p1 and p2 through crossover
	 */
	static Tree[] crossover(Tree[] pop, int tournamentSize, double [][] data, String [] target, double trainFraction) {
		Tree p1 = tournament(pop, tournamentSize);
		Tree p2 = tournament(pop, tournamentSize);
		return TreeCrossoverHandler.crossover(p1, p2,data, target, trainFraction);
	}
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents from p through mutation and returns the 
	 * smallest one
	 * @param p Original Tree
	 * @return Descendent of p by mutation
	 */
	static Tree mutation(Tree[] population, int tournSize, String [] operations, String[] terminals,
			double terminalRateForGrow, int maxDepth, double [][] data, 
			String [] target, double trainFraction) {
		Tree p = tournament(population, tournSize);
		return TreeMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
	}
}