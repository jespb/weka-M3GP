package m3gp.forest;

import m3gp.tree.Tree;
import m3gp.tree.TreeCrossoverHandler;
import m3gp.tree.TreeMutationHandler;
import m3gp.util.Mat;

class ForestFunctions {
	/**
	 * Picks as many trees from population as the size of the tournament
	 * and return the one with the lower fitness, assuming the population
	 * is already sorted
	 * @param population Tree population
	 * @return The winner tree
	 */
	static Tree tournament(Tree [] population, int tournamentSize) {
		int pick = Mat.random(population.length);
		for(int i = 1; i < tournamentSize; i ++){
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
	static Tree crossover(Tree p1, Tree p2, double [][] data, String [] target, double trainFraction) {
		Tree candidate, smaller = TreeCrossoverHandler.crossover(p1, p2,data, target, trainFraction);
		for(int i = 1; i < 0;i++) {//getTournamentSize(); i++) {
			candidate = TreeCrossoverHandler.crossover(p1, p2,data, target, trainFraction);
			if(candidate.size() < smaller.size()) {
				smaller = candidate;
			}
		}
		return smaller;
	}
	
	/**
	 * This method creates as many trees as the size of the tournament
	 * which are descendents from p through mutation and returns the 
	 * smallest one
	 * @param p Original Tree
	 * @return Descendent of p by mutation
	 */
	static Tree mutation(Tree p, int tournSize, String [] operations, String[] terminals,
			double terminalRateForGrow, int maxDepth, double [][] data, 
			String [] target, double trainFraction) {
		Tree candidate, smaller = TreeMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
		for(int i = 1; i < 0*tournSize; i++) {
			candidate = TreeMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
			if(candidate.size() < smaller.size()) {
				smaller = candidate;
			}
		}
		return smaller;
	}
}
