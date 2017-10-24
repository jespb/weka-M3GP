package m3gp.forest;

import java.io.IOException;
import java.util.ArrayList;

import m3gp.client.ClientWekaSim;
import m3gp.tree.TreeM3GP;
import m3gp.tree.TreeM3GPCrossoverHandler;
import m3gp.tree.TreeM3GPMutationHandler;
import m3gp.tree.TreeM3GPPruningHandler;
import m3gp.util.Arrays;
import m3gp.util.Mat;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
 *
 */
public class ForestM3GP{
	private boolean messages = true;

	// current and max generation
	private int generation = 0;
	private int maxGeneration = 10;

	// population
	private TreeM3GP [] population;
	private double fullTreeFraction;

	//data, target column and fraction used for training
	private double [][] data;
	private String [] target;
	private double trainFraction;

	//fraction of the population used for tournament and elitism
	private double tournamentFraction = 0.04;
	private double elitismFraction = 0.01;

	//operations and terminals used
	private String [] operations;
	private String [] terminals;
	
	//probability of a node being terminal while using grow to create a tree
	private double terminalRateForGrow = 0.10;
	
	//initial max depth
	private int maxDepth = 6;

	//from the trees with the best train rmse over the generations, this is the one with the lower test rmse
	private TreeM3GP bestTree = null;

	/**
	 * Construtor
	 * @param filename
	 * @param op
	 * @param term
	 * @param maxDepth
	 * @param t_rate
	 * @param data
	 * @param target
	 * @param populationSize
	 * @param trainFract
	 * @throws IOException
	 */
	public ForestM3GP(String filename, String [] op, String [] term, int maxDepth, 
			double [][] data, String [] target, int populationSize, double trainFract,
			String populationType, int maxGeneration) throws IOException{
		message("Creating forest...");

		this.data = data;
		this.target = target;
		this.trainFraction = trainFract;

		this.operations = op;
		this.terminals = term;
		this.maxDepth = maxDepth;

		this.maxGeneration = maxGeneration;

		population = new TreeM3GP[populationSize];

		switch(populationType) {
		case "Ramped":
			for(int i = 0; i < populationSize; i++){
				if( i < (int)(populationSize * fullTreeFraction))
					population[i] = new TreeM3GP(op, term, 0 , Math.max(i%maxDepth,2));
				else
					population[i] = new TreeM3GP(op, term, terminalRateForGrow , Math.max(i%maxDepth,2));
			}
			break;
		default: // full
			for(int i = 0; i < populationSize; i++){
				if( i < populationSize / 2)
					population[i] = new TreeM3GP(op, term, 0 , maxDepth);
			}
			break;
		}
		
		TreeM3GP.trainSize = (int)(target.length * trainFraction);
		
	}

	/**
	 * Sets the fraction of the population used in tournaments to d
	 */
	public void setTournamentFraction(double d){
		tournamentFraction = d;
	}
	
	/**
	 * Sets the fraction of the population selected by elitism to d
	 */
	public void setElitismFraction(double d){
		elitismFraction = d;
	}

	/**
	 * Trains the classifier
	 */
	public ArrayList<Double>[] train() throws IOException {
		message("Starting train...");

		generation = 0;
		while(improving()){
			if(generation%5 == 0)
				message("Generation " + generation + "...");
			double [] result = nextGeneration(generation);
			ClientWekaSim.results[generation][0]+=result[0];
			ClientWekaSim.results[generation][1]+=result[1];
			ClientWekaSim.results[generation][2]++;
			generation ++;
		}
		return null;
	}

	/**
	 * Returns true if the classifier is still improving
	 */
	public boolean improving() {
		return generation < maxGeneration;
	}	

	/**
	 * Evolves the classifier by one generation
	 */
	public double[] nextGeneration(int generation) throws IOException{
		double [] results = new double[2];

		TreeM3GP [] nextGen = new TreeM3GP [population.length];
		double [] fitnesses = new double[population.length];

		// Obtencao de fitness
		for (int i = 0; i < population.length; i++){
			fitnesses[i] = population[i].getTrainAccuracy(data, target,trainFraction);
		//	System.out.println(population[i]);
			//System.out.println(fitnesses[i]);
		}

		Arrays.mergeSortBy(population, fitnesses);
		/*for(int i = 0; i < fitnesses.length; i++) {
			System.out.println(fitnesses[i] + "         " + population[i].getTrainAccuracy(data, target,trainFraction) + "    " + population[population.length-1].reference());//TODO REEEEEE
		}*/
		//System.out.println(fitnesses[0]);
		
		//Prunning
		nextGen[0] = TreeM3GPPruningHandler.prun(population[population.length-1],data,target,trainFraction);
		
		// Elitismo
		for(int i = 1; i < getElitismSize(); i++ ){
			nextGen[i] = population[population.length-1-i];
		}
		
		
		//Selecao e reproducao
		TreeM3GP parent1, parent2;
		for(int i = getElitismSize(); i < nextGen.length; i++){
			parent1 = tournament(population);
			parent2 = tournament(population);

			if(Math.random() < 0.75) {//TODO default = 0.75
				nextGen[i] = crossover(parent1, parent2);
			}else {
				nextGen[i] = mutation(parent1);
			}
		}
		
		if(getElitismSize() == 0) {
			setBestToLast(population);
		}
			
		//bestTree = population[population.length-1];
		//bestTree = nextGen[0];
		
		//Sets the bestTree to the generation best if it has a better test error 
		if(bestTree == null || population[population.length-1].getTestAccuracy(data, target, trainFraction)>bestTree.getTestAccuracy(data, target, trainFraction)) {
			bestTree = population[population.length-1];
		}
		
		results[0] = bestTree.getTrainAccuracy(data, target, trainFraction);
		results[1] = bestTree.getTestAccuracy(data, target, trainFraction);
		
		System.out.println(generation + ": " + results[0] + " // " + results[1] + " // " + population[population.length-1].reference());
		
		population = nextGen;
		
		return results;
	}

	/**
	 * Sets the tree with the higher fitness to the index 0 of the population
	 * @param pop population
	 */
	private void setBestToLast(TreeM3GP[] pop) {
		int bestIndex = 0;
		double bestRMSE = pop[0].getTrainAccuracy(data, target,trainFraction);
		double candidateRMSE;
		for(int i = 0; i < pop.length; i++){
			candidateRMSE = pop[i].getTrainAccuracy(data, target,trainFraction);
			if(candidateRMSE > bestRMSE){
				bestRMSE = candidateRMSE;
				bestIndex = i;
			}
		}
		TreeM3GP dup = pop[pop.length-1];
		pop[pop.length-1] = pop[bestIndex];
		pop[bestIndex] = dup;
	}

	/**
	 * Returns the tournament absolute size
	 * @return number of individuals on the tournament
	 */
	private int getTournamentSize(){
		return (int) (tournamentFraction * population.length);
	}

	/**
	 * Returns the elite absolute size
	 * @return number of individuals on the elite
	 */
	private int getElitismSize(){
		return (int) (elitismFraction * population.length);
	}

	/**
	 * Returns the prediction
	 * @param v arguments
	 */
	public String predict(double [] v) {
		return bestTree.predict(v);
	}

	/**
	 * Prints a message if messages is set to true
	 * @param s
	 */
	private void message(String s){
		if(messages)
			System.out.println(s);
	}


	/**
	 * Returns the best tree of the train in it's String format
	 */
	public String toString(){
		if(bestTree == null)
			return "This classifier has not been trained yet.";
		else
			return bestTree.toString();
	}
	
	/**
	 * Picks as many trees from population as the size of the tournament
	 * and return the one with the lower fitness, assuming the population
	 * is already sorted
	 * @param population Tree population
	 * @return The winner tree
	 */
	private TreeM3GP tournament(TreeM3GP [] population) {
		int tournamentSize = getTournamentSize();
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
	private TreeM3GP crossover(TreeM3GP p1, TreeM3GP p2) {
		TreeM3GP candidate, smaller = TreeM3GPCrossoverHandler.crossover(p1, p2,data, target, trainFraction);
		for(int i = 1; i < 0;i++) {//getTournamentSize(); i++) {
			candidate = TreeM3GPCrossoverHandler.crossover(p1, p2,data, target, trainFraction);
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
	private TreeM3GP mutation(TreeM3GP p) {
		TreeM3GP candidate, smaller = TreeM3GPMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
		for(int i = 1; i < getTournamentSize(); i++) {
			candidate = TreeM3GPMutationHandler.mutation(p, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
			if(candidate.size() < smaller.size()) {
				smaller = candidate;
			}
		}
		return smaller;
	}
}