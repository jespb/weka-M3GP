package weka.classifiers.trees.m3gp.population;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import weka.classifiers.trees.m3gp.client.ClientWekaSim;
import weka.classifiers.trees.m3gp.client.Constants;
import weka.classifiers.trees.m3gp.tree.Tree;
import weka.classifiers.trees.m3gp.tree.TreeGeneticOperatorHandler;
import weka.classifiers.trees.m3gp.util.Arrays;

/**
 * 
 * @author Joao Batista, jbatista@di.fc.ul.pt
 *
 */
public class Population{
	// current and max generation
	public static int generation = 0;

	public static double[] goAffinity;

	// population
	private Tree [] population;

	//data, target column and fraction used for training
	private double [][] data;
	private String [] target;

	//fraction of the population used for tournament and elitism
	private int tournamentSize = 2;
	private int elitismSize = 1;

	//operations and terminals used
	private String [] terminals;

	//initial max depth
	private int maxDepth = 6;

	//from the trees with the best train rmse over the generations, this is the one with the lower test rmse
	private Tree bestTree = null;

	//public static double[] goAffinity;

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
	public Population(String [] term, double [][] data, String [] target) throws IOException{
		message("Creating forest...");
		

		tournamentSize = (int) (Constants.TOURNAMENT_FRACTION * Constants.POPULATION_SIZE);
		elitismSize = (int) (Constants.ELITISM_FRACTION * Constants.POPULATION_SIZE);

		this.data = data;
		this.target = target;

		this.terminals = term;

		population = new Tree[Constants.POPULATION_SIZE];

		for(int i = 0; i < Constants.POPULATION_SIZE; i++){
			population[i] = new Tree(term, 0 , maxDepth);
		}
		
		resetGOAffinity();
	}

	private void resetGOAffinity() {
		
		goAffinity = new double[Constants.NUMBER_OF_GENETIC_OPERATORS];
		for(int i = 0; i < goAffinity.length; i++) {
			goAffinity[i] = 1.0/goAffinity.length;
			//goAffinity[i] = Math.random();
		}
		goAffinity = Arrays.normalize(goAffinity);
		
		
		for(int i = 0; i < population.length; i++) {
			Tree t = population[i];
			t.setGOA(Arrays.copy(goAffinity));
		}
		
		
		/*
		for(Tree t : population) {
			
			double [] d = new double[Constants.NUMBER_OF_GENETIC_OPERATORS];
			for(int j = 0; j < d.length; j++) {
				d[j] = Math.random();
			}
			d = Arrays.normalize(d);
			
			t.setGOA(d);
		}
		*/
		
	}

	private double[] medianGOA() {
		try {
		double[][] ret = new double [population[0].getGOA().length][population.length];
		for(int i = 0; i < population.length; i++) {
			for(int k = 0; k < population[i].getGOA().length; k++) {
				ret[k][i] = population[i].getGOA()[k];
			}
		}
		double [] median = new double[population[0].getGOA().length];
		for(int i = 0; i < median.length; i++) {
			median[i] = Arrays.median(ret[i]);
		}
		return Arrays.normalize(median);
		}catch(Exception e) {
			e.printStackTrace();
		}
		return null;
	}


	/**
	 * Trains the classifier
	 */
	public ArrayList<Double>[] train() throws IOException {
		message("Starting train...");

		generation = 0;
		while(improving()){
			//resetGOAffinity();
			ClientWekaSim.datafile.write("        \"MedianGOA\":\"" + Arrays.arrayToString(medianGOA())  + "\"\n");
			ClientWekaSim.datafile.write("        \"Individuals\":{\n");

			if(generation%5 == 0)
				message("Generation " + generation + "...");
			nextGeneration();

			generation ++;

			ClientWekaSim.datafile.write("        }\n");
			if(improving())
				ClientWekaSim.datafile.write(",\n");
		}
		bestTree = prun(bestTree, data, target);

		return null;
	}

	/**
	 * Returns true if the classifier is still improving
	 */
	public boolean improving() {
		return generation < Constants.NUMBER_OF_GENERATIONS;
	}	

	// train is perfect
	boolean done = false;
	/**
	 * Evolves the classifier by one generation
	 */
	public void nextGeneration() throws IOException{
		if (done) {

			double train = bestTree.getTrainAccuracy(data, target);
			double test = bestTree.getTestAccuracy(data, target);
			message(generation + ": " + train + " // " + test + "/// (done)");
			ClientWekaSim.datafile.write(bestTree.toJSON(data, target)+"\n");
			return;
		}

		Tree [] nextGen = new Tree [population.length];
		double [] fitnesses = new double[population.length];


		//resetGOAffinity();


		// Obtencao de fitness
		long timeFitness = System.currentTimeMillis();
		for (int i = 0; i < population.length; i++) {
			fitnesses[i] = PopulationFunctions.fitnessTrain(population[i],data, target);
		}
		timeFitness = System.currentTimeMillis()-timeFitness;


		Arrays.mergeSortBy(population, fitnesses);

		long timeFile = System.currentTimeMillis();
		timeFile = System.currentTimeMillis()-timeFile;
		//ClientWekaSim.datafile.addGen(nextGen);

		//Pruning
		nextGen[0] = prun(population[population.length-1],data,target);

		// Elitismo 
		for(int i = 1; i < 1+elitismSize; i++ ){
			nextGen[i] = population[population.length-1-i];
		}


		//Selecao e reproducao
		int n_threads = Constants.NUMBER_OF_THREADS;
		Tree[][] descendents = new Tree[n_threads][nextGen.length/n_threads + 1];
		ExecutorService pool = Executors.newFixedThreadPool(n_threads);	
		
		for (int i = 0; i < n_threads; i++) {
			pool.submit(new BirthGiver(descendents[i],population));
		}
		pool.shutdown();		
		while(!pool.isTerminated());
	
		for(int i = 1+elitismSize; i < nextGen.length;){
			for(int t = 0; t < n_threads; t++) {
				for(int ind = 0; ind < descendents[t].length && i < nextGen.length; ind++) {
					nextGen[i] = descendents[t][ind];
					i++;
				}
			}
		}


		ClientWekaSim.datafile.write(population[population.length-1].toJSON(data, target)+"\n");

		if(elitismSize == 0) {
			setBestToLast(population);
		}

		bestTree = population[population.length-1];

		double train = bestTree.getTrainAccuracy(data, target);
		double test = bestTree.getTestAccuracy(data, target);

		message(generation + ": " + train + " // " + test + "///" + Arrays.arrayToString(bestTree.getGOA()));

		population = nextGen;

		if(train == 1) {
			bestTree = prun(bestTree, data, target);
			done = true;
		}
	}

	/**
	 * Sets the tree with the higher fitness to the index 0 of the population
	 * @param pop population
	 */
	private void setBestToLast(Tree[] pop) {
		int bestIndex = 0;
		double bestRMSE = pop[0].getTrainAccuracy(data, target);
		double candidateRMSE;
		for(int i = 0; i < pop.length; i++){
			candidateRMSE = pop[i].getTrainAccuracy(data, target);
			if(candidateRMSE > bestRMSE){
				bestRMSE = candidateRMSE;
				bestIndex = i;
			}
		}
		Tree dup = pop[pop.length-1];
		pop[pop.length-1] = pop[bestIndex];
		pop[bestIndex] = dup;
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
		if(Constants.MESSAGES)
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


	private Tree prun(Tree tree, double[][] data, String[] target) {
		return PopulationFunctions.prun(tree,data,target);
	}




	private class BirthGiver implements Runnable{
		Tree[] descendents;
		Tree[] population;

		public BirthGiver(Tree[] descendents, Tree[] population) {
			this.descendents = descendents;
			this.population = population;
		}

		public void run() {
			for(int i = 0; i < descendents.length; i++) {
				Tree [] cross = TreeGeneticOperatorHandler.geneticOperation(population, tournamentSize, terminals, data, target);
				for(int k = 0; k < cross.length && k+i < descendents.length; k++){
					descendents[i+k] = cross[k];
				}
				i+=cross.length-1;
			}
		}
	}
}