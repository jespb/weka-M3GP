package weka.classifiers.trees.m3gp.population;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import weka.classifiers.trees.m3gp.client.ClientWekaSim;
import weka.classifiers.trees.m3gp.tree.Tree;
import weka.classifiers.trees.m3gp.tree.TreeGeneticOperatorHandler;
import weka.classifiers.trees.m3gp.util.Arrays;

/**
 * 
 * @author Joao Batista, jbatista@di.fc.ul.pt
 *
 */
public class Population{
	private boolean messages = true;

	// current and max generation
	public static int generation = 0;

	public static double[] goAffinity;
	private int maxGeneration = 10;

	// population
	private Tree [] population;

	//data, target column and fraction used for training
	private double [][] data;
	private String [] target;
	private double trainFraction;

	//fraction of the population used for tournament and elitism
	private int tournamentSize = 2;
	private int elitismSize = 1;

	//operations and terminals used
	private String [] operations;
	private String [] terminals;
	
	//probability of a node being terminal while using grow to create a tree
	private double terminalRateForGrow = 0.1;
	
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
	public Population(String filename, String [] op, String [] term, int maxDepth, 
			double [][] data, String [] target, int populationSize, double trainFract,
			String populationType, int maxGeneration, double tournamentFraction,
			double elitismFraction) throws IOException{
		message("Creating forest...");
		resetGOAffinity();

		tournamentSize = (int) (tournamentFraction * populationSize);
		elitismSize = (int) (elitismFraction * populationSize);
		
		this.data = data;
		this.target = target;
		this.trainFraction = trainFract;

		this.operations = op;
		this.terminals = term;
		this.maxDepth = maxDepth;

		this.maxGeneration = maxGeneration;

		population = new Tree[populationSize];

		switch(populationType) {
		case "Ramped":
			for(int i = 0; i < populationSize; i++){
				if( i < (int)(populationSize * 0.75))
					population[i] = new Tree(op, term, 0 , (i% (maxDepth-1) ) +2 );
				else
					population[i] = new Tree(op, term, terminalRateForGrow , (i% (maxDepth-1) ) +2 );
			}
			break;
		default: // full
			for(int i = 0; i < populationSize; i++){
				population[i] = new Tree(op, term, 0 , maxDepth);
			}
			break;
		}
		
		Tree.trainSize = (int)(target.length * trainFraction);		
	}
	
	private void resetGOAffinity() {
		goAffinity = new double[5];
		for(int i = 0; i < goAffinity.length; i++) {
			goAffinity[i] = 1.0/goAffinity.length;
		}
	}
	
	private double[] medianGOA() {
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
		return median;
	}
	

	/**
	 * Trains the classifier
	 */
	public ArrayList<Double>[] train() throws IOException {
		message("Starting train...");

		generation = 0;
		while(improving()){
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
		bestTree = prun(bestTree, data, target, trainFraction);
		
		return null;
	}

	/**
	 * Returns true if the classifier is still improving
	 */
	public boolean improving() {
		return generation < maxGeneration;
	}	

	// train is perfect
	boolean done = false;
	/**
	 * Evolves the classifier by one generation
	 */
	public void nextGeneration() throws IOException{
		if (done) {
			
			double train = bestTree.getTrainAccuracy(data, target, trainFraction);
			double test = bestTree.getTestAccuracy(data, target, trainFraction);
			System.out.println(generation + ": " + train + " // " + test + "/// (done)");
			ClientWekaSim.datafile.write(bestTree.toJSON(data, target, trainFraction)+"\n");
			return;
		}

		Tree [] nextGen = new Tree [population.length];
		double [] fitnesses = new double[population.length];

		
		//resetGOAffinity();
		
		
		// Obtencao de fitness
		long timeFitness = System.currentTimeMillis();
		ExecutorService pool = Executors.newFixedThreadPool(3);	
		for (int i = 0; i < population.length; i++) {
			pool.submit(new FitnessCalculator(fitnesses, i, population[i]));
		}
		pool.shutdown();
		while(!pool.isTerminated());
		
		timeFitness = System.currentTimeMillis()-timeFitness;

		
		Arrays.mergeSortBy(population, fitnesses);
		
		long timeFile = System.currentTimeMillis();
		timeFile = System.currentTimeMillis()-timeFile;
		//ClientWekaSim.datafile.addGen(nextGen);
		
		//Pruning 			//TODO fix
		long timePruning = System.currentTimeMillis();
		nextGen[0] = prun(population[population.length-1],data,target,trainFraction);
		timePruning = System.currentTimeMillis()-timePruning;
		
		// Elitismo 
		for(int i = 1; i < 1+elitismSize; i++ ){
			nextGen[i] = population[population.length-1-i];
		}
		
		
		//Selecao e reproducao

		long timeSelectAndCross = System.currentTimeMillis();
		
		for(int i = 1+elitismSize; i < nextGen.length; i++){
			Tree [] cross = TreeGeneticOperatorHandler.geneticOperation(population, tournamentSize, operations, terminals, terminalRateForGrow, maxDepth, data, target, trainFraction);
			for(int k = 0; k < cross.length && k+i < population.length; k++){
				nextGen[i+k]=cross[k];
			}
			i+=cross.length-1;
		}

		
		
		timeSelectAndCross = System.currentTimeMillis()-timeSelectAndCross;
		ClientWekaSim.datafile.write(population[population.length-1].toJSON(data, target, trainFraction)+"\n");
		
		if(elitismSize == 0) {
			setBestToLast(population);
		}
			
		bestTree = population[population.length-1];

		double train = bestTree.getTrainAccuracy(data, target, trainFraction);
		double test = bestTree.getTestAccuracy(data, target, trainFraction);
		
		System.out.println(generation + ": " + train + " // " + test + "///" + Arrays.arrayToString(bestTree.getGOA()));
		
		population = nextGen;
		
		if(train == 1) {
			bestTree = prun(bestTree, data, target, trainFraction);
			done = true;
		}
	}

	/**
	 * Sets the tree with the higher fitness to the index 0 of the population
	 * @param pop population
	 */
	private void setBestToLast(Tree[] pop) {
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
	
	
	private Tree prun(Tree tree, double[][] data, String[] target, double trainFraction) {
		return PopulationFunctions.prun(tree,data,target,trainFraction);
	}
	
	private class FitnessCalculator implements Runnable{
		double [] fit;
		int index;
		Tree t;
		
		public FitnessCalculator(double [] fit, int index, Tree t) {
			this.fit = fit;
			this.index = index;
			this.t= t;
		}
		
		public void run() {
			fit[index] = PopulationFunctions.fitnessTrain(t,data, target,trainFraction);
		}
	}

}