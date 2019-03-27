package weka.classifiers.trees.m3gp.client;

import java.io.File;

public final class Constants {	
	public final static String[] DATASETS = "heart.csv mcd3.csv mcd10.csv movl.csv seg.csv vowel.csv wav.csv yeast.csv brazil.csv".split(" ");
	public final static String DATASET_DIR = "datasets" + File.separator;
	public final static String OUTPUT_DIR = "results_p99_glob" + File.separator;
	
	public final static boolean SHUFFLE_DATASET = true;	
	public final static boolean SINGLE_DATASET = true;
	
	public final static String [] OPERATIONS = "+ - * /".split(" ");

	public final static double TRAIN_FRACTION = 0.70;
	public final static double ELITISM_FRACTION = 0.002;
	public final static double TOURNAMENT_FRACTION = 0.01;

	public final static int NUMBER_OF_GENERATIONS = 100;
	public final static int NUMBER_OF_RUNS = 30;
	public final static int INITIAL_RUN_ID = 0;
	
	public final static int POPULATION_SIZE = 500;
	public final static int MAX_DEPTH = 6;
	
	public final static int DISTANCE_USED = 2; // 1-Mahalanobis; 2-Euclidean
	public final static int NUMBER_OF_GENETIC_OPERATORS = 5;
	
	public static final int NUMBER_OF_THREADS = 6;
	
	/*
	 * Method used for the adaptation of the selection probability of the GOs
	 * 0 : No adaptation
	 * Negative values : Global vector
	 * Positive values : Individual vector
	 * -1 / 1 : Metodo de correcao da Sara
	 * -2 / 2 : Metodo de correcao do Joao
	 */
	public final static int PROBABILITY_ADAPTATION = -2;
	
	public final static double LEARNING_T = 0.99;
	
	public final static boolean MESSAGES = true;
}