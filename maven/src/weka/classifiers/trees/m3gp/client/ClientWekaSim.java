package weka.classifiers.trees.m3gp.client;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.trees.m3gp.population.Population;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Data;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class ClientWekaSim {

	private static int file = 3; // ST, GS

	private static String filename = "brazil.csv heart.csv waveform.csv vowel.csv".split(" ")[file];
	private static String datasetFilename = "datasets\\" + filename;
	private static String treeType = "Ramped";

	public static String [] operations = "+ - * /".split(" ");
	private static String [] terminals = null;

	private static int speed = 1;
	
	private static double trainFraction = 0.70;
	private static double tournamentFraction = 0.01 * speed;
	private static double elitismFraction = 0.002 * speed ;

	private static int numberOfGenerations = 100;
	private static int initialRun_ID = 20;
	private static int numberOfRuns = 10;
	private static int populationSize = 500 / speed;
	private static int maxDepth = 6;

	private static boolean shuffleDataset = true;

	private static double [][] data = null;
	private static String [] target = null;
	
	public static BufferedWriter datafile;
	

	// Variables
	private static Population f = null;

	/**
	 * main
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		treatArgs(args);
		init();

		long time = System.currentTimeMillis();
		for(int run = 0 ; run < numberOfRuns; run++){
			run(run);
		}
		System.out.println((System.currentTimeMillis() - time) + "ms");
	}

	/**
	 * Prepara o cliente para a sua execucao
	 * @throws IOException
	 */
	private static void init() throws IOException{
		Object [] datatarget = Data.readDataTarget(datasetFilename);
		data = (double[][]) datatarget [0];
		target = (String[]) datatarget [1];
	}

	/**
	 * Executa uma simulacao
	 * @param run
	 * @throws IOException
	 */
	private static void run(int run) throws IOException{
		System.out.println("Run " + run + ":");
		datafile = new BufferedWriter(new FileWriter("Run_"+(run+initialRun_ID)+"_"+filename.split("[.]")[0]+".json"));
		datafile.write("{\n    \"generations\": [{\n");

		if(shuffleDataset)Arrays.shuffle(data, target);

		setTerm(data);

		double [][] train = new double [(int) (data.length*trainFraction)][data[0].length];
		double [][] test = new double [data.length - train.length][data[0].length];

		for(int i = 0; i < data.length; i++){
			if( i < train.length)
				train[i] = data[i];
			else
				test[i - train.length] = data[i];
		}

		setPopulation();
		
		f.train();
		
		System.out.println(f);
		
		datafile.write("    }]\n}");
		datafile.close();
	}

	/**
	 * Trata dos argumentos fornecidos
	 * @param args
	 */
	private static void treatArgs(String [] args){
		for(int i = 0; i < args.length; i++){
			String [] split = args[i].split(":");
			switch(split[0]){
			case "depth":
				maxDepth = Integer.parseInt(split[1]);
				break;
			case "maxgen":
				numberOfGenerations = Integer.parseInt(split[1]);
				break;
			case "popsize":
				populationSize = Integer.parseInt(split[1]);
				break;
			}
		}
	}

	/**
	 * Define o valor dos terminais
	 * @param data
	 */
	private static void setTerm(double [][] data){
		terminals = new String [data[0].length+1];
		for(int i = 0; i < terminals.length; i++)
			terminals[i] = "x"+i;
		terminals[terminals.length-1] = "r";
	}

	/**
	 * Cria uma nova floresta
	 * @throws IOException
	 */
	private static void setPopulation() throws IOException{
		f = new Population("", operations, 
				terminals, maxDepth, data, target, 
				populationSize,trainFraction, treeType,numberOfGenerations,
				tournamentFraction, elitismFraction);
	}
}