package weka.classifiers.trees.m3gp.client;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.trees.m3gp.population.Population;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Data;
import weka.classifiers.trees.m3gp.util.Files;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class ClientWekaSim {

	private static int file = 1; // ST, GS

	private static String xDataInputFilename = "datasets\\" + "Brazil_x.txt glass_x.csv cc_x.csv".split(" ")[file];
	private static String yDataInputFilename = "datasets\\" + "Brazil_y.txt glass_y.csv cc_y.csv".split(" ")[file];
	private static String resultOutputFilename = "fitovertime.csv";
	private static String treeType = "Ramped";

	private static String [] operations = "+ - * /".split(" ");
	private static String [] terminals = null;

	private static double trainFraction = 0.70;
	private static double tournamentFraction = 0.07;
	private static double elitismFraction = 0.05;

	private static int numberOfGenerations = 50;
	private static int numberOfRuns = 50;
	private static int populationSize = 80;
	private static int maxDepth = 6;

	private static boolean shuffleDataset = true;

	private static double [][] data = null;
	private static String [] target = null;


	// Variables
	public static double [][] results = new double [numberOfGenerations][3];
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


		BufferedWriter out = new BufferedWriter(new FileWriter(resultOutputFilename+".tmp"));
		out.write("Treino;Teste\n");
		for(int i = 0; i < results.length; i++){
			if(results[i][2] !=0)
				out.write(results[i][0]/results[i][2] + ";" + results[i][1]/results[i][2] + "\n");
		}
		out.close();
		Files.fixCSV(resultOutputFilename);
	}

	/**
	 * Prepara o cliente para a sua execucao
	 * @throws IOException
	 */
	private static void init() throws IOException{
		data = Data.readData(xDataInputFilename);
		target = Data.readTarget(yDataInputFilename);
	}

	/**
	 * Executa uma simulacao
	 * @param run
	 * @throws IOException
	 */
	private static void run(int run) throws IOException{
		System.out.println("Run " + run + ":");

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

/*
		// Este bloco está a certificarse que as previsoes sao consistentes com o treino
		double acc = 0;
		int hit = 0;
		String prediction = "";
		for(int i = 0; i < test.length; i++){
			prediction= f.predict(test[i]) ;
			acc += prediction.equals(target[train.length + i]) ? 1:0;
			if((i+1)%400 ==0)
				System.out.println((i+1) + "/" + test.length);
		}
		acc /= 1.0 * test.length;
		acc = Math.sqrt(acc);

		
		System.out.println("test binary classification hits: " + hit +" out of " + test.length);
		System.out.println("test RMSE calculated: " + acc);

		acc = 0;
		hit = 0;

		for(int i = 0; i < train.length; i++){
			prediction = f.predict(train[i]);
			acc+= prediction.equals(target[i]) ? 1:0;
			if((i+1)%400 ==0)
				System.out.println((i+1) + "/" + train.length);
		}
		acc /= 1.0 * train.length;
		acc = Math.sqrt(acc);

		System.out.println("train binary classification hits: " + hit +" out of " + train.length);
		System.out.println("train RMSE calculated: " + acc);
*/
		
		System.out.println(f);
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
		terminals = new String [data[0].length];
		for(int i = 0; i < terminals.length; i++)
			terminals[i] = "x"+i;
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