package weka.classifiers.trees.m3gp.client;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.trees.m3gp.population.Population;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Data;

/**
 * 
 * @author Joao Batista, jbatista@di.fc.ul.pt
 *
 */
public class ClientWekaSim {
	private static double [][] data = null;
	private static String [] target = null;

	public static BufferedWriter datafile;
	
	private static String singleRun=Constants.DATASETS[8];

	// Variables
	private static String dataset = null;
	private static String[] terminals = null;
	private static Population f = null;

	/**
	 * main
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		if (Constants.SINGLE_DATASET) {
			System.out.println("RUNNING FILE: " + singleRun);
			dataset = singleRun;
			init();

			long time = System.currentTimeMillis();
			for(int run = 0 ; run < Constants.NUMBER_OF_RUNS; run++){
				run(run + Constants.INITIAL_RUN_ID);
			}
			System.out.println((System.currentTimeMillis() - time) + "ms");

		}else {
			for( String file : Constants.DATASETS) {
				System.out.println("RUNNING FILE: " + file);
				dataset = file;

				init();

				long time = System.currentTimeMillis();
				for(int run = 0 ; run < Constants.NUMBER_OF_RUNS; run++){
					run(run +  Constants.INITIAL_RUN_ID);
				}
				System.out.println((System.currentTimeMillis() - time) + "ms");
			}
		}

	}

	/**
	 * Prepara o cliente para a sua execucao
	 * @throws IOException
	 */
	private static void init() throws IOException{
		Object [] datatarget = Data.readDataTarget(Constants.DATASET_DIR + dataset);
		data = (double[][]) datatarget [0];
		target = (String[]) datatarget [1];
	}

	/**
	 * Executa uma simulacao
	 * @param run
	 * @throws IOException
	 */
	private static void run(int run) throws IOException{
		System.out.println("Run " + run + "("+dataset.split("[.]")[0]+"):");
		datafile = new BufferedWriter(new FileWriter(Constants.OUTPUT_DIR+"Run_"+run+"_"+dataset.split("[.]")[0]+".json"));
		datafile.write("{\n    \"generations\": [{\n");

		if(Constants.SHUFFLE_DATASET)Arrays.shuffle(data, target);

		setTerm(data);

		double [][] train = new double [(int) (data.length*Constants.TRAIN_FRACTION)][data[0].length];
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
		f = new Population(terminals, data, target);
	}
}