package m3gp.client;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import m3gp.forest.ForestM3GP;
import m3gp.util.Arrays;
import m3gp.util.Data;
import m3gp.util.Files;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class ClientWekaSim {

	static int file = 2; // ST, GS

	static String xDataInputFilename = "Brazil_x.txt glass_x.csv cc_x.csv".split(" ")[file];
	static String yDataInputFilename = "Brazil_y.txt glass_y.csv cc_y.csv".split(" ")[file];
	static String resultOutputFilename = "fitovertime.csv";
	static String treeType = "Ramped";

	static String [] operations = "+ - * /".split(" ");
	static String [] terminals = null;

	static double trainPercentage = 0.70;
	static double tournamentPercentage = 0.05;
	static double elitismPercentage = 0.05;

	static int numberOfGenerations = 70;
	static int numberOfRuns = 1;
	static int populationSize = 40;
	static int maxDepth = 6;

	static boolean shuffle = true;

	static double [][] train_r = null;
	static double [][] test_r = null;
	static double [][] data = null;
	static String [] target = null;


	// Variables
	public static double [][] results = new double [numberOfGenerations][3];
	static ForestM3GP f = null;

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
		train_r = new double[numberOfGenerations][numberOfRuns];
		test_r = new double[numberOfGenerations][numberOfRuns];

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

		if(shuffle)Arrays.shuffle(data, target);

		setTerm(data);

		double [][] train = new double [(int) (data.length*trainPercentage)][data[0].length];
		double [][] test = new double [data.length - train.length][data[0].length];

		for(int i = 0; i < data.length; i++){
			if( i < train.length)
				train[i] = data[i];
			else
				test[i - train.length] = data[i];
		}

		setForest();
		setTournamentSize();
		setElitismSize();

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
	 * Actualiza o tamanho dos torneios
	 */
	private static void setTournamentSize(){
		f.setTournamentFraction(tournamentPercentage);
	}

	/**
	 * Actualiza o tamanho do elitismo
	 */
	private static void setElitismSize(){
		f.setElitismFraction(elitismPercentage);
	}

	/**
	 * Cria uma nova floresta
	 * @throws IOException
	 */
	private static void setForest() throws IOException{
		f = new ForestM3GP("", operations, 
				terminals, maxDepth, data, target, 
				populationSize,trainPercentage, treeType,numberOfGenerations);
	}
}