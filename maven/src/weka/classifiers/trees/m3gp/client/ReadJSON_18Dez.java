package weka.classifiers.trees.m3gp.client;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;


//INDEPENDENT CLASS
public class ReadJSON_18Dez {
	private static final String FS = File.separator;
	private static final String PROPERTIES_FILENAME = "ReadJSON_18Dez.properties";
	private static HashMap<String, String[]> properties = new HashMap<String, String[]>();

	/*
	 * Properties:
	 * datasets=ds1 ds2 ds3
	 * gens=gens
	 * runs=runs
	 * 
	 * Args:
	 * dir=dir
	 */
	public static void main(String [] args) throws IOException{
		treatArgs(args);
		readProperties();

		int runs = Integer.parseInt(properties.get("runs")[0]);
		int gens = Integer.parseInt(properties.get("gens")[0]);

		String dir = properties.get("dir")[0];

		for(String dataset : properties.get("datasets")){
			int [][] size = new int[gens][runs];
			int [][] dims = new int[gens][runs];

			double [][] tr = new double[gens][runs];
			double [][] te = new double[gens][runs];
			double [][][] med_goa = new double[gens][runs][5];
			double [][][] goa = new double[gens][runs][5];

			for(int r = 0; r < runs; r++) {
				System.out.print("> Reading results from " + "Run_"+ r +"_" +dataset + ".json ");

				Scanner file = new Scanner( new File ( dir + FS + dataset + FS + "Run_"+ r +"_" +dataset + ".json" ));

				for(int g = 0; g < gens; g++) {

					med_goa[g][r] = readNextArray(file, "MedianGOA");

					int [] nodes_dimensions = countNodesAndDimensions(file);

					goa[g][r] = readNextArray(file, "GOA");

					ArrayList<String> train = readTrain(file);
					ArrayList<String> test = readTest(file);


					size[g][r] = nodes_dimensions[0];
					dims[g][r] = nodes_dimensions[1];

					double [][] tr_map = new double[train.size()][dims[g][r]];
					double [][] te_map = new double[test.size()][dims[g][r]];
					String [] tr_target = new String[train.size()];
					String [] te_target = new String[test.size()];


					for(int i = 0; i < train.size(); i++){
						String line = train.get(i);
						line = line.replaceAll("\"", "");
						line = line.split("]")[0];
						line = line.substring(1,line.length());
						String [] split = line.split(",");
						for(int d = 0; d < dims[g][r]; d++){
							tr_map[i][d] = Double.parseDouble(split[d]);
						}
						tr_target[i] = split[split.length-1];
					}

					for(int i = 0; i < test.size(); i++){
						String line = test.get(i);
						line = line.replaceAll("\"", "");
						line = line.split("]")[0];
						line = line.substring(1,line.length());
						String [] split = line.split(",");
						for(int d = 0; d < dims[g][r]; d++){
							te_map[i][d] = Double.parseDouble(split[d]);
						}
						te_target[i] = split[split.length-1];
					}

					//TODO pode ser acelerado se o treino e o teste forem feitos ao mesmo tempo.
					tr[g][r] = euclideanAccuracy(tr_map, tr_target,tr_map, tr_target);
					te[g][r] = euclideanAccuracy(tr_map, tr_target,te_map, te_target);				
				}
				
				System.out.println("[DONE]");
			}

			
			System.out.print("> Writting to :" + dir + FS + dataset + ".R ");
			FileWriter out = new FileWriter(new File (dir + FS + dataset + ".R"));

			out.write(matrixToR("training", tr));

			out.write("\n\n");
			out.write(matrixToR("test", te));

			out.write("\n\n");
			out.write(matrixToR_int("size", size));

			out.write("\n\n");
			out.write(matrixToR_int("dimensions", dims));
			
			med_goa = fixMatrix(med_goa);
			goa = fixMatrix(goa);
			
			out.write("\n");
			out.write("\n" + matrixToR("median_STXO", med_goa[0]));
			out.write("\n" + matrixToR("median_SWAPDIM", med_goa[1]));
			out.write("\n" + matrixToR("median_STMUT", med_goa[2]));
			out.write("\n" + matrixToR("median_ADDDIM", med_goa[3]));
			out.write("\n" + matrixToR("median_REMDIM", med_goa[4]));
			
			out.write("\n");
			out.write("\n" + matrixToR("best_STXO", goa[0]));
			out.write("\n" + matrixToR("best_SWAPDIM", goa[1]));
			out.write("\n" + matrixToR("best_STMUT", goa[2]));
			out.write("\n" + matrixToR("best_ADDDIM", goa[3]));
			out.write("\n" + matrixToR("best_REMDIM", goa[4]));

			out.close();

			System.out.println("[DONE]");
		}


	}
	
	private static double[][][] fixMatrix(double [][][]m){
		double [][][] ret = new double[m[0][0].length][m.length][m[0].length];
		for(int op = 0; op < 5; op++) {
			for(int gen = 0; gen < m.length; gen++) {
				for(int run = 0; run < m[0].length; run++) {
					ret[op][gen][run] = m[gen][run][op];
				}
			}
		}
		return ret;
	}



	private static void treatArgs(String [] args) {
		//$ReadJSON_18Dez dir:directory
		if(args.length >= 1) {
			String[] dir = args[0].split(":");
			properties.put(dir[0], new String[] {dir[1]});
		}
	}

	private static void readProperties() throws FileNotFoundException {
		Scanner in = new Scanner(new File(properties.get("dir")[0] + FS + PROPERTIES_FILENAME ));
		for(String line = null; in.hasNextLine();) {
			line = in.nextLine();
			properties.put(line.split("=")[0], line.split("=")[1].split(" "));
		}
		in.close();		
	}


	/**
	 * 
	 * @param in
	 * @param name "MedianGOA" or "GOA"
	 * @return
	 */
	private static double[] readNextArray(Scanner in, String name) {
		String line = in.nextLine();
		while( !line.contains("\""+name+"\":")) {
			line = in.nextLine();
		}
		String medGOA = line.substring(line.indexOf("[")+1, line.indexOf("]"));
		String [] ar = medGOA.split(", ");
		double [] ret = new double[ar.length];
		for(int i = 0; i < ret.length; i++) {
			ret[i] = Double.parseDouble(ar[i]);
		}
		return ret;
	}


	private static String matrixToR(String name, double[][] m) {
		StringBuilder sb = new StringBuilder();
		sb.append(name + " = matrix( c(");
		for(int y = 0; y < m[0].length; y++) {
			for(int x = 0; x < m.length; x++) {
				sb.append(m[x][y]+", ");
			}
			sb.append("\n");
		}
		sb.deleteCharAt(sb.length()-3);
		sb.append("),ncol = " + m[0].length + ", nrow = " + m.length + ")\n"); 
		return sb.toString();
	}

	private static String matrixToR_int(String name, int[][] m) {
		StringBuilder sb = new StringBuilder();
		sb.append(name + " = matrix( c(");
		for(int y = 0; y < m[0].length; y++) {
			for(int x = 0; x < m.length; x++) {
				sb.append(m[x][y]+", ");
			}
			sb.append("\n");
		}
		sb.deleteCharAt(sb.length()-3);
		sb.append("),ncol = " + m[0].length + ", nrow = " + m.length + ")\n"); 
		return sb.toString();
	}

	private static int[] countNodesAndDimensions(Scanner in) throws IOException {
		int nodes = 0;
		int dimensions = 0;

		while(!in.nextLine().contains("Dimensions"));
		String line = in.nextLine();
		for ( ; ! line.equals("            ],") ; line=in.nextLine()){
			nodes += (line.length() - line.replace("(", "").length())*2 +1;
			dimensions ++;
		}

		return new int [] {nodes, dimensions};
	}



	private static ArrayList<String> readTrain(Scanner in) throws IOException {
		while(!in.nextLine().contains("Train"));
		ArrayList<String> data = new ArrayList<String>();
		String line = in.nextLine();
		while (! line.equals("            ],")){
			data.add(line.substring(16));
			line=in.nextLine();
		}
		return data;
	}

	private static ArrayList<String> readTest(Scanner in) throws IOException {
		while(!in.nextLine().contains("Test"));
		ArrayList<String> data = new ArrayList<String>();
		String line = in.nextLine();
		while (! line.equals("            ]")){
			data.add(line.substring(16));
			line=in.nextLine();
		}
		return data;
	}

	private static double euclideanDistance(double [] d1,double [] d2) {
		double dist = 0;
		int len = d1.length;
		for (int i = 0; i < len; i++) {
			dist += Math.pow(d1[i]-d2[i], 2);
		}
		return Math.sqrt(dist);
	}

	private static double euclideanAccuracy(double[][] tr_map, String[] tr_target, double[][] map,
			String[] target) throws IOException {
		//CRIA CENTROIDS
		ArrayList<String> classes = new ArrayList<String>();
		for(int i = 0; i<tr_target.length; i++){
			if(!classes.contains(tr_target[i])){
				classes.add(tr_target[i]);
			}
		}

		double [][] centroid = new double [classes.size()][tr_map[0].length];
		double [] class_occ = new double [classes.size()];
		for(int i = 0; i < tr_target.length; i++){
			int index = classes.indexOf(tr_target[i]);
			class_occ[index] ++;
			for(int d = 0; d < tr_map[0].length; d++){
				centroid[index][d] += tr_map[i][d];
			}
		}
		for(int i = 0; i< class_occ.length; i++){
			for(int d = 0; d < tr_map[0].length; d++){
				centroid[i][d] /= class_occ[i];
			}
		}

		//CONFUSION MATRIX	
		int [][] cf = new int [classes.size()][classes.size()];

		//FAZ A PREVISAO
		double hits = 0;
		for(int i = 0; i< map.length; i++){
			String pred = classes.get(0);
			double distance = euclideanDistance(centroid[0], map[i]);
			for(int c = 1; c < classes.size(); c++){
				double d2 = euclideanDistance(centroid[c], map[i]);
				if (d2 < distance){
					distance = d2;
					pred = classes.get(c);
				}
			}
			if(pred.equals(target[i]))
				hits ++;


			cf [classes.indexOf(target[i])][classes.indexOf(pred)]++;

		}
		hits /= map.length;


		//System.out.println("Euclidean\nrun:"+r + "  gen:"+g);
		//writeMatrixCommentedR(cf);
		return hits;
	}
}
