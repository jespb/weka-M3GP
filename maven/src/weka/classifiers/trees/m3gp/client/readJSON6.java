package weka.classifiers.trees.m3gp.client;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;


//INDEPENDENT CLASS
public class readJSON6 {
	private static String fs = File.separator;

	private static int ds = 0;
	private static int dir = 2;
	private static String dataset = "heart mcd3 mcd10 movl seg vowel wav yeast".split(" ")[ds];
	private static String dirname = new String[] {"paper"+ fs + dataset+fs, "knn_"+dataset+fs, ""}[dir];

	private static int gens = 100;
	private static int runs = 1;

	private static int g=0, r=0;

	private static boolean euclidean = false;
	private static boolean mahalanobis = false;
	private static boolean knn = false;

	private static BufferedWriter out = null;

	public static void main(String [] args) throws IOException{
		treatArgs(args);
		out = new BufferedWriter(new FileWriter("results_"+ dataset + ".r"));

		System.out.println("Reading the results from: " + dataset);		
		System.out.println("...");

		String filename = "";

		double[][] tr = new double[gens][runs];
		double[][] te = new double[gens][runs];
		int[][] size = new int[gens][runs];
		int[][] dims = new int[gens][runs];

		
		for(r = 0; r < runs; r++){
			filename = dirname  + "Run_" + r + "_" + dataset + ".json";
			System.out.println("Reading: " + filename);

			BufferedReader in = new BufferedReader(new FileReader(filename));

			// S guarda o mapa das ultimas geracoes
			// mapa das dimensoes
			// mapa do numero de nós
			for(g = 0; g < gens; g++){
				ArrayList<String> train = new ArrayList<String>();
				ArrayList<String> test = new ArrayList<String>();
				
				int [] nodes_dimensions = countNodesAndDimensions(in);
				//double [][] tmp = readGOA2(in);
				
				size[g][r] = nodes_dimensions[0];
				dims[g][r] = nodes_dimensions[0];

				train = readTrain(in);
				test = readTest(in);

				int n_dim = train.get(0).split(",\"").length -1;

				double [][] tr_map = new double[train.size()][n_dim];
				double [][] te_map = new double[test.size()][n_dim];
				String [] tr_target = new String[train.size()];
				String [] te_target = new String[test.size()];

				
				for(int i = 0; i < train.size(); i++){
					String line = train.get(i);
					line = line.replaceAll("\"", "");
					line = line.split("]")[0];
					line = line.substring(1,line.length());
					String [] split = line.split(",");
					for(int d = 0; d < n_dim; d++){
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
					for(int d = 0; d < n_dim; d++){
						te_map[i][d] = Double.parseDouble(split[d]);
					}
					te_target[i] = split[split.length-1];
				}

				if(euclidean) {
					//out.write("# euclidean accuracy for run " + r +"\n");
					double tr_euclidean_acc = euclideanAccuracy(tr_map, tr_target,tr_map, tr_target);
					double te_euclidean_acc = euclideanAccuracy(tr_map, tr_target,te_map, te_target);
					//System.out.println("  euclidian accuracy: " + tr_euclidean_acc + ", " + te_euclidean_acc);
					tr[g][r] = tr_euclidean_acc;
					te[g][r] = te_euclidean_acc;
				}				
				
				train.clear();
				test.clear();

			}	

			in.close();
		}

				out.write("\n\n");
		out.write(matrixToR("training", tr));

		out.write("\n\n");
		out.write(matrixToR("test", te));

		out.write("\n\n");
		out.write(matrixToR_int("size", size));

		out.write("\n\n");
		out.write(matrixToR_int("dimensions", dims));
		
		out.close();
	}

	private static double[][] readGOA2(BufferedReader in) throws IOException {
		String line = in.readLine();
		while( !line.startsWith("            \"GOA")) {
			line = in.readLine();
		}
		int size = line.split(",").length;
		double [] goa = new double[size];
		double [] goac = new double[size];
		line = line.substring(line.indexOf("[")+1, line.indexOf("]"));
		String [] split1 = line.split(", ");
		line=in.readLine();
		line = line.substring(line.indexOf("[")+1, line.indexOf("]"));
		String [] split2 = line.split(", ");
		for(int i = 0; i < split1.length; i++) {
			goa[i] = Double.parseDouble(split1[i]);
			goac[i] = Double.parseDouble(split2[i]);
		}
		return new double[][]{goa, goac};
	}

	private static double[] readGOA1(BufferedReader in) throws IOException {
		String line = in.readLine();
		while( !line.startsWith("        \"GOA") && !line.startsWith(",        \"GOAC")) {
			line = in.readLine();
		}
		int size = line.split(",").length;
		double [] goac = new double[size-1];
		line = line.substring(line.indexOf("[")+1, line.indexOf("]"));
		String [] split = line.split(", ");
		for(int i = 0; i < goac.length; i++) {
			goac[i] = Double.parseDouble(split[i]);
		}
		return goac;
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

	private static int[] countNodesAndDimensions(BufferedReader in) throws IOException {
		int nodes = 0;
		int dimensions = 0;

		while(!in.readLine().contains("Dimensions"));
		ArrayList<String> data = new ArrayList<String>();
		String line = in.readLine();
		for ( ; ! line.equals("            ],") ; line=in.readLine()){
			nodes += (line.length() - line.replace("(", "").length())*2 +1;
			dimensions ++;
		}

		return new int [] {nodes, dimensions};
	}

	private static int getAccuracyCount() {
		return (euclidean?2:0)+(mahalanobis?2:0)+(knn?2:0);
	}



	private static ArrayList<String> readTrain(BufferedReader in) throws IOException {
		while(!in.readLine().contains("Train"));
		ArrayList<String> data = new ArrayList<String>();
		String line = in.readLine();
		while (! line.equals("            ],")){
			data.add(line.substring(16));
			line=in.readLine();
		}
		return data;
	}

	private static ArrayList<String> readTest(BufferedReader in) throws IOException {
		while(!in.readLine().contains("Test"));
		ArrayList<String> data = new ArrayList<String>();
		String line = in.readLine();
		while (! line.equals("            ]")){
			data.add(line.substring(16));
			line=in.readLine();
		}
		return data;
	}

	public static double euclideanDistance(double [] d1,double [] d2) {
		double dist = 0;
		int len = d1.length;
		for (int i = 0; i < len; i++) {
			dist += Math.pow(d1[i]-d2[i], 2);
		}
		return Math.sqrt(dist);
	}

	public static void writeMatrixCommentedR(int [][] d) throws IOException {
		for(int z =0; z < d.length; z++) {
			out.write("#");
			for(int x = 0; x < d[0].length; x++) {
				out.write(leadingSpaces(d[z][x])+", ");
			}
			out.write("\n");
		}
		out.write("\n");
	}

	private static String leadingSpaces(int i) {
		StringBuilder sb = new StringBuilder();
		int spaces = 6-(i+"").length();
		for(int k = 0; k < spaces; k++) {
			sb.append(" ");
		}
		sb.append(i);
		return sb.toString();
	}

	private static void treatArgs(String [] args) {
		//$ReadJSON2.java directory dataset numberOfRuns -e -k -m
		if(args.length > 1) {
			dirname = args[0];
			dataset = args[1];
			runs = Integer.parseInt(args[2]);
			for(int i = 3; i < args.length; i++) {
				switch(args[i]) {
				case "-e":
					euclidean = true;
					break;
				case "-m":
					mahalanobis = true;
					break;
				case "-k":
					knn = true;
					break;
				}
			}
		}else {
			euclidean = true;
			mahalanobis = true;
			knn = true;
		}
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
