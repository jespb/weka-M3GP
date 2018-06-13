package weka.classifiers.trees.m3gp.client;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Matrix;

//INDEPENDENT CLASS
public class ReadJSON2 {
	private static String fs = File.separator;

	private static int ds = 8;
	private static int dir = 2;
	private static String dataset = "mcd3 mcd10 heart movl seg sonarall vowel wav yeast breast-cancer-wisconsin parkinsons pima-indians-diabetes ionosphere".split(" ")[ds];
	private static String dirname = new String[] {"paper"+ fs + dataset+fs, "knn_"+dataset+fs, ""}[dir];

	private static int gens = 100;
	private static int runs = 5;

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

		int [][] nodes = new int[gens][runs];
		int [][] dimensions = new int[gens][runs];

		double[][][] accs = new double[getAccuracyCount()][gens][runs];

		
		for(r = 0; r < runs; r++){
			filename = dirname  + "Run_" + r + "_" + dataset + ".json";
			System.out.println("Reading: " + filename);

			BufferedReader in = new BufferedReader(new FileReader(filename));

			// Só guarda o mapa das ultimas geracoes
			// mapa das dimensoes
			// mapa do numero de nós
			for(g = 0; g < gens; g++){
				ArrayList<String> train = new ArrayList<String>();
				ArrayList<String> test = new ArrayList<String>();

				int [] nodes_dimensions = countNodesAndDimensions(in);

				nodes [g][r] = nodes_dimensions[0];
				dimensions [g][r] = nodes_dimensions[1];

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
					//System.out.println(split.length);//TODO DELETE
					for(int d = 0; d < n_dim; d++){
						te_map[i][d] = Double.parseDouble(split[d]);
					}
					te_target[i] = split[split.length-1];
				}

				
				int a=0;

				if(euclidean) {
					//out.write("# euclidean accuracy for run " + r +"\n");
					double tr_euclidean_acc = euclideanAccuracy(tr_map, tr_target,tr_map, tr_target);
					double te_euclidean_acc = euclideanAccuracy(tr_map, tr_target,te_map, te_target);
					//System.out.println("  euclidian accuracy: " + tr_euclidean_acc + ", " + te_euclidean_acc);
					accs[a++][g][r] = tr_euclidean_acc;
					accs[a++][g][r] = te_euclidean_acc;
				}

				if(mahalanobis) {
					//out.write("# mahalanobis accuracy for run " + r +"\n");
					double tr_mahalanobis_acc = mahalanobisAccuracy(tr_map, tr_target,tr_map, tr_target);
					double te_mahalanobis_acc = mahalanobisAccuracy(tr_map, tr_target,te_map, te_target);
					//System.out.println("  mahalanobis accuracy: " + tr_mahalanobis_acc + ", " + te_mahalanobis_acc); 
					accs[a++][g][r] = tr_mahalanobis_acc;
					accs[a++][g][r] = te_mahalanobis_acc;
				}

				if(knn) {
					//out.write("# knn accuracy for run " + r +"\n");
					double tr_knn_acc = knnAccuracy(tr_map, tr_target,tr_map, tr_target,5);
					double te_knn_acc = knnAccuracy(tr_map, tr_target,te_map, te_target,5);
					//System.out.println("  knn accuracy: " + tr_knn_acc + ", " + te_knn_acc); 
					accs[a++][g][r] = tr_knn_acc;
					accs[a++][g][r] = te_knn_acc;
				}
				
				train.clear();
				test.clear();

			}	

			in.close();
		}

		System.out.print("Writing nodes matrix...");
		out.write("# Number of nodes (geration, run): \n");
		//Matrix.printMatrix(nodes);
		out.write(matrixToR("nodes", nodes));
		out.write("\n\n");
		System.out.println("(done)");


		System.out.print("Writing dimensions matrix...");
		out.write("# Number of dimensions (geration, run): \n");
		out.write(matrixToR("dimensions", dimensions));
		out.write("\n\n");
		System.out.println("(done)");
		
		
		


		System.out.print("Writing train accuracy...");
		for(int i = 0; i<accs.length;i+=2){
			out.write(matrixToR("tr_"+getNames()[i/2],accs[i]));
		}
		System.out.println("(done)");


		System.out.print("Writing test accuracy...");
		for(int i = 1; i<accs.length;i+=2){
			out.write(matrixToR("te_"+getNames()[i/2],accs[i]));
		}
		System.out.println("(done)");

		out.close();
	}

	private static String matrixToR(String name, int[][] m) {
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

	private static String getAccuracyNames() {
		String s = "";
		switch( getAccuracyCount() ) {
		case 0:
			s = "< No method was selected >";
			break;
		case 1:
			if (euclidean)
				s = "'Euclidean'";
			else if (mahalanobis)
				s = "'Mahalanobis'";
			else
				s = "'Knn'";
			break;
		default:
			boolean first = true;
			if (euclidean) {
				s += "'Euclidean'";
				first = false;
			}else if(mahalanobis) {
				if (!first) s+=", ";
				s += "'Mahalanobis'";
			}else{
				if (!first) s+=", ";
				s += "'Knn'";
			}
		}
		return s;
	}
	
	private static String[] getNames() {
		ArrayList<String> s = new ArrayList<String>();
		if (euclidean) 
			s.add("Euclidean");
		if(mahalanobis) 
			s.add("Mahalanobis");
		if (knn)
			s.add("Knn");
		String [] s2 = new String[s.size()];
		for (int i = 0; i < s2.length; i++){
			s2[i] = s.get(i);
		}
		return  s2;
	}

	private static String c(double[] d) {
		StringBuilder sb = new StringBuilder();
		sb.append("c(" + d[0]);
		for(int i = 1; i< d.length; i++){
			sb.append(", " + d[i]);
		}
		sb.append(")");
		return sb.toString();
	}

	private static double knnAccuracy(double[][] tr_map, String[] tr_target, double[][] map, String[] target,
			int knn) throws IOException {
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
			double [] distances = new double [tr_map.length];
			for(int d = 0; d < distances.length; d++){
				distances[d] = euclideanDistance(map[i], tr_map[d]);
			}
			String [] target_dup = clone(tr_target);

			Arrays.mergeSortBy(target_dup,distances);

			String [] options = new String[knn];
			for(int op = 0; op < knn; op++) {
				options[op] = target_dup[op];
			}

			String pred = Arrays.mostCommon(options);

			if(pred.equals(target[i]))
				hits ++;

			cf [classes.indexOf(target[i])][classes.indexOf(pred)]++;
		}
		hits /= map.length;

		//System.out.println("knn\nrun:"+r + "  gen:"+g  + "   hits:" + hits);
		//writeMatrixCommentedR(cf);
		return hits;
	}

	private static String[] clone(String[] v) {
		String [] dup = new String [v.length];
		for(int i = 0; i < v.length; i++){
			dup[i]=v[i];
		}
		return dup;
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

	private static double mahalanobisAccuracy(double[][] tr_map, String[] tr_target, double[][] map,
			String[] target) throws IOException {
		//CRIA CENTROIDS
		ArrayList<String> classes = new ArrayList<String>();
		ArrayList<ArrayList<double []>> clusters = new ArrayList<ArrayList<double[]>>();

		for(int i = 0; i<tr_target.length; i++){
			if(!classes.contains(tr_target[i])){
				classes.add(tr_target[i]);
				clusters.add(new ArrayList<double[]>());
			}
		}

		for(int i = 0; i < tr_target.length; i++){
			int index = classes.indexOf(tr_target[i]);
			clusters.get(index).add(tr_map[i]);
		}

		ArrayList<double[][]> covarianceMatrix = new ArrayList<double[][]>();
		for(int i = 0; i<clusters.size(); i++) {
			covarianceMatrix.add(Matrix.covarianceMatrix(clusters.get(i)));
		}

		ArrayList<double[]> mu = new ArrayList<double[]>();
		for(int i = 0; i < clusters.size();i++){
			mu.add(new double[clusters.get(i).get(0).length]);
			for(int j = 0; j < clusters.get(i).size(); j++){
				for(int k = 0; k < mu.get(i).length; k++) {
					mu.get(i)[k] += clusters.get(i).get(j)[k];
				}
			}
			for(int j = 0; j < mu.get(i).length; j++){
				mu.get(i)[j] /= clusters.get(i).size();
			}
		}

		//CONFUSION MATRIX	
		int [][] cf = new int [classes.size()][classes.size()];

		//FAZ A PREVISAO
		double hits = 0;
		for(int i = 0; i< map.length; i++){
			double [] distancias = new double[classes.size()];
			for(int d = 0; d < distancias.length; d++) {
				distancias[d] = Arrays.mahalanobisDistance(map[i], 
						mu.get(d), covarianceMatrix.get(d));
			}

			double minDist = distancias[0];
			String pred = classes.get(0);
			for(int d = 0; d < distancias.length; d++) {
				if(distancias[d] < minDist) {
					minDist = distancias[d];
					pred = classes.get(d);
				}
			}

			if(pred.equals(target[i]))
				hits ++;


			cf [classes.indexOf(target[i])][classes.indexOf(pred)]++;
		}
		hits /= map.length;

		//System.out.println("Mahalanobis\nrun:"+r + "  gen:"+g);
		//writeMatrixCommentedR(cf);
		return hits;
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
}
