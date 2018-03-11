package weka.classifiers.trees.m3gp.client;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;


public class ReadJSON {
	@SuppressWarnings("unchecked")
	public static void main(String [] args) throws Exception {
		int nruns= 30;
		double [] tr_acc = new double[nruns];
		double [] te_acc = new double[nruns];
		double [][] tr_points = new double [0][0];
		double [][] te_points = new double [0][0];

		int ds = 1, fun=1, pred = 1;
		String dataset = "brazil heart waveform vowel".split(" ")[ds];
		String sig = "BRZ HRT WAV VOW".split(" ")[ds];
		
		BufferedWriter scatterplot = new BufferedWriter(new FileWriter(dataset+"_scatterplot.txt"));
		Object []  o = readTrainPoint("resultados\\"+sig+"_"+fun+"_"+pred+"\\"+"Run_0_"+dataset+".json");
		ArrayList<double[]> points = (ArrayList<double[]>) o[0];
		ArrayList<String> classes = (ArrayList<String>) o[1];
		scatterplot.write("rows = " + points.size() + "\n");
		scatterplot.write("cols = " + (points.get(0).length+1) + "\n");
		scatterplot.write("m <- matrix(nrow=rows, ncol=cols) \n" );
		for(int i = 0; i < classes.size(); i++) {
			scatterplot.write("m[" + (i+1) + ",] = c(");
			for(int j = 0; j < points.get(0).length; j++) {
				scatterplot.write( points.get(i)[j] + ",");
			}
			scatterplot.write(classes.get(i) + ")\n");
		}
		scatterplot.write("pairs( m[,1:cols-1], data=m, col = m[,cols])\n");
		scatterplot.close();
		
		
		
		for(int i = 0; i < nruns; i++) {
			String filename = "resultados\\"+sig+"_"+fun+"_"+pred+"\\"+"Run_"+i+"_"+dataset+".json";
			Object [] tmp = confusionMatrix(filename);
			ArrayList<int[][]> train = (ArrayList<int[][]>) tmp[0];
			ArrayList<int[][]> test = (ArrayList<int[][]>) tmp[1];
			int [][] train_mat = train.get(train.size()-1);
			int [][] test_mat = test.get(test.size()-1);
			printMatrix(test_mat);
			double train_acc = sum_diag(train_mat) / sum(train_mat);
			double test_acc = sum_diag(test_mat) / sum(test_mat);
			System.out.println(train_acc + " " + test_acc);

			tr_acc[i] = train_acc;
			te_acc[i] = test_acc;
		}
		
		BufferedWriter boxplot = new BufferedWriter(new FileWriter(dataset+"_boxplot.txt"));
		
		boxplot.write("test <- c(");
		
		boxplot.write(""+te_acc[0]);
		for(int i = 1; i < te_acc.length; i++) {
			boxplot.write("," + te_acc[i]);
		}
		boxplot.write(")\ntrain <- c(" + tr_acc[0]);
		for(int i = 1; i < tr_acc.length; i++) {
			boxplot.write("," + tr_acc[i]);
		}
		boxplot.write(")\nboxplot(test,train)");
		boxplot.close();
		
		/*
		BufferedWriter out = new BufferedWriter(new FileWriter("lixo.csv"));
		ArrayList<Integer> dimensions = numberOfDimensions(filename);
		Iterator<Integer> i= dimensions.iterator();
		while(i.hasNext()) {
			out.write(i.next()+"\n");
		}
		out.close();
		 */
	}






	private static double sum_diag(int[][] m) {
		double d = 0;
		for(int y = 0; y < m.length; y++)
			d += m[y][y];
		return d;
	}

	private static double sum(int[][] m) {
		double d = 0;
		for(int y = 0; y < m.length; y++)
			for(int x=0; x < m[0].length; x++)
				d += m[y][x];
		return d;
	}


	static Object[] readTrainPoint(String filename) throws Exception{
		BufferedReader in = new BufferedReader(new FileReader(filename));
		ArrayList<double[]> points = new ArrayList<double[]>();
		ArrayList<String> classes = new ArrayList<String>();

		boolean trainMat = true;

		for(String line = in.readLine(); line != null; line = in.readLine()) {
			if(line.contains("Train")){
				trainMat = true;

				points = new ArrayList<double[]>();
				classes = new ArrayList<String>();
			}
			if(line.contains("Test")){
				trainMat = false;
			}
			if(line.startsWith("                [")) {
				String [] split = line.split("\",\"");
				String target = split[split.length-1].split("\"")[0];
				split[0] = split[0].substring(20);
				String pred = split[split.length-2];
				if(trainMat) {
					classes.add(pred);
					double [] d = new double[split.length-2];
					for(int i = 0; i < d.length; i++) {
						d[i] = Double.parseDouble(split[i]);
					}
					points.add(d);
				}
			}
		}
		in.close();
		return new Object[] {points, classes};
	}





	static ArrayList<Integer> numberOfDimensions(String filename) throws Exception {
		BufferedReader in = new BufferedReader(new FileReader(filename));
		ArrayList<Integer> dimensions = new ArrayList<Integer>();
		int count = 0;
		boolean d = false;

		for(String line = in.readLine(); line != null; line = in.readLine()) {
			if(line.contains("Dimension")) {
				count = -1;
				d = true;
			}
			if(line.equals("            ],") && d) {
				dimensions.add(count);
				d = false;
			}
			count ++;
		}

		in.close();
		return dimensions;
	}

	static ArrayList<Integer> numberOfNodes(String filename)  throws Exception{
		BufferedReader in = new BufferedReader(new FileReader(filename));
		ArrayList<Integer> nodes = new ArrayList<Integer>();
		int count = 0;
		boolean d = false;

		for(String line = in.readLine(); line != null; line = in.readLine()) {
			if(line.contains("Dimension")) {
				count = 0;
				d = true;
			}
			if(d) {
				count += (line.length() - line.replace("+", "").replace("*", "").replace("-", "").replace("/", "").length())*2+1;
			}
			if(line.equals("            ],") && d) {
				nodes.add(count);
				d = false;
			}
		}

		in.close();
		return nodes;
	}

	private static Object[] confusionMatrix(String filename) throws Exception{
		BufferedReader in = new BufferedReader(new FileReader(filename));
		ArrayList<String> classes = new ArrayList<String>();

		for(String line = in.readLine(); line != null; line = in.readLine()) {
			if(line.startsWith("                [")) {
				String [] split = line.split("\",\"");
				String target = split[split.length-1].split("\"")[0];
				if(!classes.contains(target)) {
					classes.add(target);
				}
			}
		}
		in.close();
		in = new BufferedReader(new FileReader(filename));


		ArrayList<int[][]> trains = new ArrayList<int[][]>();
		ArrayList<int[][]> tests = new ArrayList<int[][]>();

		int [][] train = new int [classes.size()][classes.size()];
		int [][] test = new int [classes.size()][classes.size()];
		boolean trainMat = true;
		int count = 0;

		for(String line = in.readLine(); line != null; line = in.readLine()) {
			if(line.contains("Train")){
				trainMat = true;
				count = 0;
				trains.add(train);
				tests.add(test);

				train = new int [classes.size()][classes.size()];
				test = new int [classes.size()][classes.size()];
			}
			if(line.contains("Test")){
				trainMat = false;
			}
			if(line.startsWith("                [")) {
				String [] split = line.split("\",\"");
				String target = split[split.length-1].split("\"")[0];
				String pred = split[split.length-2];
				if(pred.equals(target)) {
					count++;
				}
				if(trainMat) {
					train[classes.indexOf(pred)][classes.indexOf(target)]++;
				}else {
					test[classes.indexOf(pred)][classes.indexOf(target)]++;
				}
			}
		}


		in.close();

		return new Object[] {trains, tests};
	}







	private static void printMatrix(int[][] m) {
		for(int y = 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				System.out.print(m[y][x]+" ");
			}
			System.out.println();
		}
	}
}
