package weka.classifiers.trees.m3gp.client;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Iterator;


public class ReadJSON {
	public static void main(String [] args) throws Exception {
		String filename = "Run_0_heart.json";
		
		confusionMatrix(filename);
		
		/*BufferedWriter out = new BufferedWriter(new FileWriter("lixo.csv"));
		ArrayList<Integer> dimensions = numberOfNodes(filename);

		Iterator<Integer> i= dimensions.iterator();
		while(i.hasNext()) {
			out.write(i.next()+"\n");
		}
		
		out.close();*/
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
					System.out.println(target);
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
				System.out.println("XXXXX");
			}
			if(line.contains("Test")){
				trainMat = false;
			}
			if(line.startsWith("                [")) {
				String [] split = line.split("\",\"");
				String target = split[split.length-1].split("\"")[0];
				String pred = split[split.length-2];
				System.out.println(pred + " " + target);
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
