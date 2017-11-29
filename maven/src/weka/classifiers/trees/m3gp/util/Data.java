package weka.classifiers.trees.m3gp.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Data {
	private static String label_separator = ",";
	
	public static Object[] readDataTarget(String filename) throws IOException {
		double [][] data;
		String [] target;
		BufferedReader in = new BufferedReader(new FileReader(filename));
	
		String line = in.readLine();
		int n_lines = 1;
		int n_labels = line.split(label_separator).length;
		for(line = in.readLine();line != null; line = in.readLine(), n_lines++);
		in.close();
		
		target = new String [n_lines];
		data = new double [n_lines][n_labels];
		
		in = new BufferedReader(new FileReader(filename));
		int i = 0;
		for(line = in.readLine(); line != null; line = in.readLine(), i++){
			String [] split = line.split(label_separator);
			for(int j = 0; j < split.length-1; j++)
				data[i][j] = Mat.parseDouble(split[j]);
			target[i] = split[n_labels-1];
		}
		return new Object[]{data, target};
	}
}