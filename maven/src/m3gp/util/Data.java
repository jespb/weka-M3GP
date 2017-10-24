package m3gp.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Data {
	/**
	 * Reads a file and returns it's data
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static double[][] readData(String filename) throws IOException{
		BufferedReader in = new BufferedReader(new FileReader(filename));
		
		int n_lines = 0;
		for(String line = in.readLine();line != null; line = in.readLine(), n_lines++);
		in.close();
		
		in = new BufferedReader(new FileReader(filename));
		
		String line = in.readLine();
		double [][] data = new double [n_lines][line.split(";").length];

		for(int i = 0; i < n_lines; i++, line = in.readLine()){
			String [] split = line.split(";");
			for(int j = 0; j < split.length; j++)
				data[i][j] = Mat.parseDouble(split[j]);
		}
		in.close();
		
		return data;
	}
	
	/**
	 * Reads a file and returns the doubles on the first collum
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static String[] readTarget(String filename) throws IOException{
		BufferedReader in = new BufferedReader(new FileReader(filename));
		
		int n_lines = 0;
		for(String line = in.readLine();line != null; line = in.readLine(), n_lines++);
		in.close();
		
		String [] target = new String [n_lines];
		
		in = new BufferedReader(new FileReader(filename));
		int i = 0;
		for(String line = in.readLine(); line != null; line = in.readLine(), i++)
			target[i] = line;
		
		return target;
	}
}