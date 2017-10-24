package m3gp.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Files {
	/**
	 * Fixes the .csv file by replacing the periods with commas
	 * @param filename
	 * @throws IOException
	 */
	public static void fixCSV(String filename) throws IOException{
		String outputName = filename;
		
		BufferedReader in = new BufferedReader(new FileReader(filename+".tmp"));
		BufferedWriter out = new BufferedWriter(new FileWriter(outputName));

		for(String line = in.readLine(); line != null; line = in.readLine()){
			out.write( line.replace(".", ",")+"\n");
		}

		in.close();
		out.close();
		
		File f = new File(filename + ".tmp");
		f.delete();
	}
}