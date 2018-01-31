package weka.classifiers.trees.m3gp.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Mat{
	private static Random r = new Random();
	/**
	 * Returns a random int from 0 to n exclusive
	 * @param n
	 * @return
	 */
	public static int random(int n){
		return r.nextInt(n);
	}

	/**
	 * Converts a String to a double
	 * @param s
	 * @return
	 */
	public static double parseDouble(String s){
		if(s.contains("e")){
			String [] split = s.split("e");
			return Double.parseDouble(split[0]) * Math.pow(10, Double.parseDouble(split[1]));
		}else{
			if(s.length()>3&&s.charAt(s.length()-3)=='-'){
				return Math.pow(Double.parseDouble(s.substring(0, s.length()-3)), Double.parseDouble(s.substring(s.length()-3)));
			}else{
				return Double.parseDouble(s);
			}
		}
	}

	/**
	 * Return the median of a double arraylist
	 * @param al
	 * @return
	 */
	public static double median(ArrayList<Double> al) {
		Collections.sort(al);
		return al.get((int)(al.size()/2));
	}

	public static double sigmod(double d) {
		return d/(1+Math.abs(d));
	}
}