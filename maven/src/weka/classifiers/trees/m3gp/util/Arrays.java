package weka.classifiers.trees.m3gp.util;

import java.util.ArrayList;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Arrays {
	public static double euclideanDistance(double [] d1,double [] d2) {
		double dist = 0;
		int len = d1.length;
		for (int i = 0; i < len; i++) {
			dist += Math.pow(d1[i]-d2[i], 2);
		}
		return Math.sqrt(dist);
	}

	/**
	 * Calcula a distancia de mahalanobis de x com o cluster
	 * DM(x) = (x - mu)^T * S^-1 * (x-mu)
	 * @param x
	 * @param c
	 * @return
	 */
	public static double mahalanobisDistance(double [] x,double [] mu, double[][] s) {
		// DM(x) = (x - mu)^T * S^-1 * (x-mu)
		// DM(x) = a * bInv * c

		//Calcula a e c
		double[][]a= new double[1][x.length];
		double[][]c= new double[x.length][1];
		for(int i= 0; i< x.length; i++){
			a[0][i] = x[i]-mu[i];
			c[i][0] = x[i]-mu[i];
		}
		

		double [][] sInv = 	Matrix.inverseMatrix(s);
		if ( sInv[0][0] == Double.NaN || (sInv[0][0]+"").equals("NaN") || sInv[0][0] == Double.POSITIVE_INFINITY || sInv[0][0] == Double.NEGATIVE_INFINITY) {
			sInv = Matrix.moorepenroseInverseMatrix(s);
			if(sInv == null) {
				return euclideanDistance(x,mu);
			}
		}

		double [][] a_sInv = Matrix.multiply(a,sInv);
		double [][] result = Matrix.multiply(a_sInv, c);

		return Math.sqrt(result[0][0]);
	}

	/**
	 * Shuffle two arrays in the same way
	 * @param data
	 * @param target
	 */
	public static void shuffle(Object[] data, Object[] target) {
		int n = data.length;
		for (int i = 0; i < n; i++) {
			// Get a random index of the array past i.
			int random = i + Mat.random(n-i);
			// Swap the random element with the present element.
			Object randomElement = data[random];
			data[random] = data[i];
			data[i] = randomElement;

			Object randomEl = target[random];
			target[random] = target[i];
			target[i] = randomEl;
		}
	}

	/**
	 * Uses merge sort to sort both arrays by the values in a
	 * @param o object array
	 * @param a array with values
	 */
	public static void mergeSortBy(Object[]o, double[]a){
		double [] b  = a.clone();
		Object [] bo = o.clone();
		topDownSplitMerge(b,bo,0,a.length,a,o);
		topDownMerge(a,o, 0, a.length/2, a.length, b,bo);
	}

	/**
	 * Sub-method of mergeSortBy
	 * @param b
	 * @param bo
	 * @param min
	 * @param max
	 * @param a
	 * @param o
	 */
	private static void topDownSplitMerge(double[] b, Object [] bo, int min, int max, double[] a, Object [] o) {
		if(max-min < 2)
			return;
		int mean = (max+min)/2;

		topDownSplitMerge(a,o, min, mean, b,bo);
		topDownSplitMerge(a,o, mean, max, b,bo);
		topDownMerge(b,bo, min, mean, max, a,o);
	}

	/**
	 * Sub-method of mergeSortBy
	 * @param b
	 * @param bo
	 * @param min
	 * @param mean
	 * @param max
	 * @param a
	 * @param o
	 */
	private static void topDownMerge(double[] b, Object [] bo, int min, int mean, int max, double[] a, Object [] o) {
		int i = min, j = mean;
		for(int k = min; k < max; k++){
			if(i<mean && (j >= max || a[i] <= a[j])){
				b[k] = a[i];
				bo[k] = o[i];
				i++;
			}else{
				b[k] = a[j];
				bo[k] = o[j];
				j++;
			}
		}
	}
	
	public static boolean isValid(double [][] m) {
		boolean valid = true;
		int i = 0, j=0;
		while (i < m.length && valid) {
			while ( j < m[0].length && valid) {
				valid = m[i][j] != Double.NaN && m[i][j] != Double.NEGATIVE_INFINITY && m[i][j] != Double.POSITIVE_INFINITY;
			}
		}
		return valid;
	}

	public static double[] sum(double[] ds, double[] ds2, double d) {
		int n = ds.length;
		double[] ret = new double[n];

		for(int i = 0; i < n; i++) {
			ret[i] = ds[i]+ds2[i]*d;
		}
		return ret;
	}

	public static double[] multiply(double[] v, double d) {
		int n = v.length;
		double[] ret = new double[n];
		for(int i = 0; i < n; i++) {
			ret[i] = v[i]*d;
		}
		return ret;
	}

	public static double min(double[] v) {
		double min = v[0];
		for(int i = 0 ; i < v.length; i++)
			min = Math.min(min, v[i]);
		return min;
	}

	public static double manhattanDistance(double[] ds, double[] ds2) {
		double distance = 0;
		for (int i = 0; i < ds.length; i++) {
			distance += Math.abs(ds[i]-ds2[i]);
		}
		return distance;
	}

	public static String mostCommon(String[] s) {
		ArrayList<String> str = new ArrayList<String>();
		int [] ints = new int[s.length];
		for(int i = 0; i < s.length; i++) {
			if(str.contains(s[i])) {
				ints[str.indexOf(s[i])]++;
			}else {
				str.add(s[i]);
				ints[str.indexOf(s[i])]++;
			}
		}
		String predict = str.get(0);
		int occ = ints[0];
		for(int i = 1; i < ints.length; i++) {
			if(ints[i] > occ) {
				occ=ints[i];
				predict = str.get(i);
			}
		}
		return predict;
	}

	public static String arrayToString(int[] v) {
		StringBuilder sb = new StringBuilder();
		sb.append("[" + v[0]);
		for(int i = 1; i < v.length; i++) {
			sb.append(", " + v[i]);
		}
		sb.append("]");
		return sb.toString();
	}


	public static String arrayToString(double[] v) {
		StringBuilder sb = new StringBuilder();
		sb.append("[" + v[0]);
		for(int i = 1; i < v.length; i++) {
			sb.append(", " + v[i]);
		}
		sb.append("]");
		return sb.toString();
	}

	public static double[] multiply(int[] v, double d) {
		int n = v.length;
		double[] ret = new double[n];
		for(int i = 0; i < n; i++) {
			ret[i] = v[i]*d;
		}
		return ret;
	}

	public static double[] normalize(double[] np) {
		double acc = 0;
		for(double d : np) {
			acc += d;
		}
		double [] ret = new double [np.length];
		for(int i = 0; i < np.length; i++) {
			ret[i] = np[i]/acc;
		}
		return ret;
	}

	public static double sumvals(double[] d) {
		double acc = 0;
		for (double i : d)
			acc += i;
		return acc;
	}
	
	public static double[] copy(double[]d) {
		double[] ret = new double[d.length];
		for(int i = 0; i < d.length; i++) {
			ret[i]=d[i];
		}
			return ret;
	}

	public static double median(double[] ds) {
		java.util.Arrays.sort(ds);
		if (ds.length %2 != 0)
		return ds[ds.length/2];
		else
			return (ds[ds.length/2]+ds[ds.length/2 -1])/2.0;
	}

}