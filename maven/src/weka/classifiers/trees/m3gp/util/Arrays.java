package weka.classifiers.trees.m3gp.util;

import java.util.ArrayList;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Arrays {
	/**
	 * Calcula a distancia de mahalanobis de x com o cluster
	 * DM(x) = (x - mu)^T * S^-1 * (x-mu)
	 * @param x
	 * @param c
	 * @return
	 */
	public static double mahalanobisDistance(double [] x,double [] mu, double[][] b) {
		// DM(x) = (x - mu)^T * S^-1 * (x-mu)
		// DM(x) = a * bInv * c
		
		//Calcula a e c
		double[][]a= new double[1][x.length];
		double[][]c= new double[x.length][1];
		for(int i= 0; i< x.length; i++){
			a[0][i] = x[i]-mu[i];
			c[i][0] = x[i]-mu[i];
		}
		double [][] bInv = Matrix.inverseMatrix(Matrix.clone(b));
		
		if( (bInv[0][0]+"").equals("NaN") )	{
			bInv = Matrix.moorepenroseInverseMatrix(b);
			if(bInv==null) {
				bInv = b;
			}
			//bInv = Matrix.minorDiagonal(bInv.length);
			//bInv = Matrix.transposta(b);
			//bInv = b;
			//bInv = Matrix.fill(bInv.length,  1);
			//bInv = Matrix.identity(bInv.length);
		}
		
		double [][] a_bInv = Matrix.multiply(a,bInv);
		double [][] result = Matrix.multiply(a_bInv, c);
//		System.out.println(result[0][0]);
		return result[0][0];		
	}
	
	/**
	 * Shuffle two arrays in the same way
	 * @param data
	 * @param target
	 */
	public static void shuffle(Object[] data, Object[] target) {
		int n = data.length;
		for (int i = 0; i < data.length; i++) {
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
}