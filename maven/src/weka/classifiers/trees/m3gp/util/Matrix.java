package weka.classifiers.trees.m3gp.util;

import java.util.ArrayList;

public class Matrix {

	public static double[][] identity(int n){
		double [][] m = new double[n][n];
		for(int i = 0; i < n;i++)
			m[i][i] = 1;
		return m;
	}
	
	/**
	 * Calcula a matriz de covariancia do cluster
	 * @param cluster
	 * @return
	 */
	public static double[][] covarianceMatrix(ArrayList<double[]> cluster) {
		double [][] a = new double[cluster.get(0).length][cluster.size()];
		double [][] b = new double[cluster.size()][cluster.get(0).length];
		
		for(int y = 0; y < a.length;y++) {
			for(int x = 0; x < a[0].length;x++) {
				a[y][x] = cluster.get(x)[y];
				b[x][y] = cluster.get(x)[y];
			}
		}
		
		double [][] covMat = multiply(a,b);
		
				
	
		for(int y= 0; y < covMat.length; y++){
			for(int x = 0; x < covMat.length; x++){
					covMat[y][x] /= cluster.size();
			}
		}

		return covMat;
	}

	/**
	 * Multiplica a matriz A pela B
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[][] multiply(double[][] a, double[][] b) {
		double [][] result = new double [a.length][b[0].length];
		for(int y = 0; y < result.length; y++){
			for(int x = 0; x < result[0].length; x++){
				for(int k = 0; k < a[0].length; k++){
					result[y][x] += a[y][k]*b[k][x];
				}
			}
		}
		return result;
	}

	public static double[][] inverseMatrix(double a[][]){
		int n = a.length;
		double x[][] = new double[n][n];
		double b[][] = new double[n][n];
		int index[] = new int[n];

		for (int i=0; i<n; ++i) 
			b[i][i] = 1;

		// Transform the matrix into an upper triangle
		gaussian(a, index);

		// Update the matrix b[i][j] with the ratios stored
		for (int i=0; i<n-1; ++i)
			for (int j=i+1; j<n; ++j)
				for (int k=0; k<n; ++k)
					b[index[j]][k]
							-= a[index[j]][i]*b[index[i]][k];

		// Perform backward substitutions
		for (int i=0; i<n; ++i){
			x[n-1][i] = b[index[n-1]][i]/a[index[n-1]][n-1];

			for (int j=n-2; j>=0; --j){
				x[j][i] = b[index[j]][i];

				for (int k=j+1; k<n; ++k){
					x[j][i] -= a[index[j]][k]*x[k][i];
				}
				x[j][i] /= a[index[j]][j];
			}
		}
		return x;
	}

	// Method to carry out the partial-pivoting Gaussian
	// elimination.  Here index[] stores pivoting order.
	private static void gaussian(double a[][], int index[]){
		int n = index.length;
		double c[] = new double[n];

		// Initialize the index
		for (int i=0; i<n; ++i) 
			index[i] = i;

		// Find the rescaling factors, one from each row
		for (int i=0; i<n; ++i){
			double c1 = 0;

			for (int j=0; j<n; ++j){
				double c0 = Math.abs(a[i][j]);
				if (c0 > c1) c1 = c0;
			}
			c[i] = c1;
		}

		// Search the pivoting element from each column
		int k = 0;

		for (int j=0; j<n-1; ++j){
			double pi1 = 0;

			for (int i=j; i<n; ++i){
				double pi0 = Math.abs(a[index[i]][j]);

				pi0 /= c[index[i]];

				if (pi0 > pi1){
					pi1 = pi0;
					k = i;
				}
			}

			// Interchange rows according to the pivoting order
			int itmp = index[j];
			index[j] = index[k];
			index[k] = itmp;

			for (int i=j+1; i<n; ++i){
				double pj = a[index[i]][j]/a[index[j]][j];

				// Record pivoting ratios below the diagonal
				a[index[i]][j] = pj;

				// Modify other elements accordingly
				for (int l=j+1; l<n; ++l)
					a[index[i]][l] -= pj*a[index[j]][l];
			}
		}
	}
}