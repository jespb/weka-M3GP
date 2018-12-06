import java.io.IOException;

import weka.classifiers.trees.m3gp.tree.Tree;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Matrix;

public class Teste {
	public static void main(String[] args) throws IOException {

		Tree t = new Tree(null, new double[] {0.2,0.2,0.2,0.2,0.2});
		double[] tmp = t.getGOA();
		t.incGOA(1);
		System.out.println(Arrays.arrayToString(tmp));
		System.out.println(Arrays.arrayToString(t.getGOA()));
	}

	private static double[][] abs(double[][] i2) {
		double [][] m = new double [i2.length][i2[0].length];
		for(int y = 0; y < m.length; y++) {
			for(int x  = 0; x < m[0].length; x++) {
				m[y][x] = i2[y][x] > 0 ? i2[y][x] : - i2[y][x];
			}
		}
		return m;
	}

	private static double[][] randomSimMat(int n) {
		int lim = 3;
		double [][] m = new double [n][n];
		for(int y = 0; y < m.length; y++) {
			for(int x = y; x < m[0].length; x++) {
				if(x!=y) {
				m[y][x] = (int)(Math.random()*lim);
				m[x][y] = m[y][x];}
				else {
					m[y][x] = (int)(Math.random()*(lim-1))+1;
				}
			}
		}
		return m;
	}

	private static void printMatrix(double[][]m) {
		for(int y = 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				System.out.print(m[y][x]+"\t");
			}
			System.out.println();
		}
	}
	
	public static double[][] inverseMatrix_new(double [][] m) {
		int n = m.length;

		double [][] inv = Matrix.identity(n);

		double d;

		for(int i = 0; i < n; i++) {
			if(m[i][i]==0) {
				for(int k = i+1; k < n; k++) {
					if(m[k][i] != 0) {
						double [] tmp = m[i];
						m[i] = m[k];
						m[k] = tmp;

						tmp = inv[i];
						inv[i] = inv[k];
						inv[k] = tmp;

						k=n;//stop
					}
				}
			}

			for(int y = i+1; y< n; y++) {
				d= -m[y][i] / m[i][i];

				for(int k = 0; k < n; k++) {
					inv[y][k] += inv[i][k]*d;
					m[y][k] += m[i][k]*d;
				}
			}
		}
		
		if(m[n-1][n-1] == 0) {
			return new double[][] {{Double.NaN}};
		}

		for(int i = 0; i < n; i++) {
			d = m[i][i];
			for(int k = 0; k < n; k++) {
				inv[i][k] /= d;
				m[i][k] /= d;
			}
		}

		for(int x = n-1; x > 0; x--) {
			for(int y = x-1; y >= 0; y--) {
				d = -m[y][x];
				for(int k = 0; k < n; k++)
					inv[y][k] += inv[x][k] * d;
				m[y][x]   += m[x][x]   * d;
			}
		}
		
		return inv;
	}
	
	public static double[][] inverseMatrix_new2(double [][] m) {
		int n = m.length;

		double [][] inv = Matrix.identity(n);

		double d;

		for(int i = 0; i < n; i++) {
			if(m[i][i]==0) {
				for(int k = i+1; k < n; k++) {
					if(m[k][i] != 0) {
						double [] tmp = m[i];
						m[i] = m[k];
						m[k] = tmp;

						tmp = inv[i];
						inv[i] = inv[k];
						inv[k] = tmp;

						k=n;//stop
					}
				}
			}

			for(int y = i+1; y< n; y++) {
				d= -m[y][i] / m[i][i];

				for(int k = 0; k < n; k++) {
					inv[y][k] += inv[i][k]*d;
					m[y][k] += m[i][k]*d;
				}
			}
		}
		
		if(m[n-1][n-1] == 0) {
			return new double[][] {{Double.NaN}};
		}

		for(int x = n-1; x > 0; x--) {
			for(int y = x-1; y >= 0; y--) {
				d = -m[y][x]/m[x][x];
				for(int k = 0; k < n; k++)
					inv[y][k] += inv[x][k] * d;
			}
		}
		
		for(int i = 0; i < n; i++) {
			d = m[i][i];
			for(int k = 0; k < n; k++) {
				inv[i][k] /= d;
			}
		}

		
		return inv;
	}

}