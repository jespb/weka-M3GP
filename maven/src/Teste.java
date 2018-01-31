import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Matrix;

public class Teste {
	public static void main(String[] args) {
		//double [][] m = Matrix.fill(1, -1);

		//double [][] m1 = randomMat(3);		
		//	double [][] m2 = new double [][] {{0,1},{0,1}};
		//		printMatrix(myInverse(m2));


		//		printMatrix(multiply(m1,myInverse(m1)));

		System.out.println();

		long t1=0, t2=0, tx = 0;
double d;
		tx = System.currentTimeMillis();
		for(int i = 0; i < 10000; i++)
			for(int x = 0; x < 10000; x++)
			d= (Math.pow(i, 2) + Math.random());
		t1 += System.currentTimeMillis()-tx;

		tx = System.currentTimeMillis();
		for(int i = 0; i < 10000; i++)
			for(int x = 0; x < 10000; x++)
			d= (i*i + Math.random());
		t2 += System.currentTimeMillis()-tx;
		System.out.println();
		System.out.println(t1 + " " + t2);




		//System.out.println("Funcao nova: " + (System.currentTimeMillis()-time));
		/*
		time = System.currentTimeMillis();
		Matrix.multiply(Matrix.transpose(m1),m2);
		System.out.println("Funcao velha: " + (System.currentTimeMillis()-time));*/
		//double [][] pimpm1 = Matrix.moorepenroseInverseMatrix(m1);

		//printMatrix(m1);System.out.println();
		//printMatrix(pimpm1);System.out.println();
		//printMatrix(Matrix.subtract(Matrix.multiply(Matrix.multiply(m1,pimpm1),m1),m1));
	}

	private static void printMatrix(double[][]m) {
		for(int y = 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				System.out.print(m[y][x]+"\t");
			}
			System.out.println();
		}
	}

	public static double[][] myInverse(double [][] m) {
		int n = m.length;

		m = Matrix.clone(m);
		double [][] inv = Matrix.identity(n);

		double d;

		for(int i = 0; i < n; i++) {
			for(int y = i+1; y< n; y++) {
				d= -m[y][i] / m[i][i];
				for(int k = 0; k < n; k++) {
					inv[y][k] += inv[i][k]*d;
					m[y][k] += m[i][k]*d;
				}
			}
		}

		for(int i = 0; i < n; i++) {
			inv[i] = Arrays.multiply(inv[i], 1/m[i][i]);
			m[i] = Arrays.multiply(m[i], 1/m[i][i]);
		}

		for(int y = 0; y < n; y++) {
			for(int x = y+1; x< n; x++) {
				d = -m[y][x];
				for(int k = y>0?y-1:0; k < n; k++) {
					inv[y][k] += inv[x][k]*d;
					m[y][k] += m[x][k]*d;
				}

			}
		}
		return inv;
	}

	public static double determinant(double [][] m) {
		double det = 1;
		int n = m.length;
		m = Matrix.clone(m);
		for(int y = 1; y < n; y++)
			for(int y2 = y; y2< n; y2++) 
				m[y2] = Arrays.sum(m[y2], m[y-1], -m[y2][y-1]/m[y-1][y-1]);


		for(int i = 0; i < n; i++) 
			det *= m[i][i];

		return det;
	}

	private static double[][] randomMat(int n){
		double [][] m = new double [n][n];
		for(int y = 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				m[y][x] = (int)(Math.random()*100);
			}
		}
		return m;
	}



	public static double[][] multiply(double[][] a, double[][] b) {
		double [][] result = new double [a.length][b[0].length];
		int ylen = result.length, xlen = result[0].length, klen = a[0].length;
		double acc = 0;
		for(int y = 0; y < ylen; y++){
			for(int x = 0; x < xlen; x++){
				acc = 0;
				for(int k = 0; k < klen; k++){
					acc += a[y][k]*b[k][x];
				}
				result[y][x] = acc;
			}
		}
		return result;
	}

	public static double[][] multiply2(double[][] a, double[][] b) {
		double [][] result = new double [a.length][b[0].length];
		int ylen = result.length, xlen = result[0].length, klen = a[0].length;
		double acc = 0;
		for(int y = 0; y < ylen; y++)
			for(int x = 0; x < xlen; x++){
				acc = 0;
				for(int k = 0; k < klen; k++)
					acc += a[y][k]*b[k][x];			
				result[y][x] = acc;
			}
		return result;
	}

	public static double[][] multiplyBy(double [][]m, double a) {
		double [][] ret = new double[m.length][m[0].length];
		for(int y= 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				ret[y][x] = m[y][x]*a;
			}
		}
		return ret;
	}

	public static double determinanteEscada(double [][] m) {
		double det = 1;
		m = Matrix.clone(m);
		for(int y = 1; y < m.length; y++) {
			//det *= (Math.pow(1/m[x][x], m.length));
			//m = multiplyBy(m, 1/m[x][x] );
			for(int y2 = y; y2< m.length; y2++) {
				m[y2] = sum(m[y2], m[y-1], -m[y2][y-1]/m[y-1][y-1]);
			}
		}

		for(int i = 0; i < m.length; i++) {
			det *= m[i][i];
		}
		//printMatrix(m);
		return det;
	}

	private static double[] sum(double[] ds, double[] ds2, double d) {
		double[] ret = new double[ds.length];
		for(int i = 0; i < ret.length; i++) {
			ret[i] = ds[i]+ds2[i]*d;
		}
		return ret;
	}

	private static double[][] multiply(double[][] m, double pow) {
		// TODO Auto-generated method stub
		return null;
	}
}