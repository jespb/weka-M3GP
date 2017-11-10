import weka.classifiers.trees.m3gp.util.Matrix;

public class Teste {
	public static void main(String[] args) {
		//double [][] m = Matrix.fill(1, -1);
		
		double [][] m1 = randomMat(6);

		double [][] pimpm1 = Matrix.moorepenroseInverseMatrix(m1);

		//printMatrix(m1);System.out.println();
		//printMatrix(pimpm1);System.out.println();
		printMatrix(Matrix.subtract(Matrix.multiply(Matrix.multiply(m1,pimpm1),m1),m1));
	}
	
	private static void printMatrix(double[][]m) {
		for(int y = 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				System.out.print(m[y][x]+"\t");
			}
			System.out.println();
		}
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
}
