package weka.classifiers.trees.m3gp.util;

import java.util.ArrayList;

public class Matrix {
	public static double[][] multiply3(double[][] m1, double[][]m2, double[][]m3){
		double[][] ret= new double[m1.length][m3[0].length];
		int x,y,z;
		int ylen =ret.length, xlen=ret[0].length;
		for(y = 0; y < ylen; y++) {
			for(x = 0; x < xlen; x++) {
				
			}
		}
		return ret;
	}
	
	public static double[][] multiplyTransposeOfBy(double [][] d1, double [][] d2){
		double [][] ret = new double[d1[0].length][d2[0].length];
		int xlen = ret[0].length;
		int ylen = ret.length;
		int ilen = d2.length;
		int y,x,i;
		double acc;
		for(y = 0; y < ylen; y++) {
			for(x = 0; x < xlen; x++) {
				acc = 0;
				for(i = 0; i < ilen; i++) {
					acc += d1[i][y]*d2[i][x];
				}
				ret[y][x] = acc;
			}
		}
		return ret;
	}

	public static double[][] moorepenroseInverseMatrix(double [][] g) {
		int r, m = g.length-1, n = g[0].length-1;
		double [][] a, diagA, l;
		double tol;
		boolean transpose = false;

		// Transpose if m < n
		if (m < n){
			transpose = true;
			a = multiply(g,transpose(g));
			n = m;
		}else {
			a = multiply(transpose(g),g);
		}

		// Full rank Cholesky factorization of A
		diagA = diagonal(a);
		tol = min(diagA, 0)*Math.pow(10, -9);
		tol = Math.max(tol, Math.pow(10, -9));
		l = fill(a.length, 0);
		r=-1;
		for(int k = 0; k <= n; k++) {
			//System.out.println(r);
			r++;
			double [][] a1,l1,l2,lsect;
			a1 = section(a,new int[] {k,n,k,k});
			// Note: for r=0, the subtracted vector is zero
			if(r>0) {
				//System.out.println("k = " + k+ ", r = "+r);
				l1 = section(l,new int[] {k,n,0,r});
				l2 = transpose(section(l,new int[] {k,k,0,r}));
				lsect = subtract(a1,multiply(l1,l2));
			}else {
				lsect = a1;
			}
			//System.out.println(k + " "+ n);
			copyTo(lsect, l,new int[] {k,r});
			if(l[k][r]>tol) {
				l[k][r] = Math.sqrt(l[k][r]);
				if (k < n){
					for(int yi = k+1; yi <= n; yi++) {
						l[yi][r] /= l[k][r];
					}
				}
			}else {
				r--;
			}
		}
		if(r==-1) {
			return null;
		}
		l = section(l, new int[] {0,l.length-1,0,r});
		double [][] y =inverseMatrix(multiply(transpose(l),l));
		double [][] ret;
		if (transpose)
			ret = multiply(multiply(multiply(multiply(transpose(g),l),y),y),transpose(l));
		else
			ret = multiply(multiply(multiply(multiply(l,y),y),transpose(l)),transpose(g));

		return ret;
	}//TODO testar

	/**
	 * Copies the content from a matrix to another, starting at the coordinates coor
	 * @param from
	 * @param to matrix to be changed
	 * @param coor [y,x] coordinates
	 */
	private static void copyTo(double[][] from, double[][] to, int[] coor) {
		int fylen = from.length+coor[0], fxlen = from[0].length+coor[1];
		for(int y = coor[0]; y < fylen; y++) {
			for(int x = coor[1]; x < fxlen; x++) {
				to[y][x] = from[y-coor[0]][x-coor[1]];
			}
		}
	}

	/**
	 * Minimum value on the matrix bigger than m
	 * @param a
	 * @param i 
	 * @return
	 */
	private static double min(double[][] a, int m) {
		double min = -1;
		
		int ylen = a.length, xlen=a[0].length;
		for(int y = 0; y < ylen; y++) {
			for(int x = 0; x < xlen; x++) {
				if(a[y][x]>min && a[y][x]>m)
					min = a[y][x];
			}
		}
		return min;
	}

	/**
	 * Substracts b from a
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[][] subtract(double[][] a, double[][] b) {
		double [][] ret = new double[a.length][a[0].length];
		int ylen = a.length, xlen = a[0].length;
		for(int y = 0; y < ylen; y++) {
			for(int x = 0; x < xlen; x++) {
				ret[y][x] = a[y][x] - b[y][x];
			}
		}
		return ret;
	}

	/**
	 * Returns a section [xi:xf,yi:yf] of the matrix (inclusive for all vars)
	 * @param m
	 * @param ds [xi,xf,yi,yf]
	 * @return
	 */
	public static double[][] section(double[][] m, int[] ds) {
		double [][] ret = new double[(int) (Math.abs(ds[0]-ds[1])+1)][(int) (Math.abs(ds[2]-ds[3])+1)];
		boolean ydirection = ds[1]>=ds[0];
		boolean xdirection = ds[3]>=ds[2];
		//System.out.println(ds[0] + " " +ds[1] + " "+ds[2] + " "+ds[3] + " ");
		int ylen = ret.length, xlen = ret[0].length;
		for(int y = 0; y < ylen; y++) {
			for(int x = 0; x< xlen;x++) {
				ret[y][x] = m[ydirection? ds[0]+y : ds[0]-y][xdirection? ds[2]+x : ds[2]-x];
			}
		}
		return ret;
	}

	/**
	 * Returns the diagonal of the matrix on a row format
	 * @param a
	 * @return
	 */
	public static double[][] diagonal(double[][] a) {
		int ilen = a.length;
		double[][] diag = new double[1][ilen];
		for(int i = 0; i < ilen; i++) {
			diag[0][i] = a[i][i];
		}
		return diag;
	}

	/**
	 * Transpose matrix
	 * @param b
	 * @return
	 */
	public static double[][] transpose(double[][] b) {
		double[][] ret = new double[b[0].length][b.length];
		int ylen= b.length, xlen=b[0].length;
		for(int y = 0; y < ylen; y++)
			for(int x = 0; x < xlen; x++)
				ret[x][y] = b[y][x];
		return ret;
	}

	/**
	 * Identity matrix
	 * @param n
	 * @return
	 */
	public static double[][] identity(int n){
		double [][] m = new double[n][n];
		for(int i = 0; i < n;i++)
			m[i][i] = 1;
		return m;
	}

	/**
	 * Matrix filled with a value
	 * @param n
	 * @param value
	 * @return
	 */
	public static double[][] fill(int n, double value){
		double [][] m = new double[n][n];
		for(int y = 0; y < n;y++)
			for(int x = 0; x < n; x++)
				m[y][x] = value;
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

		int ylen = a.length, xlen=a[0].length;
		for(int y = 0; y < ylen ;y++) {
			for(int x = 0; x < xlen ;x++) {
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
		//System.out.println("mult: " + a.length + " " + a[0].length + " " + b.length + " " + b[0].length + " ");
		int ylen = result.length, xlen = result[0].length, klen = a[0].length;
		double acc;
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

	//TODO delete below ------------------
	public static void printMatrix(double[][]m) {
		for(int y = 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				System.out.print(m[y][x]+"\t");
			}
			System.out.println();
		}
	}

	public static double[][] clone(double[][] b) {
		double [][] ret = new double[b.length][b[0].length];
		for(int y = 0; y < b.length; y++) {
			for(int x = 0; x < b[0].length; x++) {
				ret[y][x]=b[y][x];
			}
		}
		return ret;
	}
}