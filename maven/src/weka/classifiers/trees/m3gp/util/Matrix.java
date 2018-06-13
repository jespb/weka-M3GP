package weka.classifiers.trees.m3gp.util;

import java.util.ArrayList;

public class Matrix {	
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
		double min = a[0][0];

		int ylen = a.length, xlen=a[0].length;
		for(int y = 0; y < ylen; y++) {
			for(int x = 0; x < xlen; x++) {
				if(a[y][x]<min && a[y][x]>m)
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

		// Alt 1
		/*
		boolean ydirection = ds[1]>=ds[0];
		boolean xdirection = ds[3]>=ds[2];
		//System.out.println(ds[0] + " " +ds[1] + " "+ds[2] + " "+ds[3] + " ");
		int ylen = ret.length, xlen = ret[0].length;
		for(int y = 0; y < ylen; y++) {
			for(int x = 0; x< xlen;x++) {
				ret[y][x] = m[ydirection? ds[0]+y : ds[0]-y][xdirection? ds[2]+x : ds[2]-x];
			}
		}
		 */

		// Alt 2
		int direction = (ds[1]>=ds[0]?1:0) + (ds[3]>=ds[2]?2:0); 
		int ylen = ret.length, xlen = ret[0].length;
		switch (direction) {
		case 0:
			for(int y = 0; y < ylen; y++) {
				for(int x = 0; x< xlen;x++) {
					ret[y][x] = m[ds[0]-y][ds[2]-x];
				}
			}
			break;
		case 1:
			for(int y = 0; y < ylen; y++) {
				for(int x = 0; x< xlen;x++) {
					ret[y][x] = m[ds[0]+y][ds[2]-x];
				}
			}
			break;
		case 2:
			for(int y = 0; y < ylen; y++) {
				for(int x = 0; x< xlen;x++) {
					ret[y][x] = m[ds[0]-y][ds[2]+x];
				}
			}
			break;
		case 3:
			for(int y = 0; y < ylen; y++) {
				for(int x = 0; x< xlen;x++) {
					ret[y][x] = m[ds[0]+y][ds[2]+x];
				}
			}
			break;
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
		for(int y = 0; y < n;y++)
			m[y][y] = 1;
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

		for(int y= 0; y < ylen; y++){
			for(int x = 0; x < ylen; x++){
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

	public static double[][] inverseMatrix(double  [][] m){
		m = clone(m);
		//return inverseMatrix_new(m);
		return inverseMatrix_old(m);
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
			return fill(m.length, Double.NaN);
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

	public static double[][] inverseMatrix_old(double a[][]){
		int n = a.length;
		double x[][] = new double[n][n];
		double b[][] = identity(n);
		int index[] = new int[n];

		// Transform the matrix into an upper triangle
		gaussian(a, index);

		// Update the matrix b[i][j] with the ratios stored
		for (int i=0; i<n-1; i++)
			for (int j=i+1; j<n; j++)
				for (int k=0; k<n; k++)
					b[index[j]][k]
							-= a[index[j]][i]*b[index[i]][k];

		// Perform backward substitutions
		for (int i=0; i<n; i++){
			x[n-1][i] = b[index[n-1]][i]/a[index[n-1]][n-1];

			for (int j=n-2; j>=0; j--){
				x[j][i] = b[index[j]][i];

				for (int k=j+1; k<n; k++)
					x[j][i] -= a[index[j]][k]*x[k][i];

				x[j][i] /= a[index[j]][j];
			}
		}
		return x;
	}

	// Method to carry out the partial-pivoting Gaussian
	// elimination.  Here index[] stores pivoting order.
	public static void gaussian(double a[][], int index[]){
		int n = index.length;
		double c[] = new double[n];

		// Initialize the index
		for (int i=0; i<n; i++) 
			index[i] = i;

		// Find the rescaling factors, one from each row
		for (int i=0; i<n; i++){
			double c1 = 0;

			for (int j=0; j<n; j++)
				c1 = Math.max(Math.abs(a[i][j]),c1);

			c[i] = c1;
		}

		// Search the pivoting element from each column
		int k = 0;

		for (int j=0; j<n-1; j++){
			double pi1 = 0;

			for (int i=j; i<n; i++){
				double pi0 = Math.abs(a[index[i]][j]) / c[index[i]];

				if (pi0 > pi1){
					pi1 = pi0;
					k = i;
				}
			}

			// Interchange rows according to the pivoting order
			int itmp = index[j];
			index[j] = index[k];
			index[k] = itmp;

			for (int i=j+1; i<n; i++){
				double pj = a[index[i]][j]/a[index[j]][j];

				// Record pivoting ratios below the diagonal
				a[index[i]][j] = pj;

				// Modify other elements accordingly
				for (int l=j+1; l<n; l++)
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
		for(int y = 0; y < b.length; y++) 
			for(int x = 0; x < b[0].length; x++) 
				ret[y][x]=b[y][x];


		return ret;
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

	public static void printMatrix(int[][] m) {
		StringBuilder sb = new StringBuilder();
		for(int y = 0; y < m.length; y++) {
			for(int x = 0; x < m[0].length; x++) {
				sb.append(m[y][x]+"," );
				for(int i = (m[y][x]+"").length(); i < 3; i++) {
					sb.append(" ");
				}
			}
			sb.append("\n");
		}
		System.out.println(sb);
	}
}