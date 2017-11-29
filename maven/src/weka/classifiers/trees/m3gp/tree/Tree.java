package weka.classifiers.trees.m3gp.tree;

import java.util.ArrayList;

import weka.classifiers.trees.m3gp.node.Node;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Matrix;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Tree{
	public static int trainSize;
	
	private ArrayList<Node> dimensions;
	
	private ArrayList<double[][]> covarianceMatrix = null;
	private ArrayList<double[]> mu = null;
	private ArrayList<String> classes;

	/**
	 * Constructor
	 * @param name
	 * @param op
	 * @param term
	 * @param t_rate
	 * @param depth
	 */
	public Tree(String [] op, String [] term, double t_rate, int depth){
		dimensions = new ArrayList<Node>();
		dimensions.add(new Node(op, term, t_rate,depth));
	}

	public Tree(ArrayList<Node> dim) {
		dimensions = dim;
	}

	/**
	 * Returns the TreeSTGP under it's String format
	 */
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		sb.append("\"Dimension\":\""+dimensions.get(0)+"\"");
		for(int i = 1; i< dimensions.size(); i++) {
			sb.append(",\n\"Dimension\":\""+dimensions.get(i)+"\"");
		}
		sb.append("]");
		return sb.toString();
	}

	public int getSize() {
		int size = 0;
		for(int i = 0; i < dimensions.size(); i++) {
			size += dimensions.size();
		}
		return size;
	}

	private void makeCluster(double [][] data, String [] target, double trainFract) {
		classes = new ArrayList<String>();
		ArrayList<ArrayList<double []>> clusters = new ArrayList<ArrayList<double[]>>();
		
		//Descobre o numero de classes e cria um numero de clusters igual ao numero de classes
		for(int i = 0; i < (int)(target.length * trainFract); i++) {
			if(!classes.contains(target[i])) {
				classes.add(target[i]);
				clusters.add(new ArrayList<double[]>());
			}
		}
		
		//Adiciona os pontos ao cluster
		for(int i = 0, index = -1; i < (int)(data.length * trainFract);i++) {
			index = classes.indexOf(target[i]);
			double [] d = new double[dimensions.size()];
			for(int j = 0; j < dimensions.size(); j++) {
				d[j] = calculate(j,data[i]);
			}
			clusters.get(index).add(d);
		}
		
		covarianceMatrix = new ArrayList<double[][]>();
		for(int i = 0; i<clusters.size(); i++) {
			covarianceMatrix.add(Matrix.covarianceMatrix(clusters.get(i)));
		}
		/*
		for(int i = 0; i< covarianceMatrix.size();i++) {
			for(int y = 0; y < covarianceMatrix.get(i).length;y++) {
				for(int x = 0; x< covarianceMatrix.get(i)[y].length;x++) {
					System.out.print(covarianceMatrix.get(i)[0][0]+", ");		
				}
				System.out.println();
			}
			System.out.println();
		}*/
		
		// Calculo de mu
		mu = new ArrayList<double[]>();
		for(int i = 0; i < clusters.size();i++){
			mu.add(new double[clusters.get(i).get(0).length]);
			for(int j = 0; j < clusters.get(i).size(); j++){
				for(int k = 0; k < mu.get(i).length; k++) {
					mu.get(i)[k] += clusters.get(i).get(j)[k];
				}
			}
			for(int j = 0; j < mu.get(i).length; j++){
				mu.get(i)[j] /= clusters.get(i).size();
			}
		}
		/*
		System.out.println(this);
		for(int i = 0; i < mu.get(0).length; i++)
			System.out.print(mu.get(0)[i]+", ");
		System.out.println(); 
		*/
		//TODO verificar
	}
	
	private double calculate(int dimension, double [] d) {
		return dimensions.get(dimension).calculate(d);
	//	return 1/(1 - Math.pow(Math.E,-   dimensions.get(dimension).calculate(d)   ));
	}

	public String predict(double [] d) {
		//Calcula o valor em cada dimensao
		double [] result = new double [dimensions.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = calculate(i,d);
		}
		
		double [] distancias = new double[classes.size()];
		for(int i = 0; i < distancias.length; i++) {
			//System.out.print("\""+classes.get(i)+"\", ");
			distancias[i] = Arrays.mahalanobisDistance(result, 
					mu.get(i), covarianceMatrix.get(i));
			//System.out.print(distancias[i]+", ");
		}
		//System.out.println();
		
		double minDist = distancias[0];
		String prediction = classes.get(0);
		for(int i = 0; i < distancias.length; i++) {
			if(distancias[i] < minDist) {
				minDist = distancias[i];
				prediction = classes.get(i);
			}
		}
		//System.out.println(prediction);
		
		return prediction;
		//TODO fazer e verificar
	}
	
	public double getTrainAccuracy(double [][] data, String [] target, double trainFract){
		if (covarianceMatrix == null) 
			makeCluster(data, target, trainFract);
		double hits = 0;
		for(int i = 0; i < (int)(data.length*trainFract); i++) {
			if(predict(data[i]).equals(target[i]))
				hits++;
		}
		return hits/(int)(data.length*trainFract);
	}
	
	public double getTestAccuracy(double [][] data, String [] target, double trainFract){
		if (covarianceMatrix == null) 
			makeCluster(data, target, trainFract);
		double hits = 0;
		for(int i = (int)(data.length*trainFract); i < data.length; i++) {
			if(predict(data[i]).equals(target[i]))
				hits++;
		}
		return hits/(target.length - (int)(data.length*trainFract));
	}
	
	public ArrayList<Node> getDimensions() {
		return dimensions;
	}
	
	public int getDepth() {
		int depth = dimensions.get(0).getDepth();
		for(int i = 1; i < dimensions.size(); i++) {
			depth = Math.max(depth, dimensions.get(i).getDepth());
		}
		return depth;
	}
	
	public ArrayList<Node> cloneDimensions(){
		ArrayList<Node> ret = new ArrayList<Node>();
		for(int i = 0; i < dimensions.size(); i++) {
			ret.add(dimensions.get(i).clone());
		}
		return ret;
	}
}