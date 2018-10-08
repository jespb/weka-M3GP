package weka.classifiers.trees.m3gp.tree;

import java.io.Serializable;
import java.util.ArrayList;

import weka.classifiers.trees.m3gp.node.Node;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Matrix;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Tree implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public static String[] operations;
	public static int trainSize;

	private double[] goAffinity; // Probabilidades de cada genetic operator
	private int[] goAffinityCount; // Nr de vezes que cada GO foi usado
	
	private ArrayList<Node> dimensions;

	private ArrayList<double[][]> covarianceMatrix = null;
	private ArrayList<double[]> mu = null;
	ArrayList<String> classes;
	
	private String[] target;
	private double[][] map;

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
		operations = op;
		
		goAffinity = new double [TreeGeneticOperatorHandler.numberOfGeneticOperators];
		goAffinityCount = new int [TreeGeneticOperatorHandler.numberOfGeneticOperators];
		for (int i = 0; i < goAffinity.length; i++){
			goAffinity[i]=1;
		}
	}
	

	public double[] getGOA(){
		return goAffinity;
	}

	public int[] getGOAC(){
		return goAffinityCount;
	}
	
	public void setGOA(double [] goa) {
		goAffinity = goa;
	}
	
	public void setGOAC(int [] goac) {
		goAffinityCount = goac;
	}

	public void incGOA(int operation) {
		goAffinity[operation] += 1.10;
		goAffinityCount[operation] ++;
	}

	public void decGOA(int operation) {
		goAffinity[operation] /= 1.05;
		if(goAffinity[operation]<0.1)	
			goAffinity[operation] = 0.1;
		goAffinityCount[operation] ++;
	}

	public Tree(ArrayList<Node> dim) {
		dimensions = dim;
	}
	

	public Tree(ArrayList<Node> dim, double [] goa, int [] goac) {
		dimensions = dim;

		if(goa != null){
			goAffinity = new double[goa.length];
			goAffinityCount = new int[goa.length];
			for(int i = 0; i < goa.length; i++){
				goAffinity[i] = goa[i];
				goAffinityCount[i] = goac[i];
			}
		}
	}

	/**
	 * Returns the TreeSTGP under it's String format
	 */
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("            \"Dimensions\":[\n");
		for(int i = 0; i< dimensions.size()-1; i++) {
			sb.append("                \""+dimensions.get(i).toString(operations)+"\",\n");
		}
		sb.append("                \""+dimensions.get(dimensions.size()-1).toString(operations)+"\"\n");
		sb.append("            ]");
		return sb.toString();
	}

	public int getSize() {
		int size = 0;
		for(int i = 0; i < dimensions.size(); i++) {
			size += dimensions.get(i).getSize();
		}
		return size;
	}
	
	public double[][] getMap(){
		return map;
	}
	
	public String[] getTarget() {
		return target;
	}

	private void makeCluster(double [][] data, String [] target, double trainFract) {
		this.target = target;
		this.map = new double[(int)(data.length * trainFract)][dimensions.size()];
		
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
			map[i] = d;
			clusters.get(index).add(d);
		}

		covarianceMatrix = new ArrayList<double[][]>();
		for(int i = 0; i<clusters.size(); i++) {
			covarianceMatrix.add(Matrix.covarianceMatrix(clusters.get(i)));
		}

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
	}

	private double calculate(int dimension, double [] d) {
		return dimensions.get(dimension).calculate(d);
	}
	
	double[] calculateAll(double [] d) {
		double [] result = new double [dimensions.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = calculate(i,d);
		}
		return result;
	}
	
	double[] calculateMHLNB(double [] result) {
		double [] distancias = new double[classes.size()];
		for(int i = 0; i < distancias.length; i++) {
			distancias[i] = Arrays.mahalanobisDistance(result, 
					mu.get(i), covarianceMatrix.get(i));
		}
		return distancias;
	}

	static long t_ax, t_c = 0, t_d = 0, t_i=0;
	public String predict(double [] d) {
		return Classification.predict(this, d);
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

	public String toJSON(double[][] data, String[] target, double trainFraction) {
		StringBuilder sb = new StringBuilder();
		sb.append(toString()+",\n");

		//goAffinity
		sb.append("            \"GOA\":"+ Arrays.arrayToString(goAffinity) + ",\n");
		sb.append("            \"GOAC\":"+ Arrays.arrayToString(goAffinityCount) + ",\n");

		
		//pontos treino
		sb.append("            \"Train\":[\n");
		for(int i = 0; i < (int)(data.length*trainFraction); i++) {
			sb.append("                [");
			for(int dim = 0; dim < dimensions.size(); dim++) {
				sb.append( "\"" + dimensions.get(dim).calculate(data[i]) +"\"," );
			}
			sb.append( "\"" + target[i]+"\"]");
			if (i < data.length*trainFraction-1)
				sb.append(",");
			sb.append("\n");
		}
		sb.append("            ],\n");

		//pontos teste
		sb.append("            \"Test\":[\n");
		for(int i = (int)(data.length*trainFraction); i < data.length; i++) {
			sb.append("                [");
			for(int dim = 0; dim < dimensions.size(); dim++) {
				sb.append( "\"" + dimensions.get(dim).calculate(data[i]) +"\"," );
			}
			sb.append( "\"" +  target[i]+"\"]");
			if (i < data.length-1)
				sb.append(",");
			sb.append("\n");
		}
		sb.append("            ]");

		return sb.toString();
	}
	
	// ------- ------- ------- FUNCOES DE FITNESS ------- ------- -------
	public double getTrainAccuracy(double [][] data, String [] target, double trainFract){
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFract);
		}

		double hits = 0;
		for(int i = 0; i < (int)(data.length*trainFract); i++) {
			if(predict(data[i]).equals(target[i]))
				hits++;
		}
		return hits/(int)(data.length*trainFract);
	}

	public double getTestAccuracy(double [][] data, String [] target, double trainFract){
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFract);
		}
		double hits = 0;
		for(int i = (int)(data.length*trainFract); i < data.length; i++) {
			if(predict(data[i]).equals(target[i]))
				hits++;
		}
		return hits/(target.length - (int)(data.length*trainFract));
	}

	

	public double getTrainRootMeanSquaredDistanceToCentroid(double [][] data, String[] target, double trainFract) {
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFract);
		}
				
		double acc_distance = 0;
		double set_size = (int)(data.length*trainFract);
		for(int i = 0; i < set_size; i++) {
			double [] coor = new double[dimensions.size()];
			for(int d = 0;d < dimensions.size(); d++) {
				coor[d] = dimensions.get(d).calculate(data[i]);
			}
			acc_distance += Math.pow(Arrays.euclideanDistance(coor, mu.get(classes.indexOf(target[i]) )),2 );
		}
		return Math.sqrt(acc_distance/set_size);
	}
	
	public double getTestRootMeanSquaredDistanceToCentroid(double [][] data, String[] target, double trainFract) {
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFract);
		}
		
		double acc_distance = 0;
		double set_size = data.length - (int)(data.length*trainFract);
		for(int i = (int)(data.length*trainFract); i < data.length; i++) {
			double [] coor = new double[dimensions.size()];
			for(int d = 0;d < dimensions.size(); d++) {
				coor[d] = dimensions.get(d).calculate(data[i]);
			}
			acc_distance += Math.pow(Arrays.euclideanDistance(coor, mu.get(classes.indexOf(target[i]) )),2 );
			}
		return Math.sqrt(acc_distance/set_size);
	}

	public double getMeanDistanceBetweenCentroids(double [][] data, String[] target, double trainFract) {
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFract);
		}
		
		double total_distance = 0;
		for(int i = 0; i < mu.size(); i++) {
			for(int j = i+1; j < mu.size(); j++) {
				total_distance += Arrays.euclideanDistance(mu.get(i), mu.get(j));
			}
		}
		total_distance /= mu.size()*(mu.size()-1)/2;
		// TODO Auto-generated method stub
		return total_distance;
	}

	public void clean() {
		for(int i = 0; i < dimensions.size(); i++) {
			dimensions.get(i).clean();
		}
	}

	public double getTrainRootMeanSquaredMHLNBDistanceToCentroid(double[][] data, String[] target, double trainFraction) {
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFraction);
		}
				
		double acc_distance = 0;
		double set_size = (int)(data.length*trainFraction);
		for(int i = 0; i < set_size; i++) {
			double [] result = calculateAll(data[i]);
			int index = classes.indexOf(target[i]);
			double distance = Arrays.mahalanobisDistance(result,mu.get(index), covarianceMatrix.get(index));
			acc_distance += Math.pow(distance,2);
		}
		return Math.sqrt(acc_distance/set_size);
	}
	
	public double getTestRootMeanSquaredMHLNBDistanceToCentroid(double[][] data, String[] target, double trainFraction) {
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFraction);
		}
				
		double acc_distance = 0;
		int set_size = (int)(data.length*trainFraction);
		for(int i = set_size; i < data.length; i++) {
			double [] result = calculateAll(data[i]);
			double [] distances = calculateMHLNB(result);
			acc_distance += Math.pow(distances[classes.indexOf(target[i])],2);
		}
		return Math.sqrt(acc_distance/set_size);
	}

	public double getMeanManhattanDistanceBetweenCentroids(double[][] data, String[] target, double trainFraction) {
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFraction);
		}
		
		double total_distance = 0;
		for(int i = 0; i < mu.size(); i++) {
			for(int j = i+1; j < mu.size(); j++) {
				total_distance += Arrays.manhattanDistance(mu.get(i), mu.get(j));
			}
		}
		total_distance /= mu.size()*(mu.size()-1)/2;
		// TODO Auto-generated method stub
		return total_distance;
	}

	public double getMeanManhattanDistanceToCentroids(double[][] data, String[] target, double trainFraction) {
		if (covarianceMatrix == null) {
			makeCluster(data, target, trainFraction);
		}
				
		double acc_distance = 0;
		double set_size = (int)(data.length*trainFraction);
		for(int i = 0; i < set_size; i++) {
			double [] result = calculateAll(data[i]);
			int index = classes.indexOf(target[i]);
			double distance = Arrays.manhattanDistance(result,mu.get(index));
			acc_distance += distance;
		}
		return Math.sqrt(acc_distance/set_size);
	}

	public double[] calculateEucDistances(double[] result) {
		double [] dist =  new double[mu.size()];
		for (int i = 0; i < dist.length; i++) {
			dist[i] = Arrays.euclideanDistance(result, mu.get(i));
		}
		return dist;
	}
	
}