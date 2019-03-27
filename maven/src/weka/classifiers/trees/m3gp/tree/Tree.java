package weka.classifiers.trees.m3gp.tree;

import java.io.Serializable;
import java.util.ArrayList;

import weka.classifiers.trees.m3gp.client.Constants;
import weka.classifiers.trees.m3gp.node.Node;
import weka.classifiers.trees.m3gp.population.Population;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Matrix;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
 *
 */
public class Tree implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private double[] goAffinity; // Probabilidades de cada genetic operator
	
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
	public Tree(String [] term, double t_rate, int depth){
		dimensions = new ArrayList<Node>();
		dimensions.add(new Node(term,depth));
		
		goAffinity = new double [Constants.NUMBER_OF_GENETIC_OPERATORS];
		for (int i = 0; i < goAffinity.length; i++){
			goAffinity[i]=1.0/goAffinity.length;
		}
	}
	

	@SuppressWarnings("unused")
	public double[] getGOA(){
		if(Constants.PROBABILITY_ADAPTATION > 0)
			return Arrays.copy(goAffinity);
		else
			return Population.goAffinity;
	}

	public void setGOA(double [] goa) {
		goAffinity = goa;
	}

	public void incGOA(int operation) {
		//goAffinity[operation] += 0.1;
		goAffinity[operation] = 1 - ( (1 - goAffinity[operation]) * Constants.LEARNING_T );
		fixGOA();
	}

	public void decGOA(int operation) {
		//goAffinity[operation] *= 0.95;
		goAffinity[operation] *= Constants.LEARNING_T ;
		fixGOA();
	}
	
	private void fixGOA() {
		goAffinity = Arrays.normalize(goAffinity);
		/*
		for(int ii = 0; ii< goAffinity.length; ii++) {
			goAffinity[ii] -=0.05;
			if(goAffinity[ii] < 0) {
				goAffinity[ii] = 0;
			}
		}
		setGOA(Arrays.normalize(goAffinity));
		for(int ii = 0; ii< goAffinity.length; ii++) {
			goAffinity[ii] *= 1-0.05*goAffinity.length;
			goAffinity[ii] += 0.05;
		}
		*/
	}

	public Tree(ArrayList<Node> dim) {
		dimensions = dim;
	}
	

	public Tree(ArrayList<Node> dim, double [] goa) {
		dimensions = dim;

		if(goa != null){
			goAffinity = new double[goa.length];
			for(int i = 0; i < goa.length; i++){
				goAffinity[i] = goa[i];
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
			sb.append("                \""+dimensions.get(i).toString()+"\",\n");
		}
		sb.append("                \""+dimensions.get(dimensions.size()-1).toString()+"\"\n");
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

	private void makeCluster(double [][] data, String [] target) {
		this.target = target;
		this.map = new double[(int)(data.length * Constants.TRAIN_FRACTION)][dimensions.size()];
		
		classes = new ArrayList<String>();
		ArrayList<ArrayList<double []>> clusters = new ArrayList<ArrayList<double[]>>();

		//Descobre o numero de classes e cria um numero de clusters igual ao numero de classes
		for(int i = 0; i < (int)(target.length * Constants.TRAIN_FRACTION); i++) {
			if(!classes.contains(target[i])) {
				classes.add(target[i]);
				clusters.add(new ArrayList<double[]>());
			}
		}
		
		//Adiciona os pontos ao cluster
		for(int i = 0, index = -1; i < (int)(data.length * Constants.TRAIN_FRACTION);i++) {
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
	

	public String toJSON(double[][] data, String[] target) {
		StringBuilder sb = new StringBuilder();
		sb.append(toString()+",\n");

		//goAffinity
		sb.append("            \"GOA\":"+ Arrays.arrayToString(Arrays.normalize(getGOA())) + ",\n");

		
		//pontos treino
		sb.append("            \"Train\":[\n");
		for(int i = 0; i < (int)(data.length*Constants.TRAIN_FRACTION); i++) {
			sb.append("                [");
			for(int dim = 0; dim < dimensions.size(); dim++) {
				sb.append( "\"" + dimensions.get(dim).calculate(data[i]) +"\"," );
			}
			sb.append( "\"" + target[i]+"\"]");
			if (i < data.length*Constants.TRAIN_FRACTION-1)
				sb.append(",");
			sb.append("\n");
		}
		sb.append("            ],\n");

		//pontos teste
		sb.append("            \"Test\":[\n");
		for(int i = (int)(data.length*Constants.TRAIN_FRACTION); i < data.length; i++) {
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
	public double getTrainAccuracy(double [][] data, String [] target){
		if (covarianceMatrix == null) {
			makeCluster(data, target);
		}

		double hits = 0;
		for(int i = 0; i < (int)(data.length*Constants.TRAIN_FRACTION); i++) {
			if(predict(data[i]).equals(target[i]))
				hits++;
		}
		return hits/(int)(data.length*Constants.TRAIN_FRACTION);
	}

	public double getTestAccuracy(double [][] data, String [] target){
		if (covarianceMatrix == null) {
			makeCluster(data, target);
		}
		double hits = 0;
		for(int i = (int)(data.length*Constants.TRAIN_FRACTION); i < data.length; i++) {
			if(predict(data[i]).equals(target[i]))
				hits++;
		}
		return hits/(target.length - (int)(data.length*Constants.TRAIN_FRACTION));
	}

	

	public double getTrainRootMeanSquaredDistanceToCentroid(double [][] data, String[] target) {
		if (covarianceMatrix == null) {
			makeCluster(data, target);
		}
				
		double acc_distance = 0;
		double set_size = (int)(data.length*Constants.TRAIN_FRACTION);
		for(int i = 0; i < set_size; i++) {
			double [] coor = new double[dimensions.size()];
			for(int d = 0;d < dimensions.size(); d++) {
				coor[d] = dimensions.get(d).calculate(data[i]);
			}
			acc_distance += Math.pow(Arrays.euclideanDistance(coor, mu.get(classes.indexOf(target[i]) )),2 );
		}
		return Math.sqrt(acc_distance/set_size);
	}
	
	public double getTestRootMeanSquaredDistanceToCentroid(double [][] data, String[] target) {
		if (covarianceMatrix == null) {
			makeCluster(data, target);
		}
		
		double acc_distance = 0;
		double set_size = data.length - (int)(data.length*Constants.TRAIN_FRACTION);
		for(int i = (int)(data.length*Constants.TRAIN_FRACTION); i < data.length; i++) {
			double [] coor = new double[dimensions.size()];
			for(int d = 0;d < dimensions.size(); d++) {
				coor[d] = dimensions.get(d).calculate(data[i]);
			}
			acc_distance += Math.pow(Arrays.euclideanDistance(coor, mu.get(classes.indexOf(target[i]) )),2 );
			}
		return Math.sqrt(acc_distance/set_size);
	}

	public double getMeanDistanceBetweenCentroids(double [][] data, String[] target) {
		if (covarianceMatrix == null) {
			makeCluster(data, target);
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

	public double getTrainRootMeanSquaredMHLNBDistanceToCentroid(double[][] data, String[] target) {
		if (covarianceMatrix == null) {
			makeCluster(data, target);
		}
				
		double acc_distance = 0;
		double set_size = (int)(data.length*Constants.TRAIN_FRACTION);
		for(int i = 0; i < set_size; i++) {
			double [] result = calculateAll(data[i]);
			int index = classes.indexOf(target[i]);
			double distance = Arrays.mahalanobisDistance(result,mu.get(index), covarianceMatrix.get(index));
			acc_distance += Math.pow(distance,2);
		}
		return Math.sqrt(acc_distance/set_size);
	}
	
	public double getTestRootMeanSquaredMHLNBDistanceToCentroid(double[][] data, String[] target) {
		if (covarianceMatrix == null) {
			makeCluster(data, target);
		}
				
		double acc_distance = 0;
		int set_size = (int)(data.length*Constants.TRAIN_FRACTION);
		for(int i = set_size; i < data.length; i++) {
			double [] result = calculateAll(data[i]);
			double [] distances = calculateMHLNB(result);
			acc_distance += Math.pow(distances[classes.indexOf(target[i])],2);
		}
		return Math.sqrt(acc_distance/set_size);
	}

	public double getMeanManhattanDistanceBetweenCentroids(double[][] data, String[] target) {
		if (covarianceMatrix == null) {
			makeCluster(data, target);
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

	public double getMeanManhattanDistanceToCentroids(double[][] data, String[] target) {
		if (covarianceMatrix == null) {
			makeCluster(data, target);
		}
				
		double acc_distance = 0;
		double set_size = (int)(data.length*Constants.TRAIN_FRACTION);
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