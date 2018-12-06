package weka.classifiers.trees.m3gp.population;

import weka.classifiers.trees.m3gp.tree.Tree;
import weka.classifiers.trees.m3gp.tree.TreePruningHandler;
import weka.classifiers.trees.m3gp.util.Mat;

public class PopulationFunctions {
	/*
	 * -1 : Accuracy
	 * 1 : Mean distance to centroid
	 * -2 : accuracy - mean distance to centroid as a very small value
	 * -3 : accuracy - sigmoid(#dimensions)/trainset_size
	 * -4 : sigmoid(rms dist between clusters) - sigmoin(mean distance of points to the centroids)
	 * -5 : accuracy - sigmoid(#nodes)/trainset_size
	 * -6 : sigmoid(rms mhlnb dist between clusters) - sigmoin(mean distance of points to the centroids)
	 */
	static int fitnessType = -5;
	public static double fitnessTrain(Tree t, double [][] data, String [] target, double trainFraction) {
		double d = 0,acc,dist_ce,d_size, dist_cl;
		switch (fitnessType){		
		case -4:
			dist_cl = Mat.sigmod(t.getMeanDistanceBetweenCentroids(data, target, trainFraction)/t.getDimensions().size());
			dist_ce = Mat.sigmod(t.getTrainRootMeanSquaredDistanceToCentroid(data, target, trainFraction)/t.getDimensions().size()); 
			d = dist_cl-dist_ce;
			break;
		case -5:
			acc = t.getTrainAccuracy(data, target, trainFraction); 
			d_size = 1.0*t.getSize();
			d_size = Mat.sigmod(Math.sqrt(d_size/1000.0));
			d = acc - d_size/(data.length*trainFraction);
			break;
		case -6:
			dist_cl = Mat.sigmod(t.getMeanDistanceBetweenCentroids(data, target, trainFraction)/Math.sqrt(t.getDimensions().size()));
			dist_ce = Mat.sigmod(t.getTrainRootMeanSquaredMHLNBDistanceToCentroid(data, target, trainFraction)/Math.sqrt(t.getDimensions().size())); 
			d = dist_cl-dist_ce;
			break;
		case -7:
			dist_cl = t.getMeanDistanceBetweenCentroids(data, target, trainFraction)/Math.sqrt(t.getDimensions().size());
			dist_ce = t.getTrainRootMeanSquaredDistanceToCentroid(data, target, trainFraction)/Math.sqrt(t.getDimensions().size()); 
			d = dist_cl-dist_ce;
			break;
		}
		return d;		
	}

	/**
	 * Picks as many trees from population as the size of the tournament
	 * and return the one with the lower fitness, assuming the population
	 * is already sorted
	 * @param population Tree population
	 * @return The winner tree
	 */
	static boolean smallerIsBetter = fitnessType > 0;
	public static Tree tournament(Tree [] population, int tournamentSize) {
		int pick = Mat.random(tournamentSize);
		for(int i = 1; i < tournamentSize; i ++){
			if(smallerIsBetter)
				pick = Math.min(pick, Mat.random(population.length));
			else
				pick = Math.max(pick, Mat.random(population.length));
		}
		return population[pick];
	}


	public static Tree prun(Tree tree, double[][] data, String[] target, double trainFraction) {
		double [] goa = tree.getGOA();
		
		Tree t = TreePruningHandler.prun(tree, data, target, trainFraction);
		t = TreePruningHandler.prun(t, data, target, trainFraction);
		
		t.setGOA(goa);
		return t;
	}

	public static boolean betterTrain(Tree t1, Tree t2, double[][] data, String[] target, double trainFract) {
		double t1_fit = fitnessTrain(t1,data,target,trainFract);
		double t2_fit = fitnessTrain(t2,data,target,trainFract);
		return smallerIsBetter? t1_fit < t2_fit : t1_fit > t2_fit; 
	}
	
	public static boolean betterOrEqualTrain(Tree t1, Tree t2, double[][] data, String[] target, double trainFract) {
		double t1_fit = fitnessTrain(t1,data,target,trainFract);
		double t2_fit = fitnessTrain(t2,data,target,trainFract);
		return smallerIsBetter? t1_fit <= t2_fit : t1_fit >= t2_fit; 
	}
}