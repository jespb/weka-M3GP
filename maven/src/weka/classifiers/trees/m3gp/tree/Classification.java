package weka.classifiers.trees.m3gp.tree;

import weka.classifiers.trees.m3gp.util.Arrays;

public class Classification {
	private static int method = 3;
	
	public static String predict(Tree t, double[] d, int method) {
		String p = null;
		switch(method) {
		case 1:
			p = mahalanobisDistance(t, d);
			break;
		case 2:
			p = euclideanDistance(t,d);
			break;
		case 3:
			p = knn(t,d);
			break;
		}
		return p;
	}
	
	public static String predict(Tree t, double[] d) {
		String p = null;
		switch(method) {
		case 1:
			p = mahalanobisDistance(t, d);
			break;
		case 2:
			p = euclideanDistance(t,d);
			break;
		case 3:
			p = knn(t,d);
			break;
		}
		return p;
	}
	
	public static String mahalanobisDistance(Tree t, double[] d) {
		double [] result = t.calculateAll(d);
		double [] distancias = t.calculateMHLNB(result);

		double minDist = distancias[0];
		String prediction = t.classes.get(0);
		for(int i = 0; i < distancias.length; i++) {
			if(distancias[i] < minDist) {
				minDist = distancias[i];
				prediction = t.classes.get(i);
			}
		}

		return prediction;
	}
	
	private static String euclideanDistance(Tree t, double []d) {
		double [] result = t.calculateAll(d);
		double [] distancias = t.calculateEucDistances(result);

		double minDist = distancias[0];
		String prediction = t.classes.get(0);
		for(int i = 0; i < distancias.length; i++) {
			if(distancias[i] < minDist) {
				minDist = distancias[i];
				prediction = t.classes.get(i);
			}
		}

		return prediction;
		
	}
	
	private static int knn = 5;
	private static String knn(Tree t, double []d) {
		double [] result = t.calculateAll(d);
		double [][] map = t.getMap();
		String [] classes = t.getTarget().clone();
		double [] distancesToPoints = new double[map.length];
		for(int i =0 ; i < map.length;i++) {
			distancesToPoints[i] = Arrays.euclideanDistance(result, map[i]);
		}
		Arrays.mergeSortBy(classes, distancesToPoints);

		String [] options = new String[knn];
		for(int i = 0; i < knn; i++) {
			options[i] = classes[i];
		}
		return Arrays.mostCommon(options);
	}
}
