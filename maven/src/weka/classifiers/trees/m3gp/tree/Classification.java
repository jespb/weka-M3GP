package weka.classifiers.trees.m3gp.tree;

import weka.classifiers.trees.m3gp.client.Constants;

public class Classification {
	
	public static String predict(Tree t, double[] d) {
		String p = null;
		switch(Constants.DISTANCE_USED) {
		case 1:
			p = mahalanobisDistance(t, d);
			break;
		case 2:
			p = euclideanDistance(t,d);
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
}