package weka.classifiers.trees.m3gp.util;

import java.io.Serializable;

public class Point implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double [] coor;
	String label;
	
	public Point(double [] c, String l){
		coor=c;
		label=l;
	}
}
