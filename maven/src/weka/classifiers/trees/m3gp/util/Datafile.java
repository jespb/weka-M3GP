package weka.classifiers.trees.m3gp.util;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;

import weka.classifiers.trees.m3gp.tree.Tree;

public class Datafile implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public static ArrayList<Object> generations;
	
	public Datafile() {
		generations=new ArrayList<Object>();
	}
	
	public void addGen(Tree[] population) {
		generations.add(population);
	}
	
	public void writeToFile(String filename) throws FileNotFoundException, IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
		oos.writeObject(generations);
		oos.close();
	}
	
	
}
