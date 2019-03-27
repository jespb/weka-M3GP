package weka.classifiers.trees.m3gp.tree;

import java.util.ArrayList;

import weka.classifiers.trees.m3gp.node.Node;
import weka.classifiers.trees.m3gp.population.PopulationFunctions;

public class TreePruningHandler {
/*
 * Para cada dimensao:
 * remove a dimensao e ve se o fitness piora, nesse caso volta a adiciona-la
 */
	public static Tree prun(Tree tree, double [][] data, String [] target){
		Tree t = new Tree(tree.cloneDimensions()); 
		Tree candidate = null;
		for(int i = 0; t.getDimensions().size() > 1 && i < t.getDimensions().size(); i++) {
			ArrayList<Node> newDim = t.cloneDimensions();
			newDim.remove(i);
			candidate = new Tree(newDim);
			if(PopulationFunctions.betterOrEqualTrain(candidate, t, data, target)) {
				t = candidate;
				i--;
			}
		}
		//t.clean();
		return t;
	}
}