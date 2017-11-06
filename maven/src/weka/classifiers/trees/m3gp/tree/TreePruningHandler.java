package weka.classifiers.trees.m3gp.tree;

import java.util.ArrayList;

import weka.classifiers.trees.m3gp.node.Node;

public class TreePruningHandler {
/*
 * Para cada dimensao:
 * remove a dimensao e ve se o fitness piora, nesse caso volta a adiciona-la
 */
	public static Tree prun(Tree tree, double [][] data, String [] target, double trainFract){
		Tree t = new Tree(tree.cloneDimensions()); 
		Tree candidate = null;
		for(int i = 0; t.getDimensions().size() >1 && i < t.getDimensions().size(); i++) {
			ArrayList<Node> newDim = t.cloneDimensions();
			newDim.remove(i);
			candidate = new Tree(newDim);
			if(candidate.getTrainAccuracy(data, target,trainFract) > t.getTrainAccuracy(data, target,trainFract)) {
				t = new Tree(candidate.cloneDimensions());
				i--;
			}
		}
		return t;
	}
}//TODO ver porque reduz a precisao