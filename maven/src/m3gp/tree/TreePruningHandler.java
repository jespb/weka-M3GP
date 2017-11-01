package m3gp.tree;

import java.util.ArrayList;

import m3gp.node.Node;

public class TreePruningHandler {
/*
 * Para cada dimensao:
 * remove a dimensao e ve se o fitness piora, nesse caso volta a adiciona-la
 */
	public static Tree prun(Tree tree, double [][] data, String [] target, double trainFract){
		Tree t = new Tree(clone(tree.getDimensions())), candidate = null;
		for(int i = 0; t.getDimensions().size() >1 && i < t.getDimensions().size(); i++) {
			ArrayList<Node> newDim = clone(t.getDimensions());
			newDim.remove(i);
			candidate = new Tree(newDim);
			if(candidate.getTrainAccuracy(data, target,trainFract) > t.getTrainAccuracy(data, target,trainFract)) {
				t = new Tree(clone(candidate.getDimensions()));
				i--;
			}
		}
		return t;
	}
	
	private static ArrayList<Node> clone(ArrayList<Node> dim){
		ArrayList<Node> ret = new ArrayList<Node>();
		for(int i = 0; i < dim.size(); i++) {
			ret.add(dim.get(i).clone());
		}
		return ret;
	}
}//TODO ver porque reduz a precisao