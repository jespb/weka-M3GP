package m3gp.tree;

import java.util.ArrayList;

import m3gp.node.Node;

public class TreeM3GPPruningHandler {
/*
 * Para cada dimensao:
 * remove a dimensao e ve se o fitness piora, nesse caso volta a adiciona-la
 */
	public static TreeM3GP prun(TreeM3GP tree, double [][] data, String [] target, double trainFract){
		TreeM3GP t = tree.clone(), candidate = null;
		for(int i = 0; t.getDimensions().size() >1 && i < t.getDimensions().size(); i++) {
			ArrayList<Node> newDim = clone(t.getDimensions());
			newDim.remove(i);
			candidate = new TreeM3GP( newDim);
			if(candidate.getTrainAccuracy(data, target,trainFract) > t.getTrainAccuracy(data, target,trainFract)) {
				t = candidate;
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
