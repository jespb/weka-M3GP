package m3gp.tree;

import java.util.ArrayList;

import m3gp.node.Node;
import m3gp.node.NodeHandler;
import m3gp.util.Mat;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class TreeCrossoverHandler {	
	/**
	 * Executes the crossover of a TreeSTGP by swaping a random node with 
	 * another random node from other TreeSTGP
	 * This method fails if a randomly generated number is lower than the 
	 * average of the Trees crossover odds
	 * @param parent1
	 * @param parent2
	 * @return
	 */
	public static Tree[] crossover(Tree parent1, Tree parent2, double [][] data, String [] target, double trainFract){
		// 50% crossover normal, 50% de trocar duas dimensoes
		ArrayList<Node> dim1 = clone(parent1.getDimensions());
		ArrayList<Node> dim2 = clone(parent1.getDimensions());

		switch(Mat.random(2)) {
		case 0://crossover normal
			Node p1 = dim1.get(Mat.random(dim1.size())).clone();
			Node p2 = dim2.get(Mat.random(dim2.size())).clone();
			Node r1 = NodeHandler.randomNode(p1);
			Node r2 = NodeHandler.randomNode(p2);
			NodeHandler.swap(r1,r2);
			break;
		case 1://troca dimensoes
			int index1 = Mat.random(dim1.size());
			int index2 = Mat.random(dim2.size());
			Node n = dim1.get(index1);
			dim1.set(index1, dim2.get(index2));
			dim2.set(index2, n);
			break;
		}
		return new Tree[] {new Tree(dim1), new Tree(dim2)};
	}
	
	private static ArrayList<Node> clone(ArrayList<Node> dim){
		ArrayList<Node> ret = new ArrayList<Node>();
		for(int i = 0; i < dim.size(); i++) {
			ret.add(dim.get(i).clone());
		}
		return ret;
	}
}