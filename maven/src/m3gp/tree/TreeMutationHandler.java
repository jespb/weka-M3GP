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
public class TreeMutationHandler {
	/**
	 * Mutates a TreeSTGP by selection a random node and replacing it with a new TreeSTGP
	 * This method fails if a randomly generated number if lower than the mutation odd
	 * @param string
	 * @param treeSTGP
	 * @param op
	 * @param term
	 * @param t_rate
	 * @param max_depth
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static Tree mutation(Tree original, String[] op, String[] term, 
			double t_rate, int max_depth, double [][] data, String [] target, double train_p) {
		// 33.(3)% mutacao normal, 33.(3)% adicionar uma dimensao, 33.(3)% remover uma dimensao

		ArrayList<Node> dim = clone(original.getDimensions());

		switch(Mat.random(dim.size()>1?3:2)) {
		case 0:		//Mutacao normal
			Node p1 = dim.get((int)(dim.size()*Math.random()));
			Node r1 = NodeHandler.randomNode(p1).clone();
			NodeHandler.redirect(r1, new Node(op,term,t_rate,max_depth));
			break;
		case 1:		//Add dimensao
			dim.add(new Node(op,term,t_rate,max_depth));
			break;
		case 2: 	//Remove dimensao
			dim.remove((int)(dim.size()*Math.random()));
			break;
		}
		return new Tree(dim);
	}
	
	private static ArrayList<Node> clone(ArrayList<Node> dim){
		ArrayList<Node> ret = new ArrayList<Node>();
		for(int i = 0; i < dim.size(); i++) {
			ret.add(dim.get(i).clone());
		}
		return ret;
	}
}