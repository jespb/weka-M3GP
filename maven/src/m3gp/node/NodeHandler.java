package m3gp.node;

import m3gp.util.Mat;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class NodeHandler {
	/**
	 * Returns a random node inside n
	 * @param n
	 * @return
	 */
	public static Node randomNode(Node n){
		return index(n, Mat.random(n.size()));
	}

	/**
	 * Returns a node inside n using index i
	 * @param n
	 * @param i
	 * @return
	 */
	private static Node index(Node n, int i){
		if(i==0) return n;
		int lsize = n.l.size();
		if(lsize < i){
			return index (n.r, i - lsize - 1);
		}else{
			return index (n.l, i - 1);
		}	
	}

	/**
	 * Redirects r1's subnodes and values to r2's
	 * @param r1 original node
	 * @param r2 target node
	 */
	public static void redirect(Node r1, Node r2) {
		r1.l = r2.l;
		r1.r = r2.r;
		r1.v = r2.v;
	}
}