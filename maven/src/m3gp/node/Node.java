package m3gp.node;

import java.io.Serializable;

import m3gp.util.Mat;

/**
 * 
 * @author João Batista, jbatista@di.fc.ul.pt
 *
 */
public class Node implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	String v = null;
	Node l = null;
	Node r = null;
	
	public static int index;

	/**
	 * Basic constructor
	 * @param value
	 */
	public Node(String value){
		v = value;
	}

	/**
	 * Basic constructor
	 * @param left
	 * @param right
	 * @param op
	 */
	public Node(Node left, Node right, String op){
		l = left;
		r = right;
		v = op;
	}

	/**
	 * Constructor
	 * @param op
	 * @param term
	 * @param t_rate
	 * @param depth
	 */
	public Node(String [] op, String [] term, double t_rate, int depth, int maxDepth){
		if(depth>0 && Math.random() < t_rate || depth >= maxDepth){
			v = term[Mat.random(term.length)];
		}else{
			v = op[Mat.random(op.length)];
			l = new Node(op, term, t_rate, depth+1, maxDepth);
			r = new Node(op, term, t_rate, depth+1, maxDepth );
		}
	}

	/**
	 * Used the node to calculate
	 * @param vals
	 * @return
	 */
	public double calculate(double [] vals){
		if(isLeaf()){
			if(v.startsWith("x")){
				int index = Integer.parseInt(v.substring(1));
				return vals[index];
			}else{
				if(v.equals("e"))
					return Math.E;
				else
					return Double.parseDouble(v);
			}
		}else{
			double d = 0;
			switch(v){
			case "+":
				d = l.calculate(vals)+r.calculate(vals);
				break;
			case "-":
				d = l.calculate(vals)-r.calculate(vals);
				break;
			case "*":
				d = l.calculate(vals)*r.calculate(vals);
				break;
			case "/":
				double div = r.calculate(vals);
				d = l.calculate(vals)/(div != 0 ? div : 1);
				break;
			}
			return d;
		}
	}

	/**
	 * Returns the node under the String format
	 */
	public String toString(){
		if(isLeaf()){
			return v;
		}else{
			return "( " + l + " " + v + " " + r + " )";
		}
	}

	/**
	 * Returns true if the node is a leaf
	 * It's assumed that the node either had two children or none
	 * for a faster response
	 * @return
	 */
	private boolean isLeaf(){
		return l==null;// &&r==null;
	}

	/**
	 * Constructor
	 * @param s
	 */
	public Node(String [] s) {
		if(s[index].equals("(")){
			index ++;
			l = new Node(s);
			v = s[index];
			index++;
			r = new Node(s);
			index++;
		}else{
			v = s[index];
			index++;
		}
	}

	/**
	 * Returns the number of nodes on this object, he himself included
	 * @return
	 */
	public int size(){
		if (isLeaf())
			return 1;
		else
			return 1 + l.size() + r.size();
	}

	/**
	 * Clones a node
	 */
	public Node clone(){
		if(isLeaf()){
			return new Node(v);
		}else{
			return new Node(l.clone(), r.clone(), v);
		}
	}
}