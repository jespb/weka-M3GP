package weka.classifiers.trees.m3gp.node;

import java.io.Serializable;

import weka.classifiers.trees.m3gp.util.Mat;

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
	private static final String[] operations = "+ - * /".split(" ");
	double v;
	Node l;
	Node r;
	
	/**
	 * Basic constructor
	 * @param value
	 */
	public Node(double value){
		v = value;
	}

	/**
	 * Basic constructor
	 * @param left
	 * @param right
	 * @param op
	 */
	public Node(Node left, Node right, double op){
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
	public Node(String [] op, String [] term, double t_rate, int depth){
		if(Math.random() < t_rate || depth <= 0){
			int index = Mat.random(term.length);
			v = index < term.length-1? index :Math.random();
		}else{
			v = Mat.random(op.length);
			l = new Node(op, term, t_rate, depth-1);
			r = new Node(op, term, t_rate, depth-1);
		}
	}

	/**
	 * Used the node to calculate
	 * @param vals
	 * @return
	 */
	public double calculate(double [] vals){
		if(isLeaf()){
			if (v % 1 != 0)
				return v;
			else
				return vals[(int)v];
		}else{
			double d = 0;
			int v2 = (int)v;
			switch(v2){
			case 0://   +
				d = l.calculate(vals)+r.calculate(vals);
				break;
			case 1://   -
				d = l.calculate(vals)-r.calculate(vals);
				break;
			case 2://   *
				d = l.calculate(vals)*r.calculate(vals);
				break;
			case 3://   //(protected division)
				double div = r.calculate(vals);
				d = l.calculate(vals)/(div != 0 ? div : 1);
				break;
			}
			return d;
		}
	}

	/**
	 * Returns the number of nodes on this object, he himself included
	 * @return
	 */
	public int getSize(){
		if (isLeaf())
			return 1;
		else
			return 1 + l.getSize() + r.getSize();
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
	
	/**
	 * Returns the node under the String format
	 */
	public String toString(){
		if(isLeaf()){
			if (v<1)
				return v+"";
			else
				return "x"+(int)v;
		}else{
			return "( " + l + " " + operations[(int)v] + " " + r + " )";
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

	public int getDepth() {
		if(isLeaf()) {
			return 1;
		}else {
			return 1 + Math.max(l.getDepth(), r.getDepth());
		}
	}
}