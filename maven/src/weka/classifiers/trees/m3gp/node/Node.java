package weka.classifiers.trees.m3gp.node;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Stack;

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
	 * Used the node to calculate (DO NOT USE, IT'S WAY SLOWER)
	 * @param vals
	 * @return
	 */
	public double calculate_stack(double [] vals){
		double value = 0;
		if(isLeaf()) {
			if (v != (int)v)
				value = v;
			else
				value = vals[(int)v];
		}else {
			Node curr = this;
			Stack<Node> stack = new Stack<Node>();
			while(!curr.isLeaf()) {
				stack.push(curr);
				curr = curr.l;
			}
			value = curr.calculate_stack(vals);
			while(! stack.isEmpty()) {
				curr = stack.pop();
				switch((int)curr.v){
				case 0://   +
					value += curr.r.calculate_stack(vals);
					break;
				case 1://   -
					value -= curr.r.calculate_stack(vals);
					break;
				case 2://   *
					value *= curr.r.calculate_stack(vals);
					break;
				case 3://   //(protected division)
					double div = curr.r.calculate_stack(vals);
					if(div != 0)
						value /= div;
					break;
				}
			}
		}
		return value;
	}
	
	public double calculate(double [] vals) {
		if(isLeaf()){
			int vi = (int)v;
			return v != vi ? v : vals[vi];
			/*
			if (v != vi)
				return v;
			else
				return vals[vi];*/
		}else{
			double d = l.calculate(vals);
			switch((int)v){
			case 0://   +
				d += r.calculate(vals);
				break;
			case 1://   -
				d -= r.calculate(vals);
				break;
			case 2://   *
				d *= r.calculate(vals);
				break;
			case 3://   //(protected division)
				double div = r.calculate(vals);
				if(div != 0)
					d /= div;
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
	public String toString(String [] operations){
		if(isLeaf()){
			if (v<1)
				return v+"";
			else
				return "x"+(int)v;
		}else{
			return "( " + l.toString(operations) + " " + operations[(int)v] + " " + r.toString(operations) + " )";
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