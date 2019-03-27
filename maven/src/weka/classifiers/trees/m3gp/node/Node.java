package weka.classifiers.trees.m3gp.node;

import java.io.Serializable;

import weka.classifiers.trees.m3gp.client.Constants;
import weka.classifiers.trees.m3gp.util.Mat;

/**
 * 
 * @author Jo�o Batista, jbatista@di.fc.ul.pt
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
	public Node(String [] term, int depth){
		if(
				Math.random() * (Constants.OPERATIONS.length + term.length + 1)< Constants.OPERATIONS.length +1 
				//Math.random()< t_rate 
				|| depth <= 1){
			int index = Mat.random(term.length);
			v = index < term.length-1? index :Math.random();
		}else{
			v = Mat.random(Constants.OPERATIONS.length);
			l = new Node(term, depth-1);
			r = new Node(term, depth-1);
		}
	}


	public double calculate(double [] vals) {
		if(isLeaf()){
			int vi = (int)v;
			return v != vi ? v : vals[vi];
		}else{
			double d = l.calculate(vals);
			switch((int)v){
			case 0://   +
				return d + r.calculate(vals);
			case 1://   -
				return d - r.calculate(vals);
			case 2://   *
				return d * r.calculate(vals);
			case 3://   //(protected division)
				try {
					return d / r.calculate(vals);
				}catch(Exception e) {
					return d;
				}
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
			if (v != (int)v)
				return v+"";
			else
				return "x"+(int)v;
		}else{
			return "( " + l.toString() + " " + Constants.OPERATIONS[(int)v] + " " + r.toString() + " )";
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

	public void clean() {
		if ( !isLeaf() ) {
			// + - * //
			if(v == 0 && l.isLeaf() && l.v == 0) {
				v = r.v;
				if(!r.isLeaf()) {
				l = r.l;
				r = r.r;
				}
			}
			if( (v == 0 || v==1) && r.isLeaf() && r.v == 0) {
				v = r.v;
				if(!l.isLeaf()) {
				l = l.l;
				r = l.r;
				}
			}
			if(v == 1 && l.isLeaf() && r.isLeaf() && r.v==l.v) {
				l=null;
				r=null;
				v=0;
			}
			if(v == 2 && (r.isLeaf() && r.v == 0) || (l.isLeaf() && l.v == 0)  )  {
				l = null;
				r = null;
				v = 0;
			}
			if(v == 3 && r.isLeaf() && r.v == 0) {
				v = l.v;
				if(!l.isLeaf()) {
				r= l.r;
				l = l.l;
				}
			}
		}
	}
	
	public void turnTerminal(String[] term) {
		l = null;
		r = null;
		int index = Mat.random(term.length);
		v = index < term.length-1? index :Math.random();
	}
	
	public void changeValue(String[] term) {
		if( isLeaf()){
			int index = Mat.random(term.length);
			v = index < term.length-1? index :Math.random();
		}else{
			v = Mat.random(Constants.OPERATIONS.length);
		}
	}
}