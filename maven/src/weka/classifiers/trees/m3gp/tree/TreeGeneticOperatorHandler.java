package weka.classifiers.trees.m3gp.tree;

import java.util.ArrayList;

import weka.classifiers.trees.m3gp.node.Node;
import weka.classifiers.trees.m3gp.node.NodeHandler;
import weka.classifiers.trees.m3gp.population.Population;
import weka.classifiers.trees.m3gp.population.PopulationFunctions;
import weka.classifiers.trees.m3gp.util.Mat;

public class TreeGeneticOperatorHandler {
	public static int numberOfGeneticOperators = 5;

	public static Tree[] geneticOperation(Tree[] population, int tournamentSize, String[] op, String[] term, 
			double t_rate, int max_depth, double [][] data, String [] target, double train_p){
		Tree[] p = new Tree[3];
		p[0] = PopulationFunctions.tournament(population, tournamentSize);
		p[1] = PopulationFunctions.tournament(population, tournamentSize);
		p[2] = PopulationFunctions.tournament(population, tournamentSize);

		int operation = roulette(p[0].getGOA());

		Tree [] desc = null;
		switch(operation){
		case 0:
			desc = crossover1(p[0], p[1], data, target, train_p);
			break;
		case 1:
			desc = crossover2(p[0], p[1], data, target, train_p);
			break;
		case 2:
			desc = mutation1(p[0], op, term, t_rate, max_depth, data, target, train_p);
			break;
		case 3:
			desc = mutation2(p[0], op, term, t_rate, max_depth, data, target, train_p);
			break;
		case 4:
			desc = mutation3(p[0], op, term, t_rate, max_depth, data, target, train_p);
			break;
		case 5:
			desc = mutation4(p[0], op, term);
			break;
		case 6:
			desc = mutation5(p[0],term);
			break;
		case 7:
			desc = mutation6(p[0],term);
			break;
		case 8:
			desc = mutation7(p[0],term);
			break;
		case 9:
			desc = crossover3(p[0],p[1],p[2],data,target,train_p);
		}


		for( Tree t : desc){
			PopulationFunctions.fitnessTrain(t, data, target, train_p);
		}
		
		double parents = 0;
		double descendents = 0;
		
		for(int i = 0; i < desc.length; i++) {
			parents += PopulationFunctions.fitnessTrain(p[i], data, target, train_p);
			descendents += PopulationFunctions.fitnessTrain(desc[i], data, target, train_p);
		}
		
		if(descendents > parents) {
			Population.goAffinity[operation]+=0.1;
			if(Population.goAffinity[operation] > 5)
				Population.goAffinity[operation] = 5;
		}else {
			Population.goAffinity[operation]*=0.95;
			if(Population.goAffinity[operation] < 0.1)
				Population.goAffinity[operation] = 0.1;
		}

		/*
		if ( PopulationFunctions.betterTrain(desc[0], t1, data, target, train_p)){
			desc[0].incGOA(operation);
		}else{
			desc[0].decGOA(operation);
		}

		if(desc.length > 1){
			if(PopulationFunctions.betterTrain(desc[1], t2, data, target, train_p)){
				desc[1].incGOA(operation);
			}else{
				desc[1].decGOA(operation);
			}
			if(desc.length > 2){
				if(PopulationFunctions.betterTrain(desc[2], t3, data, target, train_p)){
					desc[2].incGOA(operation);
				}else{
					desc[2].decGOA(operation);
				}
			}
		}
		*/

		return desc;
	}


	private static Tree[] mutation4(Tree t1, String[] op, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		Node n = dim1.get(index1);

		Node r1 = NodeHandler.randomNode(n);

		r1.changeValue(op, term);

		return new Tree[] {new Tree(dim1, t1.getGOA(), t1.getGOAC())};
	}

	private static Tree[] mutation5(Tree t1, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		Node n = dim1.get(index1);

		Node r1 = NodeHandler.randomNode(n);

		r1.turnTerminal(term);

		return new Tree[] {new Tree(dim1, t1.getGOA(), t1.getGOAC())};
	}

	private static Tree[] mutation6(Tree t1, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		Node n = dim1.get(index1);

		n.turnTerminal(term);

		return new Tree[] {new Tree(dim1, t1.getGOA(), t1.getGOAC())};
	}
	
	private static Tree[] mutation7(Tree t1, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();
		
		while (dim1.size() > 1){
			dim1.remove(1);
		}

		Node n = dim1.get(0);

		n.turnTerminal(term);

		return new Tree[] {new Tree(dim1, t1.getGOA(), t1.getGOAC())};
	}


	public static Tree[] crossover1(Tree t1, Tree t2, double[][] data, String[] target, double train_p){
		ArrayList<Node> dim1 = t1.cloneDimensions();
		ArrayList<Node> dim2 = t2.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		int index2 = Mat.random(dim2.size());
		Node n = dim1.get(index1);
		dim1.set(index1, dim2.get(index2));
		dim2.set(index2, n);

		return new Tree[] {new Tree(dim1, t1.getGOA(), t1.getGOAC()), new Tree(dim2, t2.getGOA(), t2.getGOAC())};
	}

	public static Tree[] crossover2(Tree t1, Tree t2, double[][] data, String[] target, double train_p){
		ArrayList<Node> dim1 = t1.cloneDimensions();
		ArrayList<Node> dim2 = t2.cloneDimensions();

		Node p1 = dim1.get(Mat.random(dim1.size())).clone();
		Node p2 = dim2.get(Mat.random(dim2.size())).clone();
		Node r1 = NodeHandler.randomNode(p1);
		Node r2 = NodeHandler.randomNode(p2);
		NodeHandler.swap(r1,r2);

		return new Tree[] {new Tree(dim1, t1.getGOA(), t1.getGOAC()), new Tree(dim2, t2.getGOA(), t2.getGOAC())};
	}
	
	public static Tree[] crossover3(Tree t1, Tree t2, Tree t3, double[][] data, String[] target, double train_p){
		ArrayList<Node> dim1 = t1.cloneDimensions();
		ArrayList<Node> dim2 = t2.cloneDimensions();
		ArrayList<Node> dim3 = t3.cloneDimensions();

		Node p1 = dim1.get(Mat.random(dim1.size())).clone();
		Node p2 = dim2.get(Mat.random(dim2.size())).clone();
		Node p3 = dim3.get(Mat.random(dim3.size())).clone();
		Node r1 = NodeHandler.randomNode(p1);
		Node r2 = NodeHandler.randomNode(p2);
		Node r3 = NodeHandler.randomNode(p3);
		NodeHandler.swap(r1,r2);
		NodeHandler.swap(r2,r3);

		return new Tree[] {new Tree(dim1, t1.getGOA(), t1.getGOAC()), new Tree(dim2, t2.getGOA(), t2.getGOAC()),
				new Tree(dim3, t3.getGOA(), t3.getGOAC())};
	}

	public static Tree[] mutation1(Tree t1, String[] op, String[] term, double t_rate, int max_depth, double[][] data, String[] target, double train_p){
		ArrayList<Node> dim = t1.cloneDimensions();

		Node p1 = dim.get( Mat.random(dim.size()) );
		Node r1 = NodeHandler.randomNode(p1);
		NodeHandler.redirect(r1, new Node(op,term,t_rate,max_depth));

		return new Tree[] {new Tree(dim, t1.getGOA(), t1.getGOAC())};
	}

	public static Tree[] mutation2(Tree t1, String[] op, String[] term, double t_rate, int max_depth, double[][] data, String[] target, double train_p){
		ArrayList<Node> dim = t1.cloneDimensions();

		dim.add(new Node(op,term,t_rate,max_depth));

		return new Tree[] {new Tree(dim, t1.getGOA(), t1.getGOAC())};
	}

	public static Tree[] mutation3(Tree t1, String[] op, String[] term, double t_rate, int max_depth, double[][] data, String[] target, double train_p){
		ArrayList<Node> dim = t1.cloneDimensions();

		if(dim.size()>1)
			dim.remove( Mat.random(dim.size()) );

		return new Tree[] {new Tree(dim, t1.getGOA(), t1.getGOAC())};
	}


	private static int roulette(double [] v){
		double acc = 0;
		//System.out.println(v);
		for (double d : v){
			acc += d;
		}
		double pick = Math.random()*acc;
		for (int i = 0; i < v.length; i++){
			if(pick <= v[i])
				return i;
			pick -= v[i];
		}
		return 1/0;
	}
}
