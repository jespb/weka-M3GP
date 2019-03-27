package weka.classifiers.trees.m3gp.tree;

import java.util.ArrayList;

import weka.classifiers.trees.m3gp.client.Constants;
import weka.classifiers.trees.m3gp.node.Node;
import weka.classifiers.trees.m3gp.node.NodeHandler;
import weka.classifiers.trees.m3gp.population.Population;
import weka.classifiers.trees.m3gp.population.PopulationFunctions;
import weka.classifiers.trees.m3gp.util.Arrays;
import weka.classifiers.trees.m3gp.util.Mat;

public class TreeGeneticOperatorHandler {
	
	public static Tree[] geneticOperation(Tree[] population, int tournamentSize, String[] term, double [][] data, String [] target){
		Tree[] p = new Tree[3];
		p[0] = changeGlobalValue(null, -1, -1, data, target,1,population,tournamentSize);
		p[1] = changeGlobalValue(null, -1, -1, data, target,1,population,tournamentSize);
		p[2] = changeGlobalValue(null, -1, -1, data, target,1,population,tournamentSize);

		int operation = roulette(p[0].getGOA());

		Tree [] desc = null;
		switch(operation){
		case 0:
			desc = crossover1(p[0], p[1], data, target);
			break;
		case 1:
			desc = crossover2(p[0], p[1], data, target);
			break;
		case 2:
			desc = mutation1(p[0], term, data, target);
			break;
		case 3:
			desc = mutation2(p[0], term, data, target);
			break;
		case 4:
			desc = mutation3(p[0], term, data, target);
			break;
		case 5:
			desc = mutation4(p[0], term);
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
			desc = crossover3(p[0],p[1],p[2],data,target);
}


		for( Tree t : desc){
			PopulationFunctions.fitnessTrain(t, data, target);
		}

		double parents = 0;

		for(int i = 0; i < desc.length; i++) {
			parents += PopulationFunctions.fitnessTrain(p[i], data, target);
		}
		parents /= desc.length;




		switch(Constants.PROBABILITY_ADAPTATION) {
		case 1:
			for(int i = 0; i < desc.length; i++) {
				if ( PopulationFunctions.fitnessTrain(desc[i], data, target) > parents) {
					desc[i].getGOA()[operation] = 1 - ( (1 - desc[i].getGOA()[operation]) * Constants.LEARNING_T );
				}else {
					desc[i].getGOA()[operation] *= Constants.LEARNING_T;
				}

				double [] np = new double [desc[i].getGOA().length];
				int acc = 0;
				int count = 0;
				for (int ii = 0; ii < np.length; ii++) {
					if( desc[i].getGOA()[ii] >= 0.05) {
						np[ii] = desc[i].getGOA()[ii];
						acc += desc[i].getGOA()[ii];
					}else {
						count++;
					}
				}
				for(int ii = 0; ii < np.length; ii++) {
					if (np[ii] == 0){
						np[ii] = (0.05 * acc) / (np.length - 0.05 * count);
					}
				}
				desc[i].setGOA( Arrays.normalize(cleanAndNormalize(np)) );
			}
			break;
			
			
		case 2:
			for(int i = 0; i < desc.length; i++) {
				if ( PopulationFunctions.fitnessTrain(desc[i], data, target) > parents) {
					desc[i].incGOA(operation);
				}else {
					desc[i].decGOA(operation);
				}
			}
			break;
			
		case -2:
			changeGlobalValue(desc, parents, operation, data, target,2,null,-1);
			break;
		}

		return desc;
	}
	
	private synchronized static Tree changeGlobalValue(Tree[] desc, double parents_fit, int operation, double[][]data, String[]target, int method, Tree[]population, int tournamentSize) {
		switch(method) {
		case 1: // obter dois descendentes
			return PopulationFunctions.tournament(population, tournamentSize);
		case 2: // actualizar valores
			for(int i = 0; i < desc.length; i++) {
				if ( PopulationFunctions.fitnessTrain(desc[i], data, target) > parents_fit) {
					Population.goAffinity[operation] = 1 - ( (1 - Population.goAffinity[operation]) * Constants.LEARNING_T );
				}else {
					Population.goAffinity[operation] *= Constants.LEARNING_T;
				}

				double [] np = Population.goAffinity;
				for(int ii = 0; ii< np.length; ii++) {
					np[ii] -=0.05;
					if(np[ii] < 0) {
						np[ii] = 0;
					}
				}
				Population.goAffinity = Arrays.normalize(np);
				np = Population.goAffinity;
				for(int ii = 0; ii< np.length; ii++) {
					np[ii] *= 1-0.05*np.length;
					np[ii] += 0.05;
				}
			}
			break;
		}
		return null;
	}

	/**
	 * Metodo da sara
	 * @param d
	 * @return
	 */
	public static double[] cleanAndNormalize(double [] d) {
		double [] np = new double [d.length];
		double acc = 0;
		double count = 0;
		double min = 0.05;
		for (int i = 0; i < np.length; i++) {
			if( d[i] >= min) {
				np[i] = d[i];
				acc += d[i];
			}else {
				count++;
			}
		}
		for(int i = 0; i < np.length; i++) {
			if (np[i] == 0){
				np[i] = (min * acc) / (np.length - min * count);
			}
		}
		return Arrays.normalize(np);

	}

	/**
	 * ST-XO
	 * @param t1
	 * @param t2
	 * @param data
	 * @param target
	 * @return
	 */
	public static Tree[] crossover1(Tree t1, Tree t2, double[][] data, String[] target){
		ArrayList<Node> dim1 = t1.cloneDimensions();
		ArrayList<Node> dim2 = t2.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		int index2 = Mat.random(dim2.size());
		Node n = dim1.get(index1);
		dim1.set(index1, dim2.get(index2));
		dim2.set(index2, n);

		return new Tree[] {new Tree(dim1, t1.getGOA()), new Tree(dim2, t2.getGOA())};
	}

	/**
	 * SWAP-DIM
	 * @param t1
	 * @param t2
	 * @param data
	 * @param target
	 * @return
	 */
	public static Tree[] crossover2(Tree t1, Tree t2, double[][] data, String[] target){
		ArrayList<Node> dim1 = t1.cloneDimensions();
		ArrayList<Node> dim2 = t2.cloneDimensions();

		Node p1 = dim1.get(Mat.random(dim1.size())).clone();
		Node p2 = dim2.get(Mat.random(dim2.size())).clone();
		Node r1 = NodeHandler.randomNode(p1);
		Node r2 = NodeHandler.randomNode(p2);
		NodeHandler.swap(r1,r2);

		return new Tree[] {new Tree(dim1, t1.getGOA()), new Tree(dim2, t2.getGOA())};
	}

	/**
	 * ST-MUT
	 * @param t1
	 * @param term
	 * @param data
	 * @param target
	 * @return
	 */
	public static Tree[] mutation1(Tree t1, String[] term, double[][] data, String[] target){
		ArrayList<Node> dim = t1.cloneDimensions();

		Node p1 = dim.get( Mat.random(dim.size()) );
		Node r1 = NodeHandler.randomNode(p1);
		NodeHandler.redirect(r1, new Node(term,Constants.MAX_DEPTH));

		return new Tree[] {new Tree(dim, t1.getGOA())};
	}

	/**
	 * ADD-DIM
	 * @param t1
	 * @param term
	 * @param data
	 * @param target
	 * @return
	 */
	public static Tree[] mutation2(Tree t1, String[] term, double[][] data, String[] target){
		ArrayList<Node> dim = t1.cloneDimensions();

		dim.add(new Node(term,Constants.MAX_DEPTH));

		return new Tree[] {new Tree(dim, t1.getGOA())};
	}

	/**
	 * REM-DIM
	 * @param t1
	 * @param term
	 * @param data
	 * @param target
	 * @return
	 */
	public static Tree[] mutation3(Tree t1, String[] term, double[][] data, String[] target){
		ArrayList<Node> dim = t1.cloneDimensions();

		if(dim.size()>1)
			dim.remove( Mat.random(dim.size()) );

		return new Tree[] {new Tree(dim, t1.getGOA())};
	}

	
	
	
	
	
	
	
	
	private static Tree[] mutation4(Tree t1, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		Node n = dim1.get(index1);

		Node r1 = NodeHandler.randomNode(n);

		r1.changeValue(term);

		return new Tree[] {new Tree(dim1, t1.getGOA())};
	}

	private static Tree[] mutation5(Tree t1, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		Node n = dim1.get(index1);

		Node r1 = NodeHandler.randomNode(n);

		r1.turnTerminal(term);

		return new Tree[] {new Tree(dim1, t1.getGOA())};
	}

	private static Tree[] mutation6(Tree t1, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();

		int index1 = Mat.random(dim1.size());
		Node n = dim1.get(index1);

		n.turnTerminal(term);

		return new Tree[] {new Tree(dim1, t1.getGOA())};
	}

	private static Tree[] mutation7(Tree t1, String[] term) {
		ArrayList<Node> dim1 = t1.cloneDimensions();

		while (dim1.size() > 1){
			dim1.remove(1);
		}

		Node n = dim1.get(0);

		n.turnTerminal(term);

		return new Tree[] {new Tree(dim1, t1.getGOA())};
}
	
	
	public static Tree[] crossover3(Tree t1, Tree t2, Tree t3, double[][] data, String[] target){
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

		return new Tree[] {new Tree(dim1, t1.getGOA()), new Tree(dim2, t2.getGOA()), new Tree(dim3, t3.getGOA())};
}
	
	
	
	
	
	
	
	

	private static int roulette(double [] v){
		double acc = 0;
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
