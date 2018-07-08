# ordered-tree-sampler
A random [ordered tree](https://en.wikipedia.org/wiki/Tree_(graph_theory)#Plane_tree) sampler with the uniform (or custom) distribution. 

The sampler creates ordered trees by generating a random walk that follows the supplied distribution. 
This was done to demonstrate an efficient way to uniformly sample an ordered tree from the set of all ordered trees of a given size. 
The number of children of each node follows the supplied distribution. 
The uniform distribution is equivalent to the child-distribution pow(2,-i-1), where i is the number of children. 


### Requirements
* [NetworkX](https://networkx.github.io/): The tree is generated as a Graph object and plotting the trees.
* [Matplotlib](https://matplotlib.org/): Plotting the analysis of custom distributions.


### Examples

* Samples a random tree with 2500 nodes, selected at random with the uniform distribution.
  The uniform distribution is the default distribution of the sampler.
  ```    
  import tree
  nodes = 2500
  random_tree = tree.sample_tree(nodes)
  tree.plot_tree(random_tree)
  ```
  
  
* Samples a random binary tree with approximately 100 nodes.
  Note that it's impossible to get a binary tree with even number of nodes.
  To solve that, the sampler can adjust the size if the iterations all fail.
  ```    
  nodes = 100
  xi_binary = [1, 0, 1]  # The weights of 0 and 2 children are positive.
  random_tree, tree_size = tree.sample_tree(nodes, xi=xi_binary, adjustable_size=True)
  print("Number of nodes: " + str(tree_size))
  tree.plot_tree(random_tree)
  ```
  
  
* Samples a random tree from a custom distribution.
  If it is not important to get the exact number of nodes in the sampled tree, the speed can be
  increased by reducing the number of iterations it tries to generate the exact number.
  ```
  nodes = 1000
  xi_binary = [1, 0, 1, 0, 1, 0, 1]
  random_tree, tree_size = tree.sample_tree(nodes, xi=xi_binary, adjustable_size=True, max_iter=100)
  print("Number of nodes: " + str(tree_size))
  tree.plot_tree(random_tree)
  ```
  
* It is possible to analyse a distribution and see possible ends of walks to see if it can reach 0.
  The expected value should be close to 0 to increase the chance of getting a successful sample.
  Note that the variance increases as the number of nodes increases.
  ```
  nodes = 1000
  xi_custom = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  expected_value, variance = tree.analyse_density_function(nodes, xi_custom, iter=10000)
  print("Expected end of walk: " + str(expected_value))
  print("Variance: " + str(variance))
  ```
