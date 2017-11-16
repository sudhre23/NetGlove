# NetGlove

The advancement of representation learning has allowed us to rep- resent graphs as vector in a lower dimensional space. The biggest advantage of doing this is to apply known algorithms in vector space to understand properties of the underlying graphs.  The focus is on the problems of community detection which can be seen as a unsupervised learning problem, and link prediction which is a supervised learning problem in the embedding space. Various other vector embedding techniques such as LINE, Node2Vec were also analyzed and compared against NetGlove embeddings in the community detection task. 

The modified loss function is given as follows:

![Equation:1](https://latex.codecogs.com/gif.latex?J%3D%5Csum_%7Bi%7D%5Csum_%7Bj%7Cd_%7Bij%7D%3Ck%7Df%5Cleft%28%5Cfrac%7B1%7D%7Bd_%7Bij%7D%7D%5Cright%29%5Cleft%28w_%7Bi%7D%5E%7BT%7Dw_%7Bj%7D-%5Cfrac%7B1%7D%7Bd_%7Bij%7D%7D%5Cright%29%5E%7B2%7D)

where $d_{ij}$ corresponds to the shortest path between node i and node j in the graph. $w_{i}$ and $w_{j}$ correspond to node vectors. We solve this optimization problem using gradient descent to obtain the node representations wi and wj. We use the average of these embeddings to obtain the node vector for any node in the graph. 

## Community Detection

A simple K-Means is run on the set of node representations we have from the graph. Lancichinetti–Fortunato–Radicchi (LFR) benchmark graphs were used to test the efficiency of the community detection algorithm with varying mixing parameters. Furthermore, since the community label information from the LFR graphs and the communities evaluated by our embeddings to evaluate the NMI measure. This metric was taken as a measure to compare our algorithm with other methods in the literature.

## Code Organization

runner.py is the main file to run the application. In order to use this code, the following arguements need to be passed:

1. Input file location : The file location where the input graph (in edge list format) is stored

1. Output Location: The directory where the generayed embeddings need to be stored

1. Number of iterations to run for optimizing the loss function

1. The learning rate to be used for the gradient descent procedure

Example command to run the application:

python runner.py --input 'data/karate.edgelist' --output 'emb/karate.emb' --iterations 10 --learning-rate 0.1

#### Optional Arguements

1. --vector-size : Number of dimensions needed in the output node embeddings (default value is 100)

1. --distance-threshold: The nodes at a certain hop distance which needs to be considered for populating the node co-occurance matrix. Default value is 5, which means that only nodes at a distance of 5 units from the source graph is used for populating the co-occurance matrix row for the source node.

1. --weighted or --unweighted : By default the graph is assumed to be unweighted. Specify the weighted parameter if the graph is weighted.


## Results of NetGlove on LFR (Lancichinetti–Fortunato–Radicchi) graphs

#### LFR Graphs

The LFR model generates scale free networks and then rewires some links so that the ratio of internal to external links (as defined with respect to community structure) changes with respect to a mixing coefficient μ. If the value of μ is greater, the proportion of external links of a node (links to outside the community) increases, leading to ill-defined community structure. For an ideal community detection method, we want the performance and accuracy to be good even for high values of μ, or show less degradation at higher values of μ. 

The LFR model also provides ground truth for generated graphs, which we used to evaluate the result of the clustering. Normalized mutual information between the results of the clustering and the ground truth was evaluated for different values of μ and different values of the average degree of nodes. The number of clusters for K Means was determined using the number of communities in the ground truth information. 

#### Comparison of NetGlove against other community detection methods on LFR graphs

Results of evaluatuon on LFR graphs with average degree equal to 20 and maximum degree equal to 50

![alt text](https://github.com/sudhre23/NetGlove/blob/master/images/NMI.png)

We can see that NetGlove (green line) outperforms the other prevalent methods at higher mixing parameters for the community detection task. 

Visualization of the communities detected by NetGlove in the karate club graph

![alt text](https://github.com/sudhre23/NetGlove/blob/master/images/karate.png)

Visualization of the communities detected by NetGlove in the Les-Miserable Graph

![alt text](https://github.com/sudhre23/NetGlove/blob/master/images/lesmis.png)
