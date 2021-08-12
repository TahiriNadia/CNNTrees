# Simulation design
The proposed approach to calculating the multiple consensus trees and supertrees was tested on the basis of the following simulation algorithm, which consists of four main steps, described as follow:

1. The first step involves generating $K$ consensus phylogenetic trees $T_1, \cdots, T_K$ containing $n$ leaves each, where $K = 1, \cdots, 5$ and $n = $8, 16, 32, 64 or 128.


2. As part of the second step, for each phylogenetic tree $T_\textit{i} (\textit{i} = 1, \cdots, K)$, which was obtained at the previous step, a set of 20 phylogenetic trees $T_\textit{i}'$, corresponding to cluster $\textit{i}$, was randomly generated, considering that each $T_\textit{i}'$ tree differed from the $T_i$ tree by a certain number of coalescence / incomplete lineage sorting, creating patterns of incongruence between gene trees. HybridSim program created by Woodhamset al.(2016) was used to implement this approach. This tool allows the creation of phylogenetic trees in the presence of hybridization and horizontal gene transfer events. During the simulation using HybridSim, we changed the values of the hybridization rate and the coalescence rate. We have used HybridSim to estimate hybridization rates ranging from 1 to 4 in our experiments; by randomly drawing a value from this distribution. We were able to estimate these rates.  Each $T_i'$ was further classified into five intervals of coalescence parameters, which added noise into gene phylogenies, ranging from 10 (low noise) to 1 (high noise). All other HybridSim parameters were left at their defaults values. 


3. The third step involves a random removal of some species (branches adjacent to these species have also been removed) from the generated trees. In particular, the following proportion of species have been deleted: $0\%$, $10\%$, $25\%$ or $50\%$. In situations where we have incomplete trees, the RF distance was calculated between the maximum subtrees of two trees that have 4 or more identical leaves. For supertrees, We filtered the two sets of species from two input trees to get all species in common. As a result, we additionally evaluated the performances of our approach to work with an incomplete dataset. 


4. The fourth step will allow us to apply the data obtained in step 3 with our new CNNtrees approach. The CNNTrees approach used in this process was run with 10 epochs and we evaluated the performances in terms of accuracy and loss until the criterion score converges or the algorithm reaches a maximum of 10 epochs in the inner loop.



It was assumed that the quantity of clusters is also known. For testing, we used a computer with an Intel i7-8750H (2.5 GHz) processor and 32 GB of RAM.
