=======
History
=======

0.3.5 (2023-02-14)
------------------
* Corrected distribution files for missing models

0.3.4 (2023-02-13)
------------------
* Corrected sampling with ConditionalInvariantModel 
* Added new weighted sampling method for aggregate graphs

0.3.3 (2023-12-25)
------------------
* Added the sparse ConditionalInvariantModel 
* Corrected issues with CremB sampling

0.3.2 (2023-12-01)
------------------
* Improved and corrected support for the computation of average nearest neighbour properties and degrees. 

0.3.1 (2023-11-20)
------------------
* Added to the spark module a function to compute the confusion matrix elements at various thresholds of the probability matrix.

0.3.0 (2023-11-03)
------------------
* Major update of graph classes into four new categories (Graph, DiGraph, MultiGraph, MultiDiGraph) in line with Networkx. 
* Cleaned up models and organized in three modules (dense, sparse, spark) based on how the computations are performed and results are stored. 
* Introduced better inheritance through Ensemble classes based on the newly defined graph classes. 
* Added testing in spark and updated testing of MultiDiGraph Ensemble classes.  

0.2.3 (2023-07-03)
------------------
* Improved and corrected spark submodule.

0.2.2 (2023-05-11)
------------------
* Created submodule spark for allowing some models to be parallelize computations using spark

0.2.1 (2021-08-03)
------------------
* Added option for faster computation of average nearest neighbour properties by allowing for multiple links between the same nodes.
* Added compression option in to_networkx function.

0.2.0 (2021-07-12)
------------------
* Added likelihood and nearest neighbour properties.
* Revisited API for measures to ensure correct recompute if necessary.

0.1.3 (2021-04-29)
------------------
* Added new option for fitting the stripe model that ensures that the minimum non-zero expected degree is one
* Corrected issue in expected degree calculations

0.1.2 (2021-04-07)
------------------
* Added scale invariant probability functional to all models
* Improved methods for convergence with change in API, xtol now a relative measure
* Added pagerank and trophic depth to the library
* Added methods for graph conversion to networkx
* Added methods for computing the adjacency matrix as a sparse matrix

0.1.1 (2021-03-29)
------------------
* Fixed bug in stripe expected degree computation
* Added testing of expected degree performance

0.1.0 (2021-03-29)
------------------
* Added the block model and group info to graphs
* Added fast implementation of theoretical expected degrees
* Fixed some compatibility issues with multiple item assignments

0.0.4 (2021-03-15)
------------------
* Fixed issues with slow pandas index conversion

0.0.3 (2021-03-14)
------------------
* Large changes in API with great improvements in usability
* Added sampling function
* Added RandomGraph model
* Added Graph classes for ease of use


0.0.2 (2020-11-13)
------------------
* Added steps for CI. 
* Corrected broken links. 
* Removed support for python 3.5 and 3.6

0.0.1 (2020-10-28)
------------------

* First release on PyPI. StripeFitnessModel available, all other model classes still dummies.

