# ChromaWalker: Exploring chromatin hierarchical organization via Markov State Modelling

ChromaWalker is a Python package for exploring chromatin architecture via the Markov State Model analysis of Hi-C data as delineated in

<img width="940" alt="image" src="https://github.com/eguarnera/ChromaWalker/assets/8078280/1384a3e1-2079-41d6-baae-c9fb7d781e15">


Tan, Z. W.; Guarnera, E.; Berezovsky, I. N. Exploring Chromatin Hierarchical Organization via Markov State Modelling. PLoS Computational Biology 2018, 14 (12). https://doi.org/10.1371/journal.pcbi.1006686.

In this work, we proposed a new approach for extracting robust genomic partitions from HiCdata, seeking to capture the hallmarks of chromatin structure and organization by considering the entire interaction landscape of this complex system. The objectives are to identify and study structural features of chromatin from Hi-C interaction data and to find a connection between these features and data on epigenetic regulation. We introduced a Markov State Model (MSM) approach with minimal assumptions and parameters on the chromatin interaction network, aiming at identifing structural partitions and their interactions. By analogy with a biomolecule moving and interacting in condensed chromatin, the MSM allows one to explore chromatin structure using a “probe” randomly walking in the contact energy landscape derived from Hi-C data. Given the multiscale nature of the data-derived contact energy landscape and the metastability of the corresponding MSM, we can identify regions of dense intra- and inter-chromosomal interactions, linkers between these regions, as well as the overall topology of individual chromosomes and the complex structures that chromosomes form by interacting with each other. The MSM approach and the metastability analisys of the contact energy landscape derived from Hi-C data are a development of a previous work in the context of protein dynamics (see Guarnera, E.; Vanden-Eijnden, E. Optimized Markov State Models for Metastable Systems. Journal of Chemical Physics 2016, 145 (2). https://doi.org/10.1063/1.4954769).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Deployment on a server should require just the same steps.

### Prerequisites

This library is built for Python 2.7.

List of required Python libraries (versions):
* Numpy (1.14)
* Scipy (1.0)
* Pandas (0.22)
* Matplotlib (2.0)

### Installing

1. Install the SciPy stack

The most convenient way is to use pip:

```
python -m pip install --user numpy scipy matplotlib pandas
```

2. Copy package source to Python path
3. Basic usage of the package can be inferred by looking at the code that's run when you execute ChromaWalker.py as a main program. In other words, look at how the program is run in ChromaWalker.py, under the conditional statement:

```
if __name__ == “__main__”:
```

## License
MIT license
