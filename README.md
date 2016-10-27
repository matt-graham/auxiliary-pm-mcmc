# Auxiliary pseudo-marginal MCMC python implementations

Python implementations of MCMC samplers in the auxiliary pseudo-marginal MCMC
framework as described in the paper 
[Pseudo-Marginal Slice Sampling](http://arxiv.org/abs/1510.02958) and
associated code for running Gaussian process classification model parameter
inference experiments.

A simpler single module Python implementation written by Iain Murray is also available [here](https://github.com/imurray/pmslice-python) - this is probably the simplest option for applying the method to your own problem.

## Requirements

The code has only been tested in 
[Python 2.7](https://www.python.org/download/releases/2.7/) and there are no 
guarantees it will work at all in other Python versions.

Minimal requirements for using the provided package are:

  * `numpy (1.9.2)`
  * `scipy (0.16.0)` (only required for `gpdemo` package)
  * `matplotlib (1.4.3)` (only required for `gpdemo` package)

The versions specified are those the code was developed and tested on - 
different versions may work as well.

To build the [Cython](http://cython.org/) modules yourself rather than using the
pre-built C-code you will also need `Cython (0.22)`.

For viewing the [IPython](http://ipython.org/) notebooks for running the 
experiments and analysing the results you will also need to have a working 
`IPython (3.1.0)` install and all the dependencies for the 
[IPython notebook](http://ipython.org/notebook.html) server. For example run

`pip install ipython[notebook]`

For the results analysis you will need to have a system `R` installation and
also have `rpy2 (2.7.0)` a python -- R interface installed.

## Installation

Run `python setup.py install` from main package directory to install the
package into the currently active python environment. This will also 
build the Cython modules in the package from the provided C-source.

If you have Cython installed you can also specify for the Cython code to be 
built directly from the Cython source by instead running 

`python setup.py install -use-cython`

For other install options refer run `python setup.py --help`

## Documentation

The code is organised in to three main sub-directories:

  * `auxpm`
  
    The Python package containing the modules implementing the different 
    auxiliary pseudo-marginal samplers variants (`auxpm.samplers`) and MCMC 
    update steps (`auxpm.mcmc_updates`).

  * `gpdemo`
  
    The Python package containing the modules implementing the functions
    specific to the  Gaussian process classification parameter inference 
    experiements.

  * `experiment_notebooks`
  
    A series of IPython notebooks using the above two packages to run
    Gaussian process classification parameter inference experiments for
    different sampling methods and analyse results.

## Running experiment notebooks

To run the experiment notebooks a local copy of any or all of the UCI 
classification datasets used in the experiments in the paper

> Filippone, Maurizio, and Mark Girolami. 
> 'Pseudo-marginal Bayesian inference for Gaussian processes.' 
> *Pattern Analysis and Machine Intelligence*, 
> IEEE Transactions on 36.11 (2014): 2214-2226.

will be required. These can be downloaded in the requisite space-delimited text
file format as part of the code associated with that paper at

<http://www.dcs.gla.ac.uk/~maurizio/Code/code_pseudo_marg.tar.gz>

The data files are in the `section4.4/DATA/clean` sub-directory of the archive.
Each dataset has a text file with suffix `_X.txt` containing the input features 
and `_y.txt` suffice containing the targets. 
    
These datasets were originally taken from the UCI Machine Learning Repository

> Lichman, M. (2013). 
> UCI Machine Learning Repository <http://archive.ics.uci.edu/ml>. 
> Irvine, CA: University of California, 
> School of Information and Computer Science.

The relevant text data files should be placed under a `uci` sub-directory which
itself is placed in a directory readable by the current user and the path to
which is specified in a environment variable `DATA_DIR` defined for the current
user. So for example if the Pima Indians dataset files are taken from the
archive linked to above, the inputs `pima_X.txt` and outputs `pima_y.txt` files
should exist respectively at

    $DATA_DIR/uci/pima_X.txt
    $DATA_DIR/uci/pima_y.txt

assuming Unix type directory separators and environment variable syntax.
    
When running the experiment notebooks it is expected that a further `EXP_DIR`
environment variable will be defined for the current user which specifies a
path writeable by the current user to output experiment results to, with
results being placed under a sub-directory `apm_mcmc` which should be created
before running any of the notebooks.
