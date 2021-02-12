# Extreme Value evolving Classifier - EVeC

The Extreme Value evolving Classifier (EVeC) is an evolving fuzzy-rule-based algorithm for online multi-label data streams classification. It offers a statistically well-founded approach to define the evolving fuzzy granules that form the antecedent and the consequent parts of the rules. The evolving fuzzy granules correspond to radial inclusion Weibull functions interpreted by the Extreme Value Theory as the limiting distribution of the relative proximity among the rules of the learning model. Regarding the consequent part of the rules, multiple versions are available:

- EVeC v.0: each rule corresponds to a single label
- EVeC v.1: the consequent part of the rules are formed by Takagi-Sugeno terms, calculated via least-squares regression
- EVeC v.2: the consequent part of the rules are formed by Takagi-Sugeno terms, calculated via Sparse Structure-Regularized Learning with Least Squares Loss (Least SRMTL)
- EVeC v.3 (not finished): the consequent part of the rules are formed by Takagi-Sugeno terms, calculated via Sparse Structure-Regularized Learning with Logistic Loss (Logistic SRMTL)

License
=======

This version of EVeC is released under the MIT license. (see LICENSE.txt).

Running
=======

EVeC was implemented with Python 3.6. After installing the prerequisites, you can run the file main.py or call the class EVeC.py.

Prerequisites
-------------

- matplotlib, sklearn, Cython: pip3 install matplotlib sklearn Cython

- libMR - Library for Meta-Recognition and Weibull based calibration of SVMdata. Used to apply the methods founded on the Extreme Value Theory. Available at https://pypi.org/project/libmr/. More information can be found in Scheirer, W. J., Rocha, A., Micheals, R. J., & Boult, T. E. (2011). Meta-recognition: The theory and practice of recognition score analysis. IEEE transactions on pattern analysis and machine intelligence, 33(8), 1689-1695. To install: pip install libmr

If running main.py, also install:

- tqdm: pip install tqdm

- MLflow - Open source platform to manage the ML lifecycle. Used to generate the results of the experiments. Available at https://mlflow.org/. To install: pip install mlflow

Contributor
===========

Developed by Amanda O. C. Ayres under supervision of Prof. Fernando J. Von Zuben

