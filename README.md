# ModelingProject_happiness
Repository for term project of Experimental Seminar: Computational Modeling

## DataPreprocessing and Modification
Running these files will cause errors. Processed csv files are already in the repo.
They are just for informing the preprocessing methods

Preprocessing includes changing the mat file to csv files.
Modifcation includes normalization and labeling process for each variable.

## Modeling
Each .py file contains models below
Modeling.py: original model
Modeling_additive.py: subjective original model
Modeling2_phatppe.py: phatppe model
Modeling2_mixed.py: subjective mixed model

Codes include MLE, individual bayesian and hierarchical bayesian estimation. 

You can run those code by using python command as below.
'''
python Modeling.py
'''

## Data simulation and parameter recovery analysis
To run DataSimulation file, files containing estimated parameters should be included in /outputs folder. 
Also, for RecoveryAnalysis file, original and simulated parameter information csv file should be included in the same folder 
