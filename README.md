# ModelingProject_happiness
Repository for term project of Experimental Seminar: Computational Modeling

## DataPreprocessing and Modification
!! These files are unnecessary to run, csv files are processed datasets. Just for informing the preprocessing methods !!
Preprocessing includes changing the mat file to csv files.
Modifcation includes normalization and labeling process for each variable.

## Modeling
Each .py file contains models below
Modeling.py: original model
Modeling_additive.py: subjective original model
Modeling2_phatppe.py: phatppe model
Modeling2_mixed.py: subjective mixed model

Codes include MLE, individual bayesian and hierarchical bayesian estimation. 

## Data simulation and parameter recovery analysis
To run DataSimulation file, files containing estimated parameters should be included in /outputs folder. 
Also, for RecoveryAnalysis file, original and simulated parameter information csv file should be included in the same folder 
