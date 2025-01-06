# CH_BioGCN

# This is the file library for our Cancer Hallmark based BioGCN work
## The current purpose of this file library is to provide reviewers with reproducible experimental results, including some experimental codes.
code file :             <br />
&emsp;&emsp; 1. 06_BioGCN_load_model_and_test_pred.py            <br />
data file :             <br />
&emsp;&emsp; 1. Training and testing data ： cleaned_data_out.csv            <br />
&emsp;&emsp; 2. Training and testing label : cleaned_data_test_12_7.xlsx            <br />
&emsp;&emsp; 3. Diagram relationship file ： adjacency_matrix            <br />
&emsp;&emsp; 4. saved model file :             <br />
&emsp;&emsp;&emsp;&emsp; train : test = 0.6 : 0.4 : save_model(0.4)_18            <br />
&emsp;&emsp;&emsp;&emsp; train : test = 0.8 : 0.2 : save_model(0.2)_12            <br />
analysis file :     <br />
&emsp;&emsp;Analysis file of experimental results : analysis    <br />


## If you wish to repeat our experiment, please refer to the following configuration :     <br />

tensorflow=2.8.2=gpu_py39hc0c9373_0     <br />
keras=2.8.0=py39h06a4308_0     <br />
shap=0.45.1     <br />
scikit-learn=1.4.1.post1     <br />
matplotlib=3.9.2     <br />

# We will upload the complete code and data to this library after the paper is accepted. Stay tuned!


