# Automatic-Feature-Engineering-for-TSC

This repository contains the code and data to reproduce results display in <link to paper>. The used TSC benchmark is the usual [UCR benchmark](https://www.timeseriesclassification.com) containing 112 datasets with same length and/or no missing values. The default train/test split is used here. In order to repeat the paper’s results, please use the following instructions.

Beside, to compare our results (Fig 1 and Fig9a in the paper) with others results (while not runing all the code) the results, in term of accuracy on the test sets, are available [here](results_tab/)

## A) Results for each libraries : 

### 1- Python part :

The `evaluator.py` provides the way to compile every tested python libraries. To see all the possible arguments one can provide in command line, type :

`python evaluator.py -h` 

In order to get results for all python libraries combined with one given classifier, e.g. RandomForestClassifier, type the following :

`python evaluator.py -d « my_path_to_data » -e RandomForestClassifier -kw ‘{«random_state»: 4}’` 

You can as well provide any sklearn compatible estimator, in that case, you should import it opening the `evaluator.py` file. All the tested classifiers are already imported.

Morover, add `-s 'csv'` to the previous statement in order to save the features files directly into datapath (modify into custom path?) in .csv format (.npz is also available), which will be useful to reproduce results from part <ref part stacking>.
  
NOTE : the `evaluator.py` is also compatible for multivariate benchmark

### 2- R part :
  
To get results for the only R tool (feasts) please type the following in your terminal :
  
`RScript feasts.R -d «my_path_to_data» -s « my_path_to_save_features»`

The features will be saved in .csv in the specified folder under the name ‘feasts_train.csv’ and ‘feasts_test.csv’.

### 3- MATLAB part :

To get results for the only matlab tool (hctsa) please type the following in your terminal (a matlab
licence is required) :
  
`matlab -r «n_features = 1000; datapath = «my_path_to_data»;savepath = «my_path_to_save_features» main; quit;»`

In order to use the matlab shortcut your .bashrc file should be provided with the path to your matlab version. 
Equivalently matlab keyword can be replace by `«Application/my_path_to_matlab»` or one can simply the Matlab desktop version and click the Run button.

Once you got every csv features for every datasets, you’re ready to get results for the part <ref part stacking>.

### 4- Get results for non-python libraries :
  
Once the csv files got saved in specified path, one can type the following :
 
`python evaluator.py -d «my_path_to_data» -e RandomForestClassifier -kw ‘{«random_state»: 4}’ --preload -np`
The `--preload` stands for preloading, which is mandatory for non-python libraries, and thus skip all features extraction step. `-np` add the non-python tools to libraries’ list.
Outputs files are then available in the results folder, created within the specified savepath folder.

  
## B) Stacking libraries :

 In order to reproduce the results for the stacking strategies, please run the following :
  
`python stacking.py -d -« my_path_to_data » -f « my_path_to_files » -m 'Features'`
  
With `-m` argument standing for «mode» (implement the 3 different stacking one can see in sota cd, i.e. Features, Features_noROCKET, Features_python) and one can add the `--all` keywords which is some boolean to decide to test all the possible stacking step for classif with provided lib or just compute the final one.

 ## Please refer to it as :
  

```
@INPROCEEDINGS{Gay2306:Automatic,
AUTHOR="Aur{\'e}lien Renault and Alexis Bondu and Vincent Lemaire and Dominique Gay",
TITLE="Automatic Feature Engineering for Time Series Classification: Evaluation
and Discussion",
BOOKTITLE="2023 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2023)",
ADDRESS="Queensland, Australia",
DAYS="17",
MONTH=jun,
YEAR=2023,
}
``` 

  
 
