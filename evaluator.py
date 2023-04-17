import time
import random
import json
import argparse
import matplotlib.pyplot as plt
from numpy import float32
from tqdm import tqdm

from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score, log_loss, accuracy_score, 
                                mean_squared_error, mean_absolute_error)
from sklearn.utils.extmath import softmax

from extractors import *
from utils import *

class Evaluator(BaseEstimator):

    def __init__(
        self, 
        library="tsfel",
        lib_kwargs={},
        clf=RandomForestClassifier,
        clf_kwargs={
            'random_state':4
        },
        pre_load=False,
        task='classification',
        scale=False    
    ):
        super().__init__()

        self.extractor_name = library
        self.clf = clf(**clf_kwargs)
        self.clf_kwargs = clf_kwargs
        self.lib_kwargs = lib_kwargs
        self.pre_load = pre_load
        self.task = task
        self.scale = scale
        self.scaler = None

        self.time_extr = 0
        self.time_tsf = 0
        self.time_clf = 0
        self.n_features = 0
        self.extractor = None

        # set of all supported libraries 
        assert self.extractor_name in ['minirocket', 'intervals', 'intervals_c22', 'shapelet', 'tsfel', 'feasts', 'hctsa', 'tsfeatures',
                                        'tsfresh', 'rocket', 'multirocket', 'catch22', 'featuretools', 'featuretools_1k', 'signature']

        # no python libraries, pre-loading is mandatory
        if self.extractor_name in ['feasts', 'hctsa']:
            assert self.pre_load == True

        if self.extractor_name == 'intervals_c22':
            self.extractor_name = 'intervals'
            lib_kwargs['transformers'] = catch22.Catch22(n_jobs=lib_kwargs["n_jobs"])
            lib_kwargs['n_intervals'] *= 10 / 22 # update n_intervals wrt max_features
            lib_kwargs['n_intervals'] = int(lib_kwargs['n_intervals'])

        if self.extractor_name == 'featuretools_1k':
            self.extractor_name = 'featuretools'
            # add features to reach 1k by allowing column combinations
            lib_kwargs['tsf_primitives_all'] = ['add_numeric', 'subtract_numeric', 'multiply_numeric']
    
    def set_params(self, **params):
        if 'clf' in params:
            self.clf = params['clf'](**self.clf_kwargs)
    
        if 'pre_load' in params:
            self.pre_load = params['pre_load']
        
        if 'scaler' in params:
            self.scaler = params['scaler']

        return self

    def prepare_X(self, X, y=None):
        self.extractor = eval(self.extractor_name.upper())(**self.lib_kwargs)
        start_time = time.time()
        # extract features
        X_features = self.extractor.fit_extract(X, y)
        self.time_extr = time.time() - start_time

        return X_features.astype(float32).fillna(0)

    def fit(self, X, y):
        if self.pre_load:
            X_fts = X
        else:
            # get features
            X_fts = self.prepare_X(X, y).fillna(0)
        self.n_features = X_fts.shape[1]

        if self.scale:
            self.scaler = StandardScaler() if not self.scaler else self.scaler
            X_fts = self.scaler.fit_transform(X_fts)

        start_time = time.time()
        self.clf.fit(X_fts, y)
        self.time_clf = time.time() - start_time

        return self
    
    def transform(self, X, y):
        if self.pre_load:
            # no effect if pre-loading
            X_tsf = X
        else:
            start = time.time()
            X_tsf = self.extractor.transform(X, y).astype(float32).fillna(0)
            self.time_tsf = time.time() - start
        return self.scaler.transform(X_tsf.fillna(0)) if self.scaler else X_tsf
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def auc(self, X, y):
        n_class = len(np.unique(y))
        try:
            multiclass = 'ovr' if n_class > 2 else 'raise'
            probas = self.clf.predict_proba(X) if n_class > 2 else self.clf.predict_proba(X)[:,1]

        # for classifier without predict_poba(), generate probas passing decision scores into softmax
        # e.g. RidgeClassifier
        except AttributeError:
            d = self.clf.decision_function(X) if n_class > 2 else np.c_[self.clf.decision_function(X)]
            probas = softmax(d)
            
        return roc_auc_score(y, probas, multi_class=multiclass)
    
    def balanced_acc(self, X, y):
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)
    
    def f1_score(self, X, y):
        n_class = len(np.unique(y))
        avrg = 'micro' if n_class > 2 else 'binary'
        return f1_score(y, self.predict(X), average=avrg)

    def neg_log_likelihood(self, X, y):
        try:
            probas = self.clf.predict_proba(X)
        except AttributeError:
            n_class = len(np.unique(y))
            d = self.clf.decision_function(X) if n_class > 2 else np.c_[self.clf.decision_function(X)]
            probas = softmax(d)

        return log_loss(y, probas)
    
    def rmse(self, X, y):
        return mean_squared_error(y, self.predict(X), squared=False)

    def mae(self, X, y):
        return mean_absolute_error(y, self.predict(X))

    
    def get_results(self, X, y):
        res_metrics = [
            self.score(X, y),
            self.balanced_acc(X, y),
            self.auc(X, y),
            self.f1_score(X, y),
            self.neg_log_likelihood(X, y)
            ] if self.task=='classification' else \
            [self.rmse(X, y), self.mae(X, y)] 

        res = res_metrics + [
            self.time_extr,
            self.time_tsf, 
            self.n_features,
            self.time_extr / self.n_features,
            self.time_clf, 
            self.time_extr + self.time_tsf + self.time_clf
        ] 

        cols_metrics = ["Accuracy", "Balanced accuracy", "AUC", "f1", "NLL"] if self.task=='classification' else \
            ["RMSE", "MAE"]

        results = pd.DataFrame(
            columns= cols_metrics + \
                ["Run time (extr)", "Run time (tsf)", "Num features", 
                "Time/feature", "Run time (classif)", "Total time"],
            data=np.expand_dims(np.array(res), 1).T
        )
        return results


def run_tests(
    datasets, 
    libraries,
    max_features, 
    classifier,
    clf_kwargs={'random_state':4},
    scale=False,
    na_mode='drop', # drop series with nan by default, other options are : "median" or some int/float
    DATAPATH="scripts/UCRArchive_2018",
    SAVEPATH="scripts/UCRArchive_2018",
    to_save=False, 
    to_load=False,
    raw_data=None,
    task='classification',
    verbose=True,
    n_jobs=1
):
    """
    Main function running tests

    Parameters
    ----------
    datasets : list of the names of the datasets on which to apply features extraction
    libraries : list of the libraries to use for feature extraction
    max_features : int limit for the number of features when infinite feature space 
    classifier : sklearn estimator to train on feature matrices
    clf_kwargs : dict of args for the estimator
    scale : bool whether to apply scaling before features being processed by estimator
    na_mode : str/int/float ["drop", "median", int/float] deal with missing values,
                either dropping series with nans or replace them with median or specified int/float 
    DATAPATH : str path to datasets 
    SAVEPATH : str path to send features files / from which to load them 
    to_save : str ["csv", "npz", False] Whether to save features matrix, in which format
    to_load : bool whether to preload features, jumping over the exctraction step
    raw_data : str [None, "only", "add"] whether to apply estimator on features only (None),
                raw series only ("only"), or both ("add")
    task : str ["classification", "regression"] task to perform, change evaluation metrics
    verbose : bool whether to display progress bars or not
    n_jobs : int number of simoultaneously parallel jobs during extraction when compatible

    Returns
    -------
    pd.Dataframe containing evaluation metrics 
    """

    assert raw_data in ['only', 'add', None]
    if raw_data is not None:
        to_save = False
        to_load = True
        if raw_data=='only':
            libraries=['tsfel'] # not used, only prevent errors to be raised
    
    if to_save:
        assert to_load==False

    classif = True if task=='classification' else False  

    if classif:
        col = ["Name", "Strategy", "Num dim", "Num class", "Prior most freq class"]
    else:
        col = ["Name", "Strategy", "Num dim"]
    df_infos = pd.DataFrame(columns=col)

    df_results = pd.DataFrame(
        columns=["Accuracy", "Balanced accuracy", "AUC", "f1", "NLL", "Run time (extr)",
        "Run time (tsf)", "Num features", "Time/feature", "Run time (classif)", "Total time"]
    ) if classif else \
    pd.DataFrame(
        columns=["RMSE", "MAE", "Run time (extr)", "Run time (tsf)", "Num features",
        "Time/feature", "Run time (classif)", "Total time"])

    pbar_data = tqdm(datasets) if verbose else datasets
    for data in pbar_data:
        pbar_data.set_description("Processing %s" % data) if verbose else -1
        # load sktime formated datasets
        X_train_nested, X_test_nested, Y_train, Y_test = load_ts_dataset(data, DATAPATH, classif=classif)

        if classif: # encode classification labels
            lb = LabelEncoder()
            y_train = lb.fit_transform(Y_train)
            y_test = lb.transform(Y_test)
        else:
            y_train = Y_train.astype(np.float32)
            y_test = Y_test.astype(np.float32)
        
        X_train_cleaned, y_train = deal_na_values(
            convert_nested(X_train_nested, to_numpy=True), y_train, mode=na_mode
        )
        X_train_cleaned = from_3d_numpy_to_nested(X_train_cleaned) # re convert to nested type to preserve compatibility

        X_test_cleaned, y_test = deal_na_values(
            convert_nested(X_test_nested, to_numpy=True), y_test, mode=na_mode
        )
        X_test_cleaned = from_3d_numpy_to_nested(X_test_cleaned)

        pbar_lib = tqdm(libraries, leave=False) if verbose else libraries
        for lib in pbar_lib:
            pbar_lib.set_description("Testing %s" % lib) if verbose else -1

            def_kwargs = get_lib_args(max_fts=max_features, n_cpu=n_jobs)
            if lib=='intervals_c22':
                lib_args = def_kwargs['intervals']
            elif lib=='featuretools_1k':
                lib_args = def_kwargs['featuretools']
            else:
                lib_args = def_kwargs[lib]

            uniq_labels, counts = np.unique(y_train, return_counts=True)
            gen_infos = [data, lib, X_train_nested.shape[1], 
                        len(uniq_labels), np.max(counts)/len(y_train)] if classif else \
                    [data, lib, X_train_nested.shape[1]]
            gen_infos = pd.DataFrame(gen_infos).T
            gen_infos.columns = col
            df_infos = pd.concat(
                (df_infos, gen_infos), axis=0
            )
            
            eva = Evaluator(
                library=lib, 
                lib_kwargs=lib_args, 
                clf=classifier,
                clf_kwargs=clf_kwargs,
                pre_load=to_load,
                task=task,
                scale=scale
            )

            if to_save:
                X_train = eva.prepare_X(X_train_nested, y_train).fillna(0)
                X_test = eva.transform(X_test_nested, y_test).fillna(0)

                is_csv = True if to_save=='csv' else False

                files = {'X_train':X_train, 'X_test':X_test}
                save_features_matrix(SAVEPATH, data, lib, to_csv=is_csv, **files)

                eva.set_params(**{'pre_load':True})
            elif to_load:
                if raw_data=='only':
                    # learn a classifier only on the raw data 
                    X_train = from_nested_to_2d_array(X_train_nested)
                    X_test = from_nested_to_2d_array(X_test_nested)
                else:
                    X_fts = load_features_matrix(SAVEPATH, data, lib)

                    up_bound = np.finfo(np.float32).max
                    low_bound = np.finfo(np.float32).min
                    
                    if raw_data=='add':
                        # concatenate extracted features and raw data
                        X_train = np.concatenate((from_nested_to_2d_array(X_train_nested).values,
                                                    np.nan_to_num(X_fts['X_train'].astype(np.float32))), axis=1)
                        X_test = np.concatenate((from_nested_to_2d_array(X_test_nested).values, 
                                                    np.nan_to_num(X_fts['X_test'].astype(np.float32))), axis=1)
                    else:
                        # nan_to_num as some of non python libraries may have nan or values to high/low for float32
                        X_train = np.nan_to_num(X_fts['X_train'].astype(np.float32), posinf=up_bound, neginf=low_bound)
                        X_test = np.nan_to_num(X_fts['X_test'].astype(np.float32), posinf=up_bound, neginf=low_bound)
            else:
                X_train = X_train_cleaned
                X_test = X_test_cleaned
        
            eva.fit(X_train, y_train)
            res = eva.get_results(
                eva.transform(X_test, y_test), y_test
            )
            df_results = pd.concat((df_results, res), axis=0)

    return pd.concat(
        (df_infos.reset_index(drop=True), 
        df_results.reset_index(drop=True)), 
        axis=1)


#################################################
# Others functions including libraries stacking #
#################################################

def lib_stacking(
    datasets,
    libraries,
    classifier,
    clf_kwargs={'random_state':4},
    stack_order = 'Accuracy',
    test_all = True, 
    FILEPATH = "exp/tests_lib/tabs/fts/res_rf_100trees_1k_fullbakeoff.csv",
    DATAPATH = "scripts/UCRArchive_2018",
    scale=False,
    verbose=True,
    other_files=None,
    features_selection=False
):

    os.chdir("/Users/orange/Documents/master")
    df = pd.read_csv(FILEPATH)
    for path in other_files:
        add_df = pd.read_csv(path)
        df = pd.concat((df, add_df), axis=0)

    col = ["Name", "Strategy", "Num class", "Prior most freq class"]
    df_infos = pd.DataFrame(columns=col)
    df_results = pd.DataFrame(
        columns=["Accuracy", "Balanced accuracy", "AUC", "f1", "NLL", "Run time (extr)", 
        "Run time (tsf)", "Num features", "Time/feature", "Run time (classif)", "Total time"]
    )
    if features_selection:
        df_results["Used Features"] = 0
    
    if stack_order not in ["random"]:
        metric = stack_order
    else:
        metric='Accuracy'
    names = list(df.loc[:,["Strategy", metric]].groupby("Strategy")[metric].apply(list).index)
    scores = list(df.loc[:,["Strategy", metric]].groupby("Strategy")[metric].apply(list))
    rankings = []

    # get libraries ranked by specify metric 
    for i in range(len(scores[0])):
        row = [col[i] for col in scores]
        row_sort = sorted(row, reverse=True)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v)-1)/2. for v in row])
    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(len(scores))]

    prov_df = pd.DataFrame({'Strategy':names, 'rankings_avg':rankings_avg })
    prov_df.sort_values(by='rankings_avg', ascending=True, inplace=True)
    ordered_strats = prov_df["Strategy"]

    # stack in random order
    if stack_order=='random':
        random.shuffle(ordered_strats)

    strats = [lib for lib in ordered_strats if lib in libraries] # select only user provided libraries

    pbar_data = tqdm(datasets) if verbose else datasets
    for idx, dataset in enumerate(pbar_data):
        pbar_data.set_description("Processing %s" % dataset) if verbose else -1

        selected_strats = []
        pbar_lib = tqdm(strats, leave=False) if verbose else strats
        for strat in pbar_lib:
            selected_strats.append(strat)

            if test_all or (len(selected_strats)==len(libraries)):
                pbar_lib.set_description("Testing %s" % ' + '.join(selected_strats)) if verbose else -1

                _, _, Y_train, Y_test = load_ts_dataset(dataset, DATAPATH)
                lb = LabelEncoder()
                y_train = lb.fit_transform(Y_train)
                y_test = lb.transform(Y_test)

                uniq_labels, counts = np.unique(y_train, return_counts=True)
                gen_infos = [dataset, ' + '.join(selected_strats),
                            len(uniq_labels), np.max(counts)/len(y_train)]
                gen_infos = pd.DataFrame(gen_infos).T
                gen_infos.columns = col
                df_infos = pd.concat(
                    (df_infos, gen_infos), axis=0
                )

            X_fts = load_features_matrix(data_path, dataset, strat)

            up_bound = np.finfo(np.float32).max 
            low_bound = np.finfo(np.float32).min 

            if len(selected_strats)==1:
                # if only one selected library
                X_train = np.nan_to_num(X_fts['X_train'].astype(np.float32), posinf=up_bound, neginf=low_bound)
                X_test = np.nan_to_num(X_fts['X_test'].astype(np.float32), posinf=up_bound, neginf=low_bound) 
            else: 
                # concatenate feature matrices
                X_train = np.concatenate((X_train, np.nan_to_num(X_fts['X_train'].astype(np.float32), posinf=up_bound, neginf=low_bound)), axis=1)
                X_test = np.concatenate((X_test, np.nan_to_num(X_fts['X_test'].astype(np.float32), posinf=up_bound, neginf=low_bound)), axis=1)
            
            if test_all or (len(selected_strats)==len(libraries)):
            # if test_all is False, 
            # only train the model for the specified libraries' concatenation
                if scale:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                eva = Evaluator(
                    library=strat, 
                    lib_kwargs=get_lib_args(), 
                    clf=classifier,
                    clf_kwargs=clf_kwargs,
                    pre_load=True
                )
                eva.fit(X_train, y_train)
                res = eva.get_results(X_test, y_test)
                df_results = pd.concat((df_results, res), axis=0)
                
                if features_selection:
                    # additional sparisty infos when estimator is LogReg
                    n_true_features = 0
                    for i in range(eva.clf.n_features_in_):
                        if sum(eva.clf.coef_[:,i]) != 0:
                            n_true_features += 1
                    df_results.iloc[idx, -1] = n_true_features

    return pd.concat(
        (df_infos.reset_index(drop=True), 
        df_results.reset_index(drop=True)), 
        axis=1)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, default="data/",
                        help="Path to data, .ts files")
    parser.add_argument("-f", "--savepath", type=str, default="data/",
                        help="Path in which features files are saved / from which are load")
    parser.add_argument("-n", "--max_features", type=int, default=1000, 
                        help="Maximum number of features to extract when infinite feature space")
    parser.add_argument("-e", "--estimator", type=str, default="RandomForestClassifier",
                        help="sklearn compatible estimator to perform desired task")
    parser.add_argument("-kw", "--kwargs", type=json.loads, default='{"random_state":4}',
                        help="Keywords arguments for your estimator")
    parser.add_argument("-t", "--task", type=str, default='classification',
                        help="Task to perform : classification or regression")
    parser.add_argument("-s", "--save", type=str, default=False,
                        help="Whether to save features matrices or not in data dir, if so specify format (npz or csv)")
    parser.add_argument("-p", "--preload", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether you want to preload features from data path")
    parser.add_argument("-sc", "--scale", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to apply standard scaling to features matrix before estimator")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Level of displayed information from 0 to 2")
    parser.add_argument("-j", "--n_jobs", type=int, default=1,
                        help="Number of jobs tu run in parrallel for compatible libraries")
    parser.add_argument("-np", "--non_python", type=argparse.BooleanOptionalAction, default=False,
                        help="Whether to add non-python libraries")
    parser.add_argument("-stck", "--stacking", type=argparse.BooleanOptionalAction, default=False,
                        help="Whether to laucnh libraries stacking or not")
    parser.add_argument("-m", "--mode", type=str, default='Features',
                        help="Which approach from the paper you want to reproduce")
    parser.add_argument("-all", "--all_lib", type=argparse.BooleanOptionalAction, default=False,
                        help="Test one by one stacking or just test the all the provided lib")
    args = parser.parse_args()

    if args.verbosity < 2:
        # max level of verbosity (2) displays all convergence/performance/deprecation warnings
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)
        warnings.simplefilter(action='ignore', category=NumbaPendingDeprecationWarning)
        np.seterr(all='ignore')

    # mid verbosity level (1) displays progress bars
    # min level (0) shows nothing 
    verb = True if args.verbosity > 0 else False 
    
    # load correct datasets' names ensemble according to selected task
    datasets = get_original_UCR_datasets() + get_UCR_2018() if args.task=='classification' else \
        get_regression_datasets()

    # go to home directory
    os.chdir(os.path.expanduser("~"))

    if args.non_python:
        add_list = ['hctsa', 'feasts']
    else:
        add_list = []

    if not args.stacking:
        res = run_tests(
            datasets=datasets[1:4],
            libraries=[
                'rocket',
                'minirocket',
                'multirocket',
                'tsfresh',
                'tsfel',
                'catch22',
                'tsfeatures',
                'featuretools',
                'signature',
                'intervals',
                'intervals_c22'
            ] + add_list,
            max_features=args.max_features,
            classifier=eval(args.estimator),
            clf_kwargs=args.kwargs,
            DATAPATH=args.datapath,
            SAVEPATH=args.savepath,
            scale=args.scale,
            to_save=args.save,
            to_load=args.preload,
            raw_data=None,
            task=args.task,
            verbose=verb,
            n_jobs=args.n_jobs
        )
        # create some results folder
        if not os.path.isdir(os.path.join(args.savepath, "results")): 
            os.mkdir(os.path.join(args.savepath, "results"))

        filename = f"results/{args.task}_{args.estimator}.csv"
        count_id = 0
        # no results file overwrite
        while os.path.isfile(os.path.join(args.savepath, filename)):
            count_id += 1
            filename = f"results/{args.task}_{args.estimator}_{count_id}.csv"
        
        res.to_csv(os.path.join(args.savepath, filename), index=False)

    else:
        # whether you may have other results in some other files 
        files = ["exp/tests_lib/tabs/fts/feasts/res_rf_100trees_fullbakeoff_feasts.csv",
                "exp/tests_lib/tabs/fts/hctsa/res_rf_100trees_fullbakeoff_hctsa.csv"]
        
        libraries=[
            'minirocket',
            'tsfresh',
            'tsfel',
            'tsfeatures',
            'catch22',
            'intervals',
            'intervals_c22',
            'featuretools',
            'signature',
            'hctsa',
            'feasts'
        ]

        if args.mode == 'Features_noROCKET':
            libraries -= ['minirocket']
        elif args.mode == 'Features_python':
            libraries -= ['hctsa', 'feasts']

        res = lib_stacking(     
            datasets=datasets,
            libraries=libraries,
            classifier=RotationForest,
            clf_kwargs={'random_state': 4},
            stack_order='Accuracy',
            test_all=args.all_lib, 
            scale=True,
            FILEPATH=args.savepath,
            DATAPATH=args.datapath,
            other_files=None,
            verbose=verb,
            features_selection=False
        )
        # create some results folder
        if not os.path.isdir(os.path.join(args.savepath, "results")): 
            os.mkdir(os.path.join(args.savepath, "results"))

        filename = f"results/{args.mode}.csv"
        count_id = 0
        # no results file overwrite
        while os.path.isfile(os.path.join(args.savepath, filename)):
            count_id += 1
            filename = f"results/{args.mode}_{count_id}.csv"
        
        res.to_csv(os.path.join(args.savepath, filename), index=False)