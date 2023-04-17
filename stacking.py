from evaluator import *

def lib_stacking(
    datasets,
    libraries,
    classifier,
    clf_kwargs={'random_state':4},
    stack_order = 'Accuracy',
    test_all=False, 
    FILEPATH="exp/tests_lib/tabs/fts/res_rf_100trees_1k_fullbakeoff.csv",
    DATAPATH="scripts/UCRArchive_2018",
    FEATURESPATH="scripts/UCRArchive_2018",
    scale=False,
    task='classification',
    verbose=True,
    features_selection=False
):

    classif = True if task=='classification' else False 

    df = pd.DataFrame()
    if isinstance(FILEPATH, list):
        for path in FILEPATH:
            add_df = pd.read_csv(path)
            df = pd.concat((df, add_df), axis=0)
    else:
        df = pd.read_csv(FILEPATH)

    if classif:
        col = ["Name", "Strategy", "Num class", "Prior most freq class"]
    else:
        col = ["Name", "Strategy"]
    df_infos = pd.DataFrame(columns=col)

    df_results = pd.DataFrame(
        columns=["Accuracy", "Balanced accuracy", "AUC", "f1", "NLL", "Run time (extr)",
        "Run time (tsf)", "Num features", "Time/feature", "Run time (classif)", "Total time"]
    ) if classif else \
    pd.DataFrame(
        columns=["RMSE", "MAE", "Run time (extr)", "Run time (tsf)", "Num features",
        "Time/feature", "Run time (classif)", "Total time"])

    if features_selection:
        df_results["Used Features"] = 0
    
    if stack_order not in ["random"]:
        metric = stack_order
    else:
        metric='Accuracy'

    names = list(df.loc[:,["Strategy", metric]].groupby("Strategy")[metric].apply(list).index)
    scores = list(df.loc[:,["Strategy", metric]].groupby("Strategy")[metric].apply(list))
    rankings = []

    # get libraries ranked by specific metric 
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

                _, _, Y_train, Y_test = load_ts_dataset(dataset, DATAPATH, classif=classif)
                if classif: # encode classification labels
                    lb = LabelEncoder()
                    y_train = lb.fit_transform(Y_train)
                    y_test = lb.transform(Y_test)
                else:
                    y_train = Y_train.astype(np.float32)
                    y_test = Y_test.astype(np.float32)

                uniq_labels, counts = np.unique(y_train, return_counts=True)
                gen_infos = [dataset, ' + '.join(selected_strats), len(uniq_labels), 
                                np.max(counts)/len(y_train)] if classif else \
                            [dataset, ' + '.join(selected_strats)]
                gen_infos = pd.DataFrame(gen_infos).T
                gen_infos.columns = col
                df_infos = pd.concat(
                    (df_infos, gen_infos), axis=0
                )

            X_fts = load_features_matrix(FEATURESPATH, dataset, strat)

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
                    task=task,
                    pre_load=True
                )
                eva.fit(X_train, y_train)
                res = eva.get_results(X_test, y_test)
                df_results = pd.concat((df_results, res), axis=0)
                
                if features_selection:
                    # additional sparsity infos when estimator is LogReg
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
    parser.add_argument("-f", "--featurespath", type=str, default="data/",
                        help="Path from which features are load")
    parser.add_argument("-r", "--resultspath", type=str, default="results/classification_RandomForestClassifier.csv",
                        help="Path to file from wich to retreive ranks for stacking if -o isn't random")
    parser.add_argument("-o", "--order", type=str, default='Accuracy',
                        help="stacking order, can be whether a metric (name of some results Dataframe) or 'random'")
    parser.add_argument("-a", "--all", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to apply classifier behind each combination or only the final one")
    parser.add_argument("-e", "--estimator", type=str, default="RandomForestClassifier",
                        help="sklearn compatible estimator to perform desired task")
    parser.add_argument("-kw", "--kwargs", type=json.loads, default='{"random_state":4}',
                        help="Keywords arguments for your estimator")
    parser.add_argument("-t", "--task", type=str, default='classification',
                        help="Task to perform : classification or regression")
    parser.add_argument("-sc", "--scale", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to apply standard scaling to features matrix before estimator")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Level of displayed information from 0 to 2")
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

    # alternatively if results are located in different csv files, one can pass a list of path as
    # it is the case in the following example and then pass it to FILEPATH: 
    files = ["Documents/master/exp/tests_lib/tabs/fts/res_rf_100trees_1k_fullbakeoff.csv",
             "Documents/master/exp/tests_lib/tabs/fts/feasts/res_rf_100trees_fullbakeoff_feasts.csv",
             "Documents/master/exp/tests_lib/tabs/fts/hctsa/res_rf_100trees_fullbakeoff_hctsa.csv"]

    print(eval(args.estimator))
    res = lib_stacking(     
        datasets=datasets[5:6],
        libraries=[
            'rocket',
            'minirocket',
            'multirocket',
            'tsfresh',
            'tsfel',
            'tsfeatures',
            'catch22',
            'intervals',
            'intervals_c22',
            'featuretools',
            'signature',
            #'hctsa',
            #'feasts'
        ],
        classifier=eval(args.estimator),
        clf_kwargs=args.kwargs,
        stack_order=args.order,
        test_all=args.all,
        FILEPATH=args.resultspath,
        DATAPATH=args.datapath,
        FEATURESPATH=args.featurespath, 
        scale=args.scale,
        verbose=verb,
        task=args.task,
        features_selection=False
    )
    # create some results folder
    if not os.path.isdir(os.path.join(args.featurespath, "results")): 
        os.mkdir(os.path.join(args.featurespath, "results"))

    filename = f"results/stacking_{args.estimator}.csv"
    count_id = 0
    # no results file overwrite
    while os.path.isfile(os.path.join(args.featurespath, filename)):
        count_id += 1
        filename = f"results/stacking_{args.estimator}_{count_id}.csv"
    
    res.to_csv(os.path.join(args.featurespath, filename), index=False) 