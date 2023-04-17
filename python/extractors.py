from tsfeatures import tsfeatures
import numpy as np
import pandas as pd
from sktime.transformations.panel import (
    tsfresh, rocket, 
    shapelet_transform, 
    random_intervals, catch22, signature_based)
from sktime.datatypes._panel._check import is_nested_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from utils import *

import featuretools as ft
from feature_tools import *
import tsfel


class TSFRESH():

    def __init__(
        self, 
        fts_to_extract="comprehensive",
        verbose=False, 
        n_jobs=1
    ):
        self.fts_to_extract = fts_to_extract
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.ext = None

    def fit_extract(self, X, y=None):
        assert is_nested_dataframe(X)
        # for multivariate, extract each feature along each dim
        self.ext = tsfresh.TSFreshFeatureExtractor(default_fc_parameters=self.fts_to_extract, n_jobs=self.n_jobs,
                                                    disable_progressbar=not self.verbose, show_warnings=False)
        return self.ext.fit_transform(X.reset_index(drop=True))
    
    def transform(self, X, y=None):
        return self.ext.transform(X.reset_index(drop=True))
    
class TSFEL():

    def __init__(
        self, 
        verbose=0,
        n_jobs=1
    ):
        self.verbose = verbose
        self.n_jobs = n_jobs
    
    def fit_extract(self, X, y=None):
        cfg_file = tsfel.get_features_by_domain()
        window = X.iloc[0][0].shape[0]
        X_tsf = convert_nested(X).fillna(0)

        # for multivariate, extract each feature along each dim
        return tsfel.time_series_features_extractor(cfg_file, X_tsf, fs=100, n_jobs=self.n_jobs,
                                                    window_size=window, verbose=self.verbose)
    
    def transform(self, X, y=None):
        return self.fit_extract(X)

class ROCKET():

    def __init__(
        self,
        n_kernels=10000,
        random_state=None,
        n_jobs=1
    ):
        self.n_kernels = n_kernels
        self.random_state = random_state
        self.ext = None
        self.n_jobs = n_jobs
    
    def fit_extract(self, X, y=None):
        assert is_nested_dataframe(X)
        # for multivariate, extract the specified number of kernels,
        # each being convolved with randomly chosen dim 
        self.ext = rocket.Rocket(num_kernels=self.n_kernels, random_state=self.random_state, n_jobs=self.n_jobs)
        return self.ext.fit_transform(X)
    
    def transform(self, X, y=None):
        return self.ext.transform(X)

class MINIROCKET():

    def __init__(
        self,
        n_kernels=10000,
        max_dilations=32,
        random_state=None,
        n_jobs=1
    ):
        self.n_kernels = n_kernels
        self.max_dilations = max_dilations
        self.random_state = random_state
        self.ext = None
        self.n_jobs = n_jobs
    
    def fit_extract(self, X, y=None):
        assert is_nested_dataframe(X)
        if X.shape[1] == 1:
            self.ext = rocket.MiniRocket(num_kernels=self.n_kernels,
                                        max_dilations_per_kernel=self.max_dilations, 
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)
        else:
            # for multivariate, extract the specified number of kernels,
            # each being convolved with randomly chosen dim 
            self.ext = rocket.MiniRocketMultivariate(num_kernels=self.n_kernels, 
                                                    max_dilations_per_kernel=self.max_dilations,
                                                    random_state=self.random_state,
                                                    n_jobs=self.n_jobs)
        
        return self.ext.fit_transform(X) 
    
    def transform(self, X, y=None):
        return self.ext.transform(X)

class MULTIROCKET():

    def __init__(
        self,
        n_kernels=5000,
        n_features_per_kernel=4,
        max_dilations=32, 
        random_state=None,
        n_jobs=1
    ):
        self.n_kernels = n_kernels
        self.n_features_per_kernel = n_features_per_kernel
        self.max_dilations = max_dilations
        self.random_state = random_state
        self.ext = None
        self.n_jobs = n_jobs
    
    def fit_extract(self, X, y=None):
        assert is_nested_dataframe(X)
        if X.shape[1] == 1:
            self.ext = rocket.MultiRocket(num_kernels=self.n_kernels,
                                        n_features_per_kernel=self.n_features_per_kernel,
                                        max_dilations_per_kernel=self.max_dilations,  
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)
        else:
            # for multivariate, extract the specified number of kernels,
            # each being convolved with randomly chosen dim 
            self.ext = rocket.MultiRocketMultivariate(num_kernels=self.n_kernels,
                                                    n_features_per_kernel=self.n_features_per_kernel, 
                                                    max_dilations_per_kernel=self.max_dilations, 
                                                    random_state=self.random_state,
                                                    n_jobs=self.n_jobs)
        
        return self.ext.fit_transform(X)
    
    def transform(self, X, y=None):
        return self.ext.transform(X)

class CATCH22():

    def __init__(
        self,
        outlier_norm=False,
        replace_nans=True, 
        n_jobs=1
    ):
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.n_jobs = n_jobs
        self.ext = None

    def fit_extract(self, X, y=None):
        # for multivariate, extract each feature along each dim
        self.ext = catch22.Catch22(outlier_norm=self.outlier_norm,
                                    replace_nans=self.replace_nans, 
                                    n_jobs=self.n_jobs)
        return self.ext.fit_transform(X)

    def transform(self, X, y=None):
        return self.ext.transform(X, y)

class INTERVALS():

    def __init__(
        self, 
        transformers=None, 
        n_intervals=30,
        random_state=None,
        n_jobs=1
    ):
        self.transformers = transformers 
        self.n_intervals = n_intervals
        self.random_state = random_state
        self.ext = None
        self.n_jobs = n_jobs

    def fit_extract(self, X, y=None):
        # for multivariate, extract the number of features matching
        # with the number of specified intervasls
        # each being taken on randomly chosen dim
        self.ext = random_intervals.RandomIntervals(
            transformers=self.transformers,
            n_intervals=self.n_intervals,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        return self.ext.fit_transform(X)
    
    def transform(self, X, y=None):
        return self.ext.transform(X)

class FEATURETOOLS():

    def __init__(
        self,
        agg_primitives=['mean', 'max', 'std'],
        tsf_primitives=[Derivative(1), Integral(1)],
        tsf_primitives_all=[],
        max_depth=None,
        max_features=1000,
        n_jobs=1
    ):
        self.max_features = max_features
        self.agg_primitives = agg_primitives
        self.tsf_primitives = tsf_primitives
        self.tsf_primitives_all = tsf_primitives_all
        self.max_depth = max_depth
        self.n_jobs = n_jobs
    
    def build_entity_set(self, X, y):
        ts_length = X.iloc[0][0].shape[0]
        X_tsf = convert_nested(X)

        root_table = pd.DataFrame()
        root_table.insert(loc=0, column='class', value=y)

        val_table = pd.DataFrame()
        for dim in X.columns:
            val_table.insert(loc=0, column=dim, value=X_tsf[dim])
        ts_ids = np.arange(0, len(y))
        val_table.insert(loc=0, column='ts_id', value=np.repeat(ts_ids, ts_length))

        es = ft.EntitySet(id='time_series')
        es = es.add_dataframe(
            dataframe_name='root', 
            dataframe=root_table, 
            make_index=True,
            index='ts_id'
        )
        es = es.add_dataframe(
            dataframe_name='values', 
            dataframe=val_table, 
            make_index=True,
            index='val_id'
        )
        es = es.add_relationship('root', 'ts_id', 'values', 'ts_id')

        return es
    
    def fit_extract(self, X, y):

        # for multivariate, extract each feature along each dim
        feature_matrix, _ = ft.dfs(
        entityset=self.build_entity_set(X, y), 
        target_dataframe_name='root',
        agg_primitives=self.agg_primitives,
        trans_primitives = self.tsf_primitives_all,
        groupby_trans_primitives = self.tsf_primitives,
        max_depth=self.max_depth, 
        ignore_columns={'root': ['class']},
        verbose=False,
        max_features=self.max_features,
        n_jobs=self.n_jobs
        )

        return feature_matrix
    
    def transform(self, X, y):
        return self.fit_extract(X, y)

class TSFEATURES():

    def __init__(
        self,
        freq=1,
        threads=1
    ):
        self.freq = freq
        self.threads = threads
    
    def fit_extract(self, X, y=None):
        fts = pd.DataFrame()
        X_tsf = convert_nested(X, index=True)
        # for multivariate, extract each feature along each dim
        # iterate over each dim to support multivariate inputs
        for dim in X_tsf.columns:
            if dim != 'unique_id':
                X_tsf_dim = X_tsf[['unique_id', dim]].rename(columns={dim:'y'})
                fts_dim = tsfeatures(X_tsf_dim, freq=self.freq, threads=self.threads).set_index('unique_id')
                fts_dim = fts_dim.add_suffix("_"+str(dim))
                fts = pd.concat((fts, fts_dim), axis=1)
        return fts.astype(np.float32)

    def transform(self, X, y=None):
        return self.fit_extract(X, y)

class SIGNATURE():

    def __init__(
        self,
        window_name='dyadic',
        window_depth=3,
        window_length=None, 
        window_step=None,
        sig_tfm='signature',
        depth=4
    ):

        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.ext = None

    def fit_extract(self, X, y=None):
        assert is_nested_dataframe(X)
        self.ext = signature_based.SignatureTransformer(window_name=self.window_name,
                                                        window_depth=self.window_depth,
                                                        window_length=self.window_length,
                                                        window_step=self.window_step,
                                                        sig_tfm=self.sig_tfm,
                                                        depth=self.depth)
        return self.ext.fit_transform(X)
    
    def transform(self, X, y=None):
        return self.ext.transform(X)

# Supervised method : not used 
class SHAPELET():

    def __init__(
        self, 
        n_shapelets=100,
        max_shapelets=50,
        batch_size=64,
        time_limit=0,
        random_state=None,
        n_jobs=1
    ):
        self.n_shapelets = n_shapelets
        self.max_shapelets = max_shapelets
        self.batch_size = batch_size
        self.time_limit = time_limit
        self.random_state = random_state
        self.ext = None
        self.n_jobs = n_jobs

    def fit_extract(self, X, y):
        self.ext = shapelet_transform.RandomShapeletTransform(
            n_shapelet_samples=self.n_shapelets,
            max_shapelets=self.max_shapelets,
            batch_size=self.batch_size,
            time_limit_in_minutes=self.time_limit,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        return self.ext.fit_transform(X, y)
    
    def transform(self, X, y):
        return self.ext.transform(X, y)
