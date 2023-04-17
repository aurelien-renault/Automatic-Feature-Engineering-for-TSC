import os
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from multiprocessing import cpu_count
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy, from_3d_numpy_to_nested, from_nested_to_2d_array
from sktime.transformations.series.summarize import SummaryTransformer

from feature_tools import AutoCorr, Derivative, FourrierTransform, Integral
from data_loader_regr import load_from_tsfile_to_dataframe_regr


def load_ts_dataset(dataset_name, PATH="scripts/UCRArchive_2018", classif=True):
    train = dataset_name+'_TRAIN.ts'
    test = dataset_name+'_TEST.ts'
    
    if classif:
        X_train, y_train = load_from_tsfile_to_dataframe(os.path.join(PATH, dataset_name, train))
        X_test, y_test = load_from_tsfile_to_dataframe(os.path.join(PATH, dataset_name, test))
    else:
        X_train, y_train = load_from_tsfile_to_dataframe_regr(os.path.join(PATH, dataset_name, train))
        X_test, y_test = load_from_tsfile_to_dataframe_regr(os.path.join(PATH, dataset_name, test))
    
    return X_train, X_test, y_train, y_test

def save_features_matrix(PATH, dataset_name, lib_name, to_csv=True, **files):
    # create folder for specific dataset if not existing 
    os.makedirs(os.path.join(PATH, dataset_name), exist_ok=True)
    name = lib_name+"_FEATURES"
    if to_csv:
        for split in files.keys():
            files[split].to_csv(os.path.join(PATH, dataset_name, f"{split}_{name}.csv"))
    else:
        np.savez(os.path.join(PATH, dataset_name, name), **files)

def load_features_matrix(PATH, dataset_name, lib_name, is_csv=False):
    try:
        name = lib_name+"_FEATURES.npz"
        return np.load(os.path.join(PATH, dataset_name, name), allow_pickle=True)
    except FileNotFoundError:
        name = lib_name+"_FEATURES.csv"
        X_train = pd.read_csv(os.path.join(PATH, dataset_name, f"X_train_{name}"), index_col=0)
        X_test = pd.read_csv(os.path.join(PATH, dataset_name, f"X_test_{name}"), index_col=0)
        return {'X_train':X_train, 'X_test':X_test}
        
    
def convert_nested(df_nested, index=False, to_numpy=False):
    X_numpy = from_nested_to_3d_numpy(df_nested)
    n_col = X_numpy.shape[1]
    length = X_numpy.shape[2]
    X_tsf = pd.DataFrame(
        X_numpy.transpose((0, 2, 1)).reshape((-1, n_col)),
        columns=df_nested.columns
    )
    if index:
        X_tsf['unique_id'] = np.repeat(np.arange(0, X_numpy.shape[0], 1), length)
    return X_numpy if to_numpy else X_tsf

def convert_to_nested(X, value_idx = 2, time_col='timestamp'):
    n_timesteps = len(X[time_col].unique())
    n_col = len(X.columns[value_idx:])
    X_tmp = X.iloc[:,value_idx:].values.reshape((-1, n_timesteps, n_col)).transpose((0,2,1))
    return from_3d_numpy_to_nested(X_tmp, column_names=X.columns[value_idx:]) 

def deal_na_values(X, y, mode='drop'):
     # X is 3d numpy with shape (n_sample, n_dim, ts_length)
    n_sample, n_dim, _ = X.shape

    nan_idx = set()
    for dim in range(n_dim):
        for i, serie in enumerate(X[:,dim,:]):
            if np.sum(np.isnan(serie)):
                nan_idx.add(i)
    
    #if len(nan_idx)/n_sample > 0.1 and mode=='drop':
    #    mode = 'median'
    
    if mode=='drop':
        X_clean = X[[idx for idx in range(n_sample) if idx not in nan_idx]]
        y_clean = y[[idx for idx in range(n_sample) if idx not in nan_idx]]
    else:
        tmp = X[list(nan_idx)]
        for dim in range(n_dim):
            for i, serie in enumerate(X[:,dim,:]):
                nan_val = mode if isinstance(mode, int) or isinstance(mode, float) else np.nanmedian(serie)
                tmp[i,dim,:] = np.nan_to_num(serie, nan=nan_val)
        # replace series with nan in final tab
        X[list(nan_idx)] = tmp

    return X_clean, y_clean

# default libraries' arguments 
def get_lib_args(max_fts=1000, n_cpu=1):

    if n_cpu==-1:
        n_cpu = cpu_count - 1

    default_args = {
        'tsfresh' : {
            'fts_to_extract':"comprehensive",
            'verbose':False,
            'n_jobs':n_cpu
        },
        'tsfel' : {
            'verbose':0,
            'n_jobs':n_cpu
        },
        'rocket' : {
            'n_kernels':max_fts//2,
            'random_state':4,
            'n_jobs':n_cpu
        },
        'minirocket' : {
            'n_kernels':int(max_fts),
            'random_state':4,
            'n_jobs':n_cpu
        },
        'multirocket' : {
            'n_kernels':max_fts//8,
            'n_features_per_kernel':6,
            'random_state':4,
            'n_jobs':n_cpu
        },
        'catch22':{
            'replace_nans':True,
            'n_jobs':n_cpu
        },
        'shapelet' : {
            'n_shapelets':max_fts,
            'max_shapelets':max_fts,
            'batch_size':256,
            'time_limit':0,
            'n_jobs':n_cpu
        },
        'intervals' : {
            'transformers':SummaryTransformer(
                summary_function=('mean', 'std', 'min', 'max', 'count', 'sum', 'skew', 'median'),
                quantiles=(0.25, 0.75)), 
            'n_intervals':max_fts//10,
            'random_state':4,
            'n_jobs':n_cpu
        },
        'featuretools' : {
            'agg_primitives':[
                'sum', 'mean', 'std', 'min', 'max', 
                'count', 'skew', 'median'],
            'tsf_primitives':[Derivative(1), Derivative(2), Integral(1), 
                                Integral(2), AutoCorr, FourrierTransform],
            'tsf_primitives_all':[],
            'max_depth':None,
            'max_features':max_fts,
            'n_jobs':n_cpu
        },
        'feasts' : { 
        },
        'hctsa' : {  
        },
        'tsfeatures' : {
            'freq':1,
            'threads':n_cpu
        },
        'signature' : {
            'window_name':'dyadic',
            'window_depth':2,
            'window_length':None,
            'window_step':None,
            'sig_tfm':'signature',
            'depth':2
        }
    }
    return default_args

def get_list_missing_and_vary():
    return [
        'ALlGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'DodgerLoopDay',
        'DodgerLoopGame',
        'DodgerLoopWeekend',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GestureMidAirZ1',
        'GestureMidAirZ2',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PLAID',
        'ShakeGestureWiimoteZ'
    ]

def get_original_UCR_datasets():
    return [
        "Adiac",
	    "ArrowHead",
	    "Beef",
	    "BeetleFly",
	    "BirdChicken",
	    "Car",
	    "CBF",
	    "ChlorineConcentration",
	    "CinCECGtorso",
	    "Coffee",
	    "Computers",
	    "CricketX",
	    "CricketY",
	    "CricketZ",
	    "DiatomSizeReduction",
	    "DistalPhalanxOutlineAgeGroup",
	    "DistalPhalanxOutlineCorrect",
	    "DistalPhalanxTW",
	    "Earthquakes",
	    "ECG200",
	    "ECG5000",
	    "ECGFiveDays",
	    "ElectricDevices",
	    "FaceAll",
	    "FaceFour",
	    "FacesUCR",
	    "FiftyWords",
	    "Fish",
	    "FordA",
	    "FordB",
	    "GunPoint",
	    "Ham",
	    "HandOutlines",
	    "Haptics",
	    "Herring",
	    "InlineSkate",
	    "InsectWingbeatSound",
	    "ItalyPowerDemand",
	    "LargeKitchenAppliances",
	    "Lightning2",
	    "Lightning7",
	    "Mallat",
	    "Meat",
	    "MedicalImages",
	    "MiddlePhalanxOutlineAgeGroup",
	    "MiddlePhalanxOutlineCorrect",
	    "MiddlePhalanxTW",
	    "MoteStrain",
	    "NonInvasiveFetalECGThorax1",
	    "NonInvasiveFetalECGThorax2",
	    "OliveOil",
	    "OSULeaf",
	    "PhalangesOutlinesCorrect",
	    "Phoneme",
	    "Plane",
	    "ProximalPhalanxOutlineAgeGroup",
	    "ProximalPhalanxOutlineCorrect",
	    "ProximalPhalanxTW",
	    "RefrigerationDevices",
	    "ScreenType",
	    "ShapeletSim",
	    "ShapesAll",
	    "SmallKitchenAppliances",
	    "SonyAIBORobotSurface1",
	    "SonyAIBORobotSurface2",
	    "StarLightCurves",
	    "Strawberry",
	    "SwedishLeaf",
	    "Symbols",
	    "SyntheticControl",
	    "ToeSegmentation1",
	    "ToeSegmentation2",
	    "Trace",
	    "TwoLeadECG",
	    "TwoPatterns",
	    "UWaveGestureLibraryX",
	    "UWaveGestureLibraryY",
	    "UWaveGestureLibraryZ",
	    "UWaveGestureLibraryAll",
	    "Wafer",
	    "Wine",
	    "WordSynonyms",
	    "Worms",
	    "WormsTwoClass",
	    "Yoga"
    ]

def get_UCR_2018():
    return [
        "ACSF1",
        "BME",
        "Chinatown",
        "Crop",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        #"Fungi", 1 instance of each class
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",
        "HouseTwenty",
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "PowerCons",
        "Rock",
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "SmoothSubspace",
        "UMD"
    ]

def get_UEA_multi():
    return [
        "ArticularyWordRecognition",
        "AtrialFibrillation",
        "BasicMotions",
        "Cricket",
        "DuckDuckGeese",
        #"EigenWorms", kill RAM
        "Epilepsy",
        "EthanolConcentration",
        "ERing",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        #"PenDigits", need padding rocket
        "PEMS-SF",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "StandWalkJump",
        "UWaveGestureLibrary"
    ]

def get_regression_datasets():
    return ["AustraliaRainfall", 
            "HouseholdPowerConsumption1",
            "HouseholdPowerConsumption2",
            "BeijingPM25Quality",
            "BeijingPM10Quality",
            "Covid3Month",
            "LiveFuelMoistureContent",
            "FloodModeling1",
            "FloodModeling2",
            "FloodModeling3",
            "AppliancesEnergy",
            "BenzeneConcentration",
            "NewsHeadlineSentiment",
            "NewsTitleSentiment",
            "BIDMC32RR",
            "BIDMC32HR",
            "BIDMC32SpO2",
            "IEEEPPG",
            "PPGDalia"]

def convert_csv_to_npz(PATH_from, PATH_to, name):

    train = pd.read_csv(os.path.join(PATH_from, name, "features_train.csv"), index_col=0, delimiter=',')
    test = pd.read_csv(os.path.join(PATH_from, name, "features_test.csv"), index_col=0, delimiter=',')

    files = {'X_train':train.to_numpy(), 'X_test':test.to_numpy()}
    np.savez(os.path.join(PATH_to, name, "feasts_FEATURES"), **files)


if __name__ == '__main__':
    pass
    data_bakeoff = get_UCR_2018()
    os.chdir("/Users/orange/Documents/master")
    for name in data_bakeoff:
        convert_csv_to_npz("datasets/Univariate_arff", "scripts/UCRArchive_2018", name)
