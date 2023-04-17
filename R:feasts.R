library(feasts)
library(fracdiff)
library(tseries)
library(Gmisc)
library(dplyr)
library(optparse)

UCR_classif <- list(
  "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration",
  "CinCECGtorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ", "DiatomSizeReduction",
  "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes",
  "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords",
  "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate", 
  "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7",
  "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect",
  "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil",
  "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup",
  "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim",
  "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
  "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2",
  "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
  "UWaveGestureLibraryAll", "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga")

UCR_classif_2018 <- list(
  "ACSF1", "BME", "Chinatown", "Crop", "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel",
  "FreezerRegularTrain","FreezerSmallTrain", "GunPointAgeSpan", "GunPointMaleVersusFemale",
  "GunPointOldVersusYoung", "HouseTwenty", "InsectEPGRegularTrain", "InsectEPGSmallTrain",
  "MixedShapesRegularTrain", "MixedShapesSmallTrain", "PigAirwayPressure", "PigArtPressure",
  "PigCVP", "PowerCons", "Rock", "SemgHandGenderCh2","SemgHandMovementCh2", "SemgHandSubjectCh2", "UMD")

option_list = list(
  make_option(c("-d", "--datapath"), action="store", type='character', default='data/',
              help="Path from which datasets can be found"),
  make_option(c("-s", "--savepath"), action="store", type='character', default='data/',
              help="Path in which features .csv files are stored"),
  make_option(c("-v", "--verbose"), action="store_true", default=TRUE,
              help="Whether to display datasets names while processing")
)
opt = parse_args(OptionParser(option_list=option_list))

feasts_extract <- function(ts){
  fts <- feat_acf(ts)
  fts <- append(fts, feat_pacf(ts))
  fts <- append(fts, ljung_box(ts))
  fts <- append(fts, box_pierce(ts))
  fts <- append(fts, var_tiled_mean(ts))
  fts <- append(fts, var_tiled_var(ts))
  fts <- append(fts, shift_kl_max(ts))
  fts <- append(fts, shift_level_max(ts))
  fts <- append(fts, shift_var_max(ts))
  fts <- append(fts, feat_stl(ts, .period = 1))
  fts <- append(fts, feat_spectral(ts))
  fts <- append(fts, stat_arch_lm(ts))
  fts <- append(fts, n_crossing_points(ts))
  fts <- append(fts, longest_flat_spot(ts))
  fts <- append(fts, coef_hurst(ts))
  fts <- append(fts, guerrero(ts))
  
  return(fts)
  } 

transform_dataset <- function(datapath, savepath, dataset_name, split){
  file <- paste0(dataset_name, "_", split, ".txt")

  df <- read.table(pathJoin(datapath, dataset_name, file), header=FALSE)

  samp <- df %>% slice(1:1)
  ts <- as.numeric(samp)[2:ncol(df)]
  fts <- data.frame(as.list(feasts_extract(ts)), row.names=NULL) 
  
  for(idx in 2:nrow(df)){
    samp <- df %>% slice(idx:idx)
    ts <- as.numeric(samp)[2:ncol(df)]
    
    fts_ts <- data.frame(as.list(feasts_extract(ts)), row.names=NULL) 
    fts <- rbind(fts, fts_ts)
  }
  
  if(!dir.exists(pathJoin(savepath, dataset_name))) {
    dir.create(pathJoin(savepath, dataset_name))
  }
  
  write.csv(fts, pathJoin(savepath, dataset_name, paste0("feasts_", tolower(split), ".csv")), row.names=TRUE)
}

run_benchmark <- function(benchmark, datapath, savepath){
  extract_times_train <- list()
  extract_times_test <- list()
  
  for(i in 1:length(benchmark)){
    data_name <- as.character(benchmark[i])
    
    if(opt$verbose) {
      print(paste0("Processing ", data_name, "..."))
    }
    
    x_time <- system.time(transform_dataset(datapath, savepath, data_name, "TRAIN"))
    extract_times_train <- append(extract_times_train, as.numeric(x_time[3]))
    
    y_time <- system.time(transform_dataset(datapath, savepath, data_name, "TEST"))
    extract_times_test <- append(extract_times_test, as.numeric(y_time[3]))
  }
  
  write.csv(as.numeric(extract_times_train), pathJoin(savepath, "extract_times_train.csv"), row.names=benchmark)
  write.csv(as.numeric(extract_times_test), pathJoin(savepath, "extract_times_test.csv"), row.names=benchmark)
}

run_benchmark(UCR_classif+UCR_classif_2018, opt$datapath, opt$savepath)

