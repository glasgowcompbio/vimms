# needs xcms3, see https://bioconductor.org/packages/release/bioc/html/xcms.html for installation
library(xcms)
library(tidyverse)
source('./centWave_new_modified.R')

### some useful functions ###

getDesign = function(mzml_files, label) {
  design <- data.frame(sample_name = sub(basename(mzml_files), pattern = '.mzML', replacement = "", fixed = TRUE),
                       sample_group = c(rep(label, length(mzml_files))),
                       stringsAsFactors = FALSE)
  return(design)  
}

readData = function(files, design, mode="onDisk") {
  raw_data <- MSnbase::readMSData(files = files, pdata = new("NAnnotatedDataFrame", design),
                                  mode = mode, centroided. = TRUE, msLevel. = 1)
  return(raw_data)
}

get_scanrange <- function(scanrange) {
  if (length(scanrange) < 2) {
    scanrange <- c(1, length(object@scantime))
  }
  else {
    scanrange <- range(scanrange)
  }
  if (min(scanrange) < 1 | max(scanrange) > length(object@scantime)) {
    scanrange[1] <- max(1, scanrange[1])
    scanrange[2] <- min(length(object@scantime), scanrange[2])
    message("Provided scanrange was adjusted to ", scanrange)
  }
  scanrange
}


get_regions_of_interest = function(object, ppm = 25, peakwidth = c(20, 50), snthresh = 10, prefilter = c(3, 100), mzCenterFun = "wMean",
                        integrate = 1, mzdiff = -0.001, fitgauss = FALSE, noise = 0,
                        sleep = 0, verboseColumns = FALSE, roiList = list(), firstBaselineCheck = TRUE,
                        roiScales = NULL) {
  args <- list(X = 1:length(fileNames(object)),
               FUN = filterFile, object = object)
  applied_args = do.call("lapply", args)
  df = map_dfr(applied_args, function(x) get_region_of_interest(x, ppm, peakwidth, snthresh, prefilter, mzCenterFun, integrate, mzdiff,
                                                                fitgauss, noise, sleep, verboseColumns, roiList, firstBaselineCheck, roiScales))
  return(df)
  }


get_region_of_interest <- function(object, ppm = 25, peakwidth = c(20, 50), snthresh = 10, prefilter = c(3, 100), mzCenterFun = "wMean",
                                   integrate = 1, mzdiff = -0.001, fitgauss = FALSE, noise = 0,
                                   sleep = 0, verboseColumns = FALSE, roiList = list(), firstBaselineCheck = TRUE,
                                   roiScales = NULL) {
  x = xcms::spectra(object, BPPARAM = SerialParam())
  rt = rtime(object)
  mzs <- lapply(x, mz)
  vals_per_spect <- lengths(mzs, FALSE)

  mz = unlist(mzs, use.names = FALSE)
  int = unlist(lapply(x, xcms::intensity), use.names = FALSE)
  valsPerSpect = vals_per_spect
  scantime = rt

  df = centWave_new_modified(rawData=object, mz=mz, int=int, scantime=scantime, valsPerSpect=valsPerSpect, ppm=ppm, peakwidth=peakwidth, snthresh=snthresh,
                             prefilter=prefilter, mzCenterFun=mzCenterFun,integrate=integrate, mzdiff=mzdiff, fitgauss=fitgauss,
                             noise=noise, sleep=sleep, roiList=roiList, firstBaselineCheck=firstBaselineCheck, roiScales=roiScales)
  return(df)
}

#### end ####

# # # Perform ROI picking for the beer data
# mzml_dir <- 'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\Data\\multibeers_urine_data\\beers\\fullscan'
# mzml_files <- list.files(path=mzml_dir, pattern='*.mzML', full.names=TRUE)
# design <- getDesign(mzml_files, "group1")
# 
# # mzml_files = mzml_files[1]
# # design <- getDesign(mzml_files, "group1")
# 
# # ppm value is set fairly large.
# # Other parameters for peakwidth, snthresh, prefilter for justin's data, taken from 
# # https://www.dropbox.com/home/Meta_clustering/ms2lda/large_study/r/beer_method_3_pos?preview=xcmsPeakPicking.R
# ppm = 10
# peakwidth = c(5, 100)
# snthresh = 3
# prefilter = c(3, 1000)
# mzdiff = 0.001

# outfile <- 'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\Data\\multibeers_urine_data\\beers\\fullscan\\rois.csv'
# raw_data <- readData(mzml_files, design, mode='onDisk')
# pos_df = get_regions_of_interest(raw_data, ppm=ppm, peakwidth=peakwidth, snthresh=snthresh, prefilter=prefilter, mzdiff=mzdiff)
# pos_df$mode = rep('Positive', nrow(pos_df))
# write_csv(pos_df, outfile)

# # Perform ROI picking for the urine data
mzml_dir <- 'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\Data\\multibeers_urine_data\\urines\\fullscan'
mzml_files <- list.files(path=mzml_dir, pattern='*.mzML', full.names=TRUE)
design <- getDesign(mzml_files, "group1")

# mzml_files = mzml_files[1]
# design <- getDesign(mzml_files, "group1")

# ppm value is set fairly large.
# Other parameters for peakwidth, snthresh, prefilter for justin's data, taken from 
# https://www.dropbox.com/home/Meta_clustering/ms2lda/large_study/r/beer_method_3_pos?preview=xcmsPeakPicking.R
ppm = 10
peakwidth = c(5, 100)
snthresh = 3
prefilter = c(3, 1000)
mzdiff = 0.001

outfile <- 'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\Data\\multibeers_urine_data\\urines\\fullscan\\rois.csv'
raw_data <- readData(mzml_files, design, mode='onDisk')
pos_df = get_regions_of_interest(raw_data, ppm=ppm, peakwidth=peakwidth, snthresh=snthresh, prefilter=prefilter, mzdiff=mzdiff)
pos_df$mode = rep('Positive', nrow(pos_df))
write_csv(pos_df, outfile)