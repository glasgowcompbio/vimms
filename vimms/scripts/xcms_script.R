library(xcms)
library(magrittr)
library(optparse)

parser <- OptionParser()

#centwave params
parser <- add_option(parser, c("--ppm"), type="integer", default=15, help="Centwave Parts-Per-Million")
parser <- add_option(parser, c("--pwlower"), type="integer", default=15, help="Centwave lower bound for peakwidth")
parser <- add_option(parser, c("--pwupper"), type="integer", default=80, help="Centwave upper bound for peakwidth")
parser <- add_option(parser, c("--snthresh"), type="integer", default=5, help="Centwave snthresh")
parser <- add_option(parser, c("--noise"), type="integer", default=1000, help="Centwave noise")
parser <- add_option(parser, c("--prefilterlower"), type="integer", default=3, help="Centwave lower bound for prefilter")
parser <- add_option(parser, c("--prefilterupper"), type="integer", default=500, help="Centwave upper bound for prefilter")
parser <- add_option(parser, c("--mzdiff"), type="double", default=0.001, help="Centwave min RT separation before merging peaks")

#groupPeaksNearest params
parser <- add_option(parser, c("--mzvsrtbalance"), type="integer", default=10, help="Group peaks nearest m/z vs RT distance weighting")
parser <- add_option(parser, c("--absmz"), type="double", default=0.2, help="Group peaks nearest max m/z tolerance")
parser <- add_option(parser, c("--absrt"), type="double", default=15, help="Group peaks nearest max RT tolerance")
parser <- add_option(parser, c("--kNN"), type="integer", default=10, help="Group peaks nearest max number of neighbours checked")

args <- parse_args(parser, positional_arguments=TRUE)
output = args$args[1]
files = args$args[-1]

cwp <- CentWaveParam(
  ppm = args$options$ppm, 
  peakwidth = c(args$options$pwlower, args$options$pwupper), 
  snthresh = args$options$snthresh, 
  noise = args$options$noise, 
  prefilter = c(args$options$prefilterlower, args$options$prefilterupper),
  mzdiff = args$options$mzdiff
)

owp <- ObiwarpParam(binSize = 0.6)

npp <- NearestPeaksParam(
  mzVsRtBalance = args$options$mzvsrtbalance, 
  absMz = args$options$absmz,
  absRt = args$options$absrt,
  kNN = args$options$kNN,
)

cols_per_peak = 9
make_headers <- function(fname){
  bname <- basename(fname)
  return(
    c(
      paste(bname, 'filtered Peak status'),
      paste(bname, 'filtered Peak m/z'),
      paste(bname, 'filtered Peak RT'),
      paste(bname, 'filtered Peak RT start'), 
      paste(bname, 'filtered Peak RT end'), 
      paste(bname, 'filtered Peak height'), 
      paste(bname, 'filtered Peak area'), 
      paste(bname, 'filtered Peak m/z min'), 
      paste(bname, 'filtered Peak m/z max')
    )
  )
}

raw_data <- readMSData(
  files = files,
  msLevel.=1,
  mode = "onDisk"
)
xdata <- findChromPeaks(raw_data, param = cwp)

if(length(files) < 2){

  cp = chromPeaks(xdata)
  if(length(cp) >= 1){
    data_out = data.frame(
      1:nrow(cp), 
      cp[, 'mz'], 
      cp[, 'rt'], 
      'DETECTED', 
      cp[, 'mz'],
      cp[, 'rt'],
      cp[, 'rtmin'], 
      cp[, 'rtmax'], 
      cp[, 'maxo'], 
      cp[, 'into'], 
      cp[, 'mzmin'], 
      cp[, 'mzmax']
    )
    #data_out = data_out[order(data_out[,2]),]
    data_out[,1] = 1:nrow(cp)
  }else{
    data_out <- data.frame(matrix(ncol = 3 + length(files) * cols_per_peak, nrow = 0))
  }
  
}else{

  #xdata <- adjustRtime(xdata, param = owp)
  #pdp <- PeakDensityParam(sampleGroups = xdata$sample_group, minFraction = 0.4, bw = 30)
  #xdata <- groupChromPeaks(xdata, param = pdp)
  xdata <- groupChromPeaks(xdata, param=npp)
  cp = chromPeaks(xdata)
  features = featureDefinitions(xdata)
  
  print(head(cp))
  print(head(features))
  
  if(nrow(features) >= 1){
    num_col = 3 + length(files) * cols_per_peak
    data_out <- data.frame(matrix(ncol = num_col, nrow = nrow(features)))
    data_out[] = 0.0
    data_out[, 1:3] = data.frame(1:nrow(features), features[, 'mzmed'], features[, 'rtmed'])
    data_out[, seq(4, num_col, cols_per_peak)] = 'UNKNOWN'
    
    for(feat_idx in 1:nrow(features)){
      idx_ls <- features[[feat_idx, 'peakidx']]
      for(peak_idx in idx_ls){
        sample_idx <- 4 + cols_per_peak * (cp[peak_idx, 'sample'] - 1)
        data_out[feat_idx, sample_idx:(sample_idx+8)] = data.frame(
          'DETECTED',
          cp[[peak_idx, 'mz']],
          cp[[peak_idx, 'rt']],
          cp[[peak_idx, 'rtmin']],
          cp[[peak_idx, 'rtmax']],
          cp[[peak_idx, 'maxo']],
          cp[[peak_idx, 'into']],
          cp[[peak_idx, 'mzmin']],
          cp[[peak_idx, 'mzmax']]
        )
      }
    }
  }else{
    data_out <- data.frame(matrix(ncol = 3 + length(files) * cols_per_peak, nrow = 0))
  }

}

colnames(data_out) = c(
  'row ID', 'row m/z', 'row retention time', unlist(lapply(files, make_headers))
)
write.csv(data_out, file=output, row.names=FALSE, quote=FALSE)