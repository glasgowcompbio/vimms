library(xcms)
library(magrittr)

files = c('Fullscan_QCA.mzML', 'Fullscan_QCB.mzML',
          'SmartROI_QCA_ms1_shift_0.mzML', 'SmartROI_QCB_ms1_shift_0.mzML',
          'TopN_QCA.mzML', 'TopN_QCB.mzML')

# filename = 'Fullscan_QCA.mzML'
# mzr = c(215.0704, 215.0708)
# rtr = c(67.29037, 92.00665)
# raw_data %>%
#   filterRt(rt = rtr) %>%
#   filterMz(mz = mzr) %>%
#   plot(type = "XIC")

cwp <- CentWaveParam(ppm=15, peakwidth = c(15, 80), snthresh=5, noise = 1000,
                     prefilter = c(3, 500))

for (filename in files) {
  raw_data <- readMSData(files = filename,
                         pdata = new("NAnnotatedDataFrame",
                                     data.frame(sample_name=tools::file_path_sans_ext(filename))), msLevel.=1,
                         mode = "onDisk")
  xdata <- findChromPeaks(raw_data, param = cwp)
  cp = chromPeaks(xdata)
  data_out = data.frame(1:nrow(cp), cp[,'mz'], cp[,'rt'],
                        cp[,'rtmin'], cp[,'rtmax'],
                        cp[,'maxo'], cp[,'into'],
                        cp[,'mzmin'], cp[,'mzmax'])
  data_out = data_out[order(data_out[,2]),]
  data_out[,1] = 1:nrow(cp)
  colnames(data_out) = c('row ID', 'row m/z', 'row retention time', paste(filename, 'Peak RT start'),
                         paste(filename, 'Peak RT end'), paste(filename, 'Peak height'),
                         paste(filename, 'Peak area'), paste(filename, 'Peak m/z min'),
                         paste(filename, 'Peak m/z max'))
  write.csv(data_out, file=paste0(tools::file_path_sans_ext(filename), '_xcms_box.csv'), row.names=FALSE)
}
