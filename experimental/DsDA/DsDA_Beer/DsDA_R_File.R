#if (!requireNamespace("BiocManager", quietly = TRUE))
# install.packages("BiocManager")
#BiocManager::install("xcms", version = "3.8")


library(xcms)
library(mzR)

DsDA<-function(
  negmode=FALSE,
  mzppm=20, 
  pw=c(3,30), 
  sn=20,
  msscantime=0.18,
  msmsscantime=0.035,
  lockmassscantime=0.18,
  isd=0.02, #not used
  nmsms=4,
  lockmassfrequency=20,
  runtime=1020, # seem to need last time of schedule
  lockmassrefine=FALSE,
  fileIgnore='trash',
  filterCond=c(2,20),
  initLM=4.7,
  msCE=6,
  initCE=25,
  altCE=7:60,
  idleStop=10,
  tofMult=16.625,
  repeatMSMSfill=8,
  schedule.file='DsDA_schedule.csv',
  minPeaks=100,
  min.rel.fragment.int=0.01,
  min.abs.fragment.int=10,
  init.CE.opt=10,
  ionsLessThanPrecursorOnly=TRUE,
  maximal.depth=NULL,
  prefilter = c(20, 80),
  mzdiff = 0.001,
  dsda_dir=NA
) {
  
  
  
  
  #library(xcms)
  #library(mzR)
  
  schedule<-read.csv(schedule.file, header=TRUE)
  
  ## having a run time as long as the method can cause WRENS problems, s
  ## so we do not use the last 5 seconds of the run for WRENS  this is now handled within WRENS - more stable
  # runtime<- runtime - 5
  
  schedule<-schedule[which(schedule$rt < (max(schedule$rt)-5)),]
  headers<-c("MS1_MASS_DAC_SETTING",	"LM_RESOLUTION_SETTING",	"SOURCE_BIAS_SETTING",	
             "SAMPLE_CONE_VOLTAGE_SETTING",	"targetMass",	"targetTime")
  
  if(any(msCE == altCE)) {
    warning("CE ", msCE, " removed from altCE to ensure msCE is unique", '\n')
    altCE <- altCE[!altCE %in% msCE]
  }
  
  if(!is.null(maximal.depth)) {
    depth.injections <- maximal.depth*(1:10000)
  } else {
    depth.injections <- NULL
  }
  inj.number <- 0
  
  setwd(dsda_dir)
  lm<-which(schedule$type=="lm")
  ms<-which(schedule$type=="ms")
  msms<-which(schedule$type=="msms")
  
  #outtxt<-vector(length=nrow(schedule), mode="character")
  # for(i in 1:length(schedule)) {
  # outtxt[i]<-
  
  # }
  
  outcsv<-matrix(nrow=nrow(schedule), ncol=length(headers))
  dimnames(outcsv)[[2]]<-headers
  outcsv[,"targetTime"]<-schedule$rt
  outcsv[,"SAMPLE_CONE_VOLTAGE_SETTING"]<- 30
  outcsv[lm,"SOURCE_BIAS_SETTING"]<-NA
  outcsv[ms,"SOURCE_BIAS_SETTING"]<-NA
  outcsv[msms,"SOURCE_BIAS_SETTING"]<-NA
  outcsv[msms,"LM_RESOLUTION_SETTING"]<-NA
  cleanoutcsv<-outcsv
  
  ## assign random masses
  outcsv[msms, "MS1_MASS_DAC_SETTING"]<- (sample((50000:2000000), length(msms))/1000)  *tofMult
  
  # if we want to run in negative ionization mode, we must set CE and cone voltage to a negative value
  if(negmode) {
    outcsv[,"SOURCE_BIAS_SETTING"]<- -1*abs(outcsv[,"SOURCE_BIAS_SETTING"])
    outcsv[,"SAMPLE_CONE_VOLTAGE_SETTING"] <- -1*abs(outcsv[,"SAMPLE_CONE_VOLTAGE_SETTING"])
  }
  
  outtxt<-vector(length=nrow(schedule), mode="character")
  for(i in 1:nrow(schedule)) {
    ##string command="addToQueueV,";// comma delimit split
    ## for( int y=0; y<values.Length-2;y++) // for each property column from csv - 
    ## // length minus 1 is the retention time, length minus 2 is human readable targetMass
    ## {
    ##command+=names[y]+","+values[y].ToString()+",false ";
    tmp<-"addToQueueV,"	
    for(j in which(!is.na(outcsv[i,1:4]))) {
      tmp<-paste0(tmp, headers[j], "," , outcsv[i,j], ",false ")
    }
    # tmp<-paste0(tmp,"false ")
    tmp<-substr(tmp,1,nchar(tmp)-1)
    outtxt[i]<-tmp
  }
  
  # shorten the script for the first two DDA injections to make sure it doesn't run over.
  # outtxt<-outtxt[1:round((length(outtxt)/2))]
  
  
  dir.create("Data")
  dir.create('xset')
  dir.create('settings')
  dir.create('processed')
  dir.create('controllers')
  
  # write.csv(outcsv, file="csv/active.csv", row.names=FALSE, quote=FALSE) 
  write(outtxt, file="settings/active.txt")
  cycletime<-round(1000*max(schedule[ms,"rt"])/length(ms))
  write(cycletime, file="settings/cycletime.txt")
  
  ## list all raw files in directory
  nfilesold<-0
  nfilesnew<-0
  
  repeat {
    idle<-vector(length=0)
    cat("waiting for new .mzml files", '\n')
    startwhile <-Sys.time()
    currentwhile<-Sys.time()
    while(length(idle) < 1) { # this loop goes round and round until it finds a file, at which point idle become something which has length > 0
      Sys.sleep(10)
      raw<-list.files(path="./Data", pattern = '[mM][zZ][mM][lL]')
      raw<-raw[!grepl(fileIgnore, raw)]
      currenttime<-Sys.time()
      times = rep(2000,length(raw))
      base<-gsub('.[mM][zZ][mM][lL]', "", raw)
      idle<-which(times > (runtime)+60)  ## add file size > new file.  I think this will keep it from triggering when injection is being made. 
      mzML<-list.files(path="./mzML", pattern = 'mzML')
      mlbase<-gsub(".mzML", "", mzML)
      idle<-which(!base[idle] %in% mlbase)
      currentwhile<-Sys.time()
      if(as.numeric(difftime(currentwhile, startwhile, units='mins'))> idleStop ) {
        #dev.off()
        break()
      }
    }
    
    if(as.numeric(difftime(currentwhile, startwhile, units='mins'))> idleStop ) {
      #dev.off()
      break()
    }
    
    ##if there are any that are idle proceed with further processing
    if(length(idle)>0) {
      do<-idle[1]
      if(length(do)>0) {
        
        # copy data across
        cat("Found file. Sleeping to allow file to fully copy over", '\n')
        Sys.sleep(60)

        # Use XCMS Centwave algorithm to detect peaks
        
        xsraw<-xcmsRaw(paste0(getwd(),'/Data/', base[do], ".mzML"), includeMSn=TRUE)
        f<-paste0(getwd(),'/Data/', base[do], ".mzML")
        lg<-0
        xset<-xcmsSet(files=f, nSlaves=1, method="centWave", ppm=mzppm, peakwidth=pw, snthresh=sn, 
                      fitgauss=TRUE, verbose.columns=TRUE, prefilter=prefilter,mzdiff=mzdiff)
        save(xset, file=paste0("xset/", base[do], ".Rdata"))
        
        if(nrow(xset@peaks) < minPeaks) {
          cat("file found with fewer than", minPeaks, "peaks", '\n')
          break
        }
        
        if(!any(ls()=="allPeaks")) {  # if allPeaks does not exist, make an empty version of it.
          
          allPeaks<-data.frame(xset@peaks[,], 
                               "ctms" = rep(1, nrow(xset@peaks)), 
                               "qualms" = rep(0, nrow(xset@peaks)), 
                               "tic.msms" = rep(0, nrow(xset@peaks)),
                               "ctmsms" = rep(0, nrow(xset@peaks)), 
                               "qualmsms" = rep(0, nrow(xset@peaks)), 
                               "priority" = rep(0, nrow(xset@peaks)), 
                               "nextce" = rep(initCE, nrow(xset@peaks)), 
                               "nmsms" = rep(1, nrow(xset@peaks)))
          
          if(!any(ls()=="allMSMS")) {allMSMS<-as.list(rep(NA, nrow(allPeaks)))}
          s<-data.frame("file"="temp" , 
                        "rt"=0, 
                        "ce"=0, 
                        "mz"=0, 
                        "index"=0, 
                        "scannum"=0,
                        "int"= 0,
                        "wmmz"=0, 
                        "precRem"=0, 
                        "complexity"=0, 
                        "ctms"=1,
                        "qualmsms"=0, 
                        "nextce"=0
          )
          sclean<-s
          s<-s[0,]
          msmstemp<-as.list(c("score"=0, "summary"=NA, "spectra"=NA))
          msmstemp$summary<-s
          msmstemp$spectra<-as.list(vector(length=0))
          rm(s)
          
        } 
        # print(cat("check1", '\n'))
        lenAll<-nrow(allPeaks)
        
        av<-names(allPeaks)[!names(allPeaks) %in% "ctms"]
        av<-av[!av %in% c("rt", "rtmin", "rtmax")]
        newPeaks<-data.frame(xset@peaks[,], 
                             "ctms" = rep(0, nrow(xset@peaks)), 
                             "qualms" = rep(0, nrow(xset@peaks)), 
                             "tic.msms" = rep(0, nrow(xset@peaks)),
                             "ctmsms" = rep(0, nrow(xset@peaks)), 
                             "qualmsms" = rep(0, nrow(xset@peaks)), 
                             "priority" = nrow(xset@peaks) - rank(xset@peaks[,"into"]), 
                             "nextce" = rep(initCE, nrow(xset@peaks)), 
                             "nmsms" = rep(0, nrow(xset@peaks)))
        
        
        if(!identical(allPeaks, newPeaks)){
          xm<-data.frame("allPeaksInd"=1:nrow(allPeaks), 
                         "allPeaksrt"=allPeaks[,"rt"], 
                         "newPeaksInd"=rep(NA, nrow(allPeaks)), 
                         "newPeaksrt"=rep(NA, nrow(allPeaks)))
          
          lengths<-vector()
          for(x in 1:nrow(newPeaks)) {
            mtch<-which( abs(allPeaks$mz - newPeaks[x,"mz"])< (newPeaks[x,"mz"]*mzppm/1000000) &
                           allPeaks$rt >= newPeaks$rtmin[x] &
                           allPeaks$rt <= newPeaks$rtmax[x])
            
            if(length(mtch)==0) {
              allPeaks<-rbind(allPeaks, newPeaks[x,])
            }
            if(length(mtch)>0) {
              if(length(mtch)>1) {
                mtch<-mtch[which.max(allPeaks$into[mtch])]
              }
              xm[mtch,"newPeaksInd"]<-x
              xm[mtch,"newPeaksrt"]<-newPeaks[x, "rt"]
              allPeaks[mtch,av]<-(allPeaks[mtch,av]+newPeaks[x,av])/2
              #allPeaks[mtch,ct]<-pmax(allPeaks[mtch,su]+newPeaks[x,su], na.rm=TRUE)
              allPeaks[mtch,"ctms"]<-allPeaks[mtch,"ctms"]+1
            }
            rm(mtch)
          }
          
          xm$dev<-xm$allPeaksrt-xm$newPeaksrt
          plot(xm$allPeaksrt, xm$dev, pch=19, xlab="allPeaks rt (sec)", ylab="newPeaks rt deviation (sec)", main= "red points: quadratic fit")
          fit<-lm(dev~allPeaksrt+ I(allPeaksrt^2), data=xm)
          oldind<-which(!is.na(xm[,"allPeaksrt"]))
          predold<-predict(fit, newdata=xm[oldind,])
          points(xm[oldind, "allPeaksrt"], predold, pch=19, cex=0.5, col="red")
          allPeaks[oldind, "rt"]<-allPeaks[oldind, "rt"]-predold
          allPeaks[oldind, "rtmin"]<-allPeaks[oldind, "rtmin"]-predold
          allPeaks[oldind, "rtmax"]<-allPeaks[oldind, "rtmax"]-predold
        }
        
        length(allMSMS) <- nrow(allPeaks)
        isnull<-unlist(sapply(1:length(allMSMS), FUN=function(x) {if(is.null(allMSMS[[x]])) return(x)}))
        fillna<-unlist(sapply(1:length(allMSMS), FUN=function(x) if(length(allMSMS[[x]])==1) {return(x)}))
        if(length(isnull)>0) {for(k in isnull) {allMSMS[[k]]<-msmstemp}}
        if(length(fillna)>0) {for(k in fillna) {allMSMS[[k]]<-msmstemp}}
        
        if(is.null(xsraw)) next
        for(i in 1:length(xsraw@msnScanindex)) {
          mtch<-which(abs(allPeaks$mz - xsraw@msnPrecursorMz[i]) <(xsraw@msnPrecursorMz[i]*mzppm/1000000)  & 
                        # abs(allPeaks$mz - newPeaks[x,"mz"])< (newPeaks[x,"mz"]*mzppm/1000000)
                        allPeaks$rtmin <= xsraw@msnRt[i] & 
                        allPeaks$rtmax >= xsraw@msnRt[i])
          
          if(length(mtch)>0) {
            
            for(j in 1:length(mtch)) {
              tmpmsms<-getMsnScan(xsraw, i)
              if(nrow(tmpmsms) < 1) {break}
              fullmsms<-tmpmsms
              if(ionsLessThanPrecursorOnly) {
                use<-which(tmpmsms[,1]<= (xsraw@msnPrecursorMz[i] + 0.5) )
                if(length(use)>0) {
                  tmpmsms<-tmpmsms[use,, drop=FALSE]
                }
                rm(use)
              }
              if(is.numeric(min.abs.fragment.int)) {
                use<-which(tmpmsms[,2] >= min.abs.fragment.int)
                if(length(use)>0) {
                  tmpmsms<-tmpmsms[use,, drop=FALSE]
                }
                rm(use)
              }
              if(is.numeric(min.rel.fragment.int)) {
                use<-which(tmpmsms[,2] >= (min.rel.fragment.int*max(tmpmsms[,2])))
                if(length(use)>0) {
                  tmpmsms<-tmpmsms[use,, drop=FALSE]
                }
                rm(use)
              }
              
              if(nrow(tmpmsms)>=1)
              {
                int<-max(1, sum(tmpmsms[,2]), na.rm=TRUE)
                wmmz<-weighted.mean(tmpmsms[,1], tmpmsms[,2])/xsraw@msnPrecursorMz[i]
                use<-which(abs(tmpmsms[,1]-xsraw@msnPrecursorMz[i])<=0.1)
                if(length(use)>0) {precRem<-max(0, sum(tmpmsms[use,2])/int, na.rm=TRUE)} else {precRem<-0}
                complexity<-weighted.mean(tmpmsms[,2]/max(tmpmsms[,2]), weights=(1/tmpmsms[,2]))
                quality<-int #*complexity
                # nextce<-round(xsraw@msnPrecursorMz[i] + (xsraw@msnPrecursorMz[i] * (wmmz-0.5)))
                nextce<-sample(altCE, 1)
              } else {
                int<- 0
                wmmz<-NA
                precRem<-0
                complexity<-0
                quality<-0
                nextce<-sample(altCE, 1)
              }
              
              s<-sclean
              s$file=xsraw@filepath[1] 
              s$rt=xsraw@msnRt[i] 
              s$ce=xsraw@msnCollisionEnergy[i]
              s$mz=xsraw@msnPrecursorMz[i] 
              s$index=i 
              s$scannum=xsraw@acquisitionNum[i]
              s$int=int
              s$wmmz=wmmz 
              s$precRem=precRem
              s$complexity=complexity
              s$qualmsms=quality
              s$nextce=nextce
              
              allMSMS[[mtch[j]]]$summary<-rbind(allMSMS[[mtch[j]]]$summary, s)
              allMSMS[[mtch[j]]]$spectra[[nrow(allMSMS[[mtch[j]]]$summary)]]<-fullmsms
              allMSMS[[mtch[j]]]$score<- sum(allMSMS[[mtch[j]]]$summary$quality, na.rm=TRUE)
              rm(tmpmsms); rm(s); rm(int); rm(wmmz); rm(precRem); rm(complexity); rm(quality); rm(use);
            }
            rm(mtch);
          }  ## end if(length(mtch)>0)
        } ## end for(i in 1:length(xsraw@msnScanindex))
        for(i in 1:nrow(allPeaks)){
          if(length(raw)>=init.CE.opt) {
            ## this is where the polynomial fit would go.  see commented lines above.
            allPeaks[i,"nextce"]<-sample(altCE, 1)
          } else {
            allPeaks[i,"nextce"]<-sample(altCE, 1)
          }
        }
        msms.qual<-unlist(sapply(1:length(allMSMS), FUN=function(x) {
          if(nrow(allMSMS[[x]]$summary) > 0) {
            sum(allMSMS[[x]]$summary$qualmsms)
          } else {
            0
          }
          
        }))
        
        msms.count<-unlist(sapply(1:length(allMSMS), FUN=function(x) {nrow(allMSMS[[x]]$summary)}))
        if(sd(msms.qual)==0) {
          msms.rnk<-rep(0, nrow(allPeaks))
        } else {
          msms.rnk<-(rank(msms.qual, ties.method = "random")) ; msms.rnk<-1+max(msms.qual)-msms.qual
        }
        allPeaks$tic.msms<-msms.qual
        allPeaks$qualmsms<-msms.rnk
        allPeaks$ctmsms<-msms.count
        
        ## update all scores in MS data
        if(mean(allPeaks$ctms) <= 1) {
          ctms.rnk<-rep(0, nrow(allPeaks))
        } else {
          ctms.rnk<-rank(allPeaks$ctms, ties.method = "random") ; ctms.rnk<-1+max(ctms.rnk)-ctms.rnk
        }
        into.rnk<-rank(allPeaks$into, ties.method = "random") ; into.rnk<-1+ max(into.rnk)-into.rnk
        ms.rnk<-rank((ctms.rnk+into.rnk)/2) #; ms.rnk<-1+ max(ms.rnk)-ms.rnk
        allPeaks$qualms<-ms.rnk
        
        allPeaks$priority<-rank(allPeaks$qualms-allPeaks$qualmsms, ties.method = "random")
        
        if(is.null(depth.injections) | !any(inj.number %in% depth.injections)) {
          priority.order<-order(allPeaks$priority)
        } else {
          missingmsms<-which(allPeaks$ctmsms == 0)
          priority.order<-missingmsms[order(allPeaks$priority[missingmsms])]
        }
        allPeaks$"nmsms"<-rep(1, nrow(allPeaks))
        addone<-which(allPeaks[,"into"] <= as.numeric(quantile(allPeaks[,"into"])[3]))
        allPeaks[addone,"nmsms"]<-allPeaks[addone,"nmsms"]+1
        
        rm(mtch)
        d<-Sys.time()
        a<-Sys.time()
        suppressWarnings(dir.create("summaryTables"))
        dt<-format(Sys.time(), format="%Y%m%d_%H%M%S")
        save(allPeaks, file=paste0('processed/allpeaks_', dt, ".Rdata"))
        save(allMSMS, file=paste0('processed/allmsms_', dt, ".Rdata"))
        # cat("check5", '\n')
        cat("nrow allPeaks: ", nrow(allPeaks), '\n')
        
        outcsv<-cleanoutcsv
        open<-msms[which(is.na(outcsv[msms,"MS1_MASS_DAC_SETTING"]))]
        opens<-data.frame(matrix(nrow=0, ncol=2)); names(opens)<-c("n", "open")
        # opens[nrow(opens)+1,] <-c(n, length(open))
        cat(" - assigning MS/MS events to time slots from file:", base[do], '\n')
        #cat("n", 0, ":  openvals =", length(open), '\n' )
        for(n in 1:repeatMSMSfill) {
          if(n==1) {cat("n", 0, ":  openvals =", length(open), '\n' )}
          for(i in priority.order) {
            for(j in 1:allPeaks[i,"nmsms"]){
              {
                open<-msms[which(is.na(outcsv[msms,"MS1_MASS_DAC_SETTING"]))]
                tarrow<-open[which(outcsv[open,"targetTime"] > allPeaks[i,"rtmin"] &  
                                     outcsv[open,"targetTime"] < allPeaks[i,"rtmax"])]
                if(length(tarrow)>0) {
                  if(length(tarrow)>1) {
                    tarrow<-tarrow[which.min(abs(outcsv[tarrow,"targetTime"] - allPeaks[i,"rt"] ) )]
                  }
                  outcsv[tarrow,1:5]<-c(
                    round(tofMult*allPeaks[i,"mz"], digits=4), 
                    NA,
                    #round(allPeaks[i,"nextce"]),  #nextce<-sample(altCE, 1)
                    sample(altCE, 1),
                    30,
                    round(allPeaks[i,"mz"], digits=4)
                  )
                }
              }
            }
          }
          cat("n", n, ":  openvals =", length(open), '\n' )
          opens[nrow(opens)+1,] <-c(n, length(open))
        }
        
        cat("filling remaining open rows which fall within feature times", '\n')
        open<-msms[which(is.na(outcsv[msms,"MS1_MASS_DAC_SETTING"]))]
        for(i in open) {
          tarcmpd<-which(allPeaks$rtmax > outcsv[i,"targetTime"] & allPeaks$rtmin < outcsv[i,"targetTime"])
          
          if(length(tarcmpd) > 0 ) {
            tarcmpd<-tarcmpd[which.min(allPeaks[tarcmpd,"priority"])]
            outcsv[i,1:5]<-c(
              round(tofMult*allPeaks[tarcmpd,"mz"], digits=4), 
              NA,
              round(allPeaks[tarcmpd,"nextce"]), 
              30,
              round(allPeaks[tarcmpd,"mz"], digits=4)
            )
          }
        }
        open<-msms[which(is.na(outcsv[msms,"MS1_MASS_DAC_SETTING"]))]
        cat("final :  openvals =", length(open), '\n' )
        
        outtime<-Sys.time()
        
        # if we want to run in negative ionization mode, we must set CE to a negative value
        if(negmode) {
          outcsv[,"SOURCE_BIAS_SETTING"]<- -1*abs(outcsv[,"SOURCE_BIAS_SETTING"])
          outcsv[,"SAMPLE_CONE_VOLTAGE_SETTING"] <- -1*abs(outcsv[,"SAMPLE_CONE_VOLTAGE_SETTING"])
        }
        
        write.csv(outcsv, file=paste0("settings/", format(outtime, format="%Y-%m-%d-%H-%M-%S"), ".csv"))
        # cat(head(outcsv))
        outtxt<-vector(length=nrow(schedule), mode="character")
        for(i in 1:nrow(schedule)) {
          tmp<-"addToQueueV,"	
          for(j in which(!is.na(outcsv[i,1:4]))) {
            tmp<-paste0(tmp, headers[j], "," , outcsv[i,j], ",false ")
          }
          tmp<-substr(tmp,1,nchar(tmp)-1)
          outtxt[i]<-tmp
        }
        
        write(outtxt, file="settings/active.txt")
        write(outtxt, file=paste0("settings/", format(outtime, format="%Y-%m-%d-%H-%M-%S"), ".txt"))
        
      } ## end if(length(do)>0)
      if(nrow(xset@peaks) < minPeaks) {
        break
      }
    }   ## end if length(idle)...
    if(nrow(xset@peaks) < minPeaks) {
      next
    }
    b<-Sys.time() 
    if(any(ls()=="a")) {  
      if(as.numeric(difftime(b, a, units='mins'))> idleStop ) {
        dev.off()
        break()
      }
    }
    
    nmsms<-unlist(sapply(1:length(allMSMS), FUN=function(x) {nrow(allMSMS[[x]]$summary)}))
    perMSMS<-length(which(allPeaks$ctmsms>0))/length(allPeaks$ctmsms)
    perMSMS10<-length(which(allPeaks$ctmsms>=10))/length(allPeaks$ctmsms)
    cat(round(100*(perMSMS), digits=2), "% of features have MS/MS spectra", '\n')
    cat(round(100*(perMSMS10), digits=2), "% of features have 10 or more MS/MS spectra", '\n', '\n', '\n')
  } ## end repeat
  
  cat("finish guiding MS/MS acquisition", "in project", getwd(), '\n')
  
  out<-as.list(c(NA, NA)); names(out)<-c("allPeaks", "allMSMS")
  out$allPeaks<-allPeaks
  out$allMSMS<-allMSMS
  
  return(out)
}  ## end function

## run function
out<-DsDA(
  negmode = TRUE,
  mzppm=3, 
  pw=c(5,100), 
  sn=3,
  msscantime=0.18,
  msmsscantime=0.035,
  lockmassscantime=0.18,
  isd=0.02,
  nmsms=4,
  lockmassfrequency=20,
  runtime=1020,
  lockmassrefine=FALSE,
  fileIgnore='trash',
  filterCond=c(2,20),
  initLM=4.7,
  msCE=6,
  initCE=22,
  altCE=seq(60, 120, 20),
  idleStop=10, # adjust this to choose how quickly it stops. it will wait T minutes between files, beyond that it will stop and save some extra files
  tofMult=16.625,
  repeatMSMSfill=4,
  schedule.file='../DsDA_Timing_schedule.csv',
  minPeaks=100,
  min.rel.fragment.int=0.001,
  min.abs.fragment.int=NULL,
  init.CE.opt=10,
  ionsLessThanPrecursorOnly=TRUE,
  maximal.depth=NULL,
  prefilter = c(3,100),
  mzdiff = 0.001,
  dsda_dir = getwd()
)


