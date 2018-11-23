#install.packages('AR1seg')
#install.packages('wbs')
#install.packages("readxl")

#################################################################
# Set the working directory
#################################################################
getwd()
setwd('sumitomo')



#####################################################################
# readxl
#####################################################################
library(readxl)
help("readxl")

raw_data <- read_excel(path = "data/IOT/Abnormality Detection May17 ~ Jan18.xlsx",
                      range = cell_limits(c(4, 6), c(NA, 12)),
                      col_names = TRUE)
sx <- raw_data[,3]
sx <- sx[[1]]
na_ind <- is.na(sx)
sx <- sx[!na_ind]
plot(sx[1:10000])

##################################################################
# AR1seg
##################################################################
library(AR1seg)
help("AR1seg")
data(y)
res=AR1seg_func(y,Kmax=15,rho=TRUE)
a=c(1,res$PPSelectedBreaks[1:(res$PPselected-1)]+1)
b=res$PPSelectedBreaks[1:(res$PPselected)]
Bounds=cbind(a,b)
mu.fit=rep(res$PPmean,Bounds[,2]-Bounds[,1]+1)
plot(y)
lines(mu.fit,col="red")

# my data
res=AR1seg_func(sx,Kmax=5,rho=TRUE)
a=c(1,res$PPSelectedBreaks[1:(res$PPselected-1)]+1)
b=res$PPSelectedBreaks[1:(res$PPselected)]
Bounds=cbind(a,b)
mu.fit=rep(res$PPmean,Bounds[,2]-Bounds[,1]+1)
plot(sx)
lines(mu.fit,col="red")


##################################################################
# wbs
##################################################################
library(wbs)
help(wbs)
x <- rnorm(300) + c(rep(1,50),rep(0,250))
w <- wbs(x)
plot(w)
w.cpt <- changepoints(w)
w.cpt
th <- c(w.cpt$th,0.7*w.cpt$th)
w.cpt <- changepoints(w,th=th)
w.cpt$cpt.th

##########################################################################
#
#############################################################################
## Nile data with one breakpoint: the annual flows drop in 1898
## because the first Ashwan dam was built
library(strucchange)

data("Nile")
plot(Nile)
bp.nile <- breakpoints(Nile ~ 1)
summary(bp.nile)
plot(bp.nile)
## compute breakdates corresponding to the
## breakpoints of minimum BIC segmentation
breakdates(bp.nile)
## confidence intervals
ci.nile <- confint(bp.nile)
breakdates(ci.nile)
ci.nile
plot(Nile)
lines(ci.nile)



bp.nile <- breakpoints(sx ~ 1)
summary(bp.nile)
plot(bp.nile)
## compute breakdates corresponding to the
## breakpoints of minimum BIC segmentation
breakdates(bp.nile)
## confidence intervals
ci.nile <- confint(bp.nile)
breakdates(ci.nile)
ci.nile
plot(Nile)
lines(ci.nile)




