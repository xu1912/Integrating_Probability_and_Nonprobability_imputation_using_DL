

# All Dr. Chen functions ----
# function1
# function2_5_14_2019
# gedata_test
# respoint


# Libraries ----
library(tidyverse)
pacman::p_load(
  tidyverse,
  tidymodels,
  tidyquant,
  mgcv,
  np,
  survey,
  sampling
  # torch,  # Uncomment to install! Once installed, the mlverse tabnet demo doesn't call the torch library (it's like those other libraries used in tidymodels engines)
  # tabnet
)

# library(tidyverse)
# library(tidymodels)
# library(tidyquant)
# 
# library(mgcv)     # function1
# library(np)       # function1
# 
# library(survey)   # function2_2019
# library(sampling) # gedata_test


# <><> function1 (the 6 estimators) ----
# fM_A ... A mean
# fM_B ... B mean
# fP_A ... PMIE (from sample A; uses lm())
# fKS  ... NPMIEK
# fKS1 ... (not used)
# fGAM ... NPMIEG
# fPW  ... PWE
# 1.0 - The sample mean from sample A ----
fM_A <- function(dat){

  # dat is the indat[[1]] from FRES
  # indat[[1]] is Res from gedata
  # Res is an array 1000 matrices deep of those 10,000x8 matrices
  # M   <- cbind(My,  # will become dat[,1]
  #              Mx1, # will become dat[,2]
  #              Mx2, # will become dat[,3]
  #
  #              Mx3, # will become dat[,4]
  #
  #              Mx4, # will become dat[,5]
  #
  #              MsIA,# will become dat[,6]
  #              MsIB,# will become dat[,7]
  #              Mw)  # will become dat[,8]
  y       <- dat[,1]
  x1      <- dat[,2]
  sIA     <- dat[,6]
  syA     <- y[sIA == 1]
  sx1A    <- x1[sIA == 1]
  
  etheta1 <- mean(syA)
  etheta2 <- median(syA)
  etheta  <- c(etheta1,etheta2)

  return(etheta)
}
# 
# # 2.0 The naive estimator from sample B ----
# # (sample mean)
fM_B <- function(dat){

  # M   <- cbind(My,  # will become dat[,1]
  #              Mx1, # will become dat[,2]
  #              Mx2, # will become dat[,3]
  #
  #              Mx3, # will become dat[,4]
  #
  #              Mx4, # will become dat[,5]
  #
  #              MsIA,# will become dat[,6]
  #              MsIB,# will become dat[,7]
  #              Mw)  # will become dat[,8]
  y       <- dat[,1]
  x1      <- dat[,2]
  sIB     <- dat[,7]
  syB     <- y[sIB==1]
  sx1B    <- x1[sIB==1]
  
  etheta1 <- mean(syB)
  etheta2 <- median(syB)
  etheta  <- c(etheta1,etheta2)

  return(etheta)
}


# . ----
# <><> gedata (generation of simulated data) ----
# gedata
## fsA
## fsrs
# OLD: B = 1000; N = 10000; nA = 500; nB = 1000; id_m = 1,2,3, or 4
# NEW: C = the controlling parameter for the sample size of sample B
## C = -.4.4 for Models 1-3 (n ~ 500); C = -20 for Model 4 (n ~ 500)
gedata_modified <- function(B,N,nA,C,id_m){
  
  # Super population model ----
  #set.seed(4123) # IMPORTANT: This won't make all datasets equal. If B = 1000 then you'll be able to make 1000 different datasets.
  
  x1 <- rnorm(B*N)
  
  x2 <- rnorm(B*N)
  
  # ADD x3 ----
  x3 <- rnorm(B*N)
  
  # ... ADD x4 ----
  x4 <- rnorm(B*N)
  
  epsilon <- rnorm(B*N)
  
  # For index == 1 ----
  if (id_m == 1){
    y <- 1 + x1 + x2 + x3 + x4 + epsilon
  }
  
  # For index == 2 ----
  #if (id_m == 2){
   # y <- 1 + x1^2 + x2^2 + x3^2 + x4^2 + epsilon
  #}
  
  # For index == 3 ----
  if (id_m == 3){
    y <- 1 + x1^2 + x1*x2*x3 + x1*x2 + x1*x2*x3*(x4^2) + epsilon
  }
  
  # For index == 4 ----
  if (id_m == 4){
    range_min <- -2
    range_max <- 2
    
    beta_0_thru_3 <- runif(4,range_min,range_max)
    
    alpha_1.1 <- runif(21,range_min,range_max)
    alpha_2.1 <- runif(21,range_min,range_max)
    alpha_3.1 <- runif(21,range_min,range_max)
    
    x1 <- runif(B*N,-1,1)
    x2 <- runif(B*N,-1,1)
    x3 <- runif(B*N,-1,1)
    x4 <- runif(B*N,-1,1)
    x5 <- runif(B*N,-1,1)
    
    x6 <- runif(B*N,-1,1)
    x7 <- runif(B*N,-1,1)
    x8 <- runif(B*N,-1,1)
    x9 <- runif(B*N,-1,1)
    x10 <- runif(B*N,-1,1)
    
    x11 <- runif(B*N,-1,1)
    x12 <- runif(B*N,-1,1)
    x13 <- runif(B*N,-1,1)
    x14 <- runif(B*N,-1,1)
    x15 <- runif(B*N,-1,1)
    
    x16 <- runif(B*N,-1,1)
    x17 <- runif(B*N,-1,1)
    x18 <- runif(B*N,-1,1)
    x19 <- runif(B*N,-1,1)
    x20 <- runif(B*N,-1,1)
    
    # First layer
    a_1.1 <- log( 1 + exp( alpha_1.1[1] + alpha_1.1[2]*x1 + alpha_1.1[3]*x2 + alpha_1.1[4]*x3 + alpha_1.1[5]*x4 + alpha_1.1[6]*x5 + alpha_1.1[7]*x6 + alpha_1.1[8]*x7 + alpha_1.1[9]*x8 + alpha_1.1[10]*x9 + alpha_1.1[11]*x10 + alpha_1.1[12]*x11 + alpha_1.1[13]*x12 + alpha_1.1[14]*x13 + alpha_1.1[15]*x14 + alpha_1.1[16]*x15 + alpha_1.1[17]*x16 + alpha_1.1[18]*x17 + alpha_1.1[19]*x18 + alpha_1.1[20]*x19 + alpha_1.1[21]*x20 ) )
    a_2.1 <- log( 1 + exp( alpha_2.1[1] + alpha_2.1[2]*x1 + alpha_2.1[3]*x2 + alpha_2.1[4]*x3 + alpha_2.1[5]*x4 + alpha_2.1[6]*x5 + alpha_2.1[7]*x6 + alpha_2.1[8]*x7 + alpha_2.1[9]*x8 + alpha_2.1[10]*x9 + alpha_2.1[11]*x10 + alpha_2.1[12]*x11 + alpha_2.1[13]*x12 + alpha_2.1[14]*x13 + alpha_2.1[15]*x14 + alpha_2.1[16]*x15 + alpha_2.1[17]*x16 + alpha_1.1[18]*x17 + alpha_2.1[19]*x18 + alpha_2.1[20]*x19 + alpha_2.1[21]*x20 ) )
    a_3.1 <- log( 1 + exp( alpha_3.1[1] + alpha_3.1[2]*x1 + alpha_3.1[3]*x2 + alpha_3.1[4]*x3 + alpha_3.1[5]*x4 + alpha_3.1[6]*x5 + alpha_3.1[7]*x6 + alpha_3.1[8]*x7 + alpha_3.1[9]*x8 + alpha_3.1[10]*x9 + alpha_3.1[11]*x10 + alpha_3.1[12]*x11 + alpha_3.1[13]*x12 + alpha_3.1[14]*x13 + alpha_3.1[15]*x14 + alpha_3.1[16]*x15 + alpha_3.1[17]*x16 + alpha_3.1[18]*x17 + alpha_3.1[19]*x18 + alpha_3.1[20]*x19 + alpha_3.1[21]*x20 ) )
    
    alpha_1.2 <- runif(4,range_min,range_max)
    alpha_2.2 <- runif(4,range_min,range_max)
    alpha_3.2 <- runif(4,range_min,range_max)
    
    # Second layer
    a_1.2 <- log( 1 + exp( alpha_1.2[1] + alpha_1.2[2]*a_1.1 + alpha_1.2[3]*a_2.1 + alpha_1.2[4]*a_3.1 ) )
    a_2.2 <- log( 1 + exp( alpha_2.2[1] + alpha_2.2[2]*a_1.1 + alpha_2.2[3]*a_2.1 + alpha_2.2[4]*a_3.1 ) )
    a_3.2 <- log( 1 + exp( alpha_3.2[1] + alpha_3.2[2]*a_1.1 + alpha_3.2[3]*a_2.1 + alpha_3.2[4]*a_3.1 ) )
    
    # y = third layer
    y <- beta_0_thru_3[1] + beta_0_thru_3[2]*a_1.2 + beta_0_thru_3[3]*a_2.2 + beta_0_thru_3[4]*a_3.2 + epsilon
  }
  
  # Selecting sample A ----
  # Sample A is just a random n=500 sample from 10000 0's (see id0 line below in fsA())
  M  <- rep(1:N,B)
  M2 <- matrix(M,B,N,byrow=T)
  
  # Function for SRS ----
  fsA <- function(a_large_matrix_row){            # That is, a large matrix row of integers: 1:10000
    sI0      <- rep(0,length(a_large_matrix_row)) # A vector of zeros
    id0      <- sample(a_large_matrix_row,nA)     # That way this is a random index, get it?
    sI0[id0] <- 1                                 # So the vector of zeros can be indexed randomly!
    return(sI0)
  }
  
  # This is where M2 is passed to fsA() one row at a time
  MsIA0 <- apply(M2,1,fsA)
  sIA   <- as.numeric(MsIA0) # This is the first time sample A indicator shows up! 9.5 million 0's & .5 million 1's.
  # Since nA=500, there are 500,000 1's in sIA, because of the fact that there are 1,000 matrices.
  w     <- rep((N/nA),(B*N))

  sampling_method="SRS"
  sampling_method="PPS"
  if(sampling_method=="PPS"){
  s<-0.5*rchisq(B*N,1)+1

  fPi<-function(zz){
    res_Pi<-nA*zz/sum(zz)
    return(res_Pi)
  }

  Ms<-matrix(s,B,N,byrow=T)
  Ms2<-apply(Ms,1,fPi)
  Pi<-as.numeric(Ms2)
  Ms3<-apply(Ms2,2,UPrandomsystematic)
  sIA<-as.numeric(Ms3)
  w<-1/Pi
  }
  # Selecting sample B ----
  # WE USED STRATIFIED RANDOM SAMPLING BEFORE, THAT'S WHY GAM PERFORMED SO WELL. THAT SRS MECHANISM
  # DID NOT GENERATE INFORMATION THAT WAS "INFORMATIVE" ENOUGH. THIS TIME THE BERNOULLI PROCESS
  # FOR SELECTING THE B SAMPLE SHOULD REVEAL THE SUPERIORITY OF THE OTHER ML METHODS OVER GAM.
  # REALLY?????? WHY WOULD THE METHOD FOR SELECTING B AFFECT THE PERFORMANCE OF THE ML METHODS SO
  # MUCH???
  # Sample B is also just a simple random sample (though it does have two strata, n1, n2, within it)--see
  # samp1 and samp2 lines below in fsrs()
  # selecting sample B
  
  if (id_m %in% c(1,2,3)){
    ## M1-M3 sample B ----
    ### Experiment with c = -4, something else, to get a good proportion of the data--about 500.
    ## C=-4.4 => nb=500   C=-3.4 => nb=1000
    C=-4.4
    p   <- exp(C + x1+x2+x3+x4)/(1+exp(C + x1+x2+x3+x4))
    sIB <- rbinom(B*N,1,p) # 'p' is only used here. sIB is obviously used below.
  }
 

  #MsIA0 <- apply(M2,1,fsA)
  #sIB   <- as.numeric(MsIA0)

 
  if (id_m == 4){
    rho <- runif(3,min = 1,max = 3)
    ## M4 sample B ----
    ### Experiment with c = 1, etc. until you get sample B's n = ~500.
    p   <- exp(C + rho[1]*a_1.1+rho[2]*a_2.1+rho[3]*a_3.1) / 
      ( 1 + exp(C + rho[1]*a_1.1+rho[2]*a_2.1+rho[3]*a_3.1) )
    sIB <- rbinom(B*N,1,p)

    #C=-3.4 #for x1-x4
    C=-5.2
    cpv=x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20
    p   <- exp(C + cpv)/(1+exp(C + cpv))
    sIB <- rbinom(B*N,1,p) # 'p' is only used here. sIB is obviously used below.


  }
  
  
   
  # This is where a3 is passed to fsrs one matrix at a time
  # MsIB0 <- apply(a3,3,fsrs)  # MsIB0 is a thousand 10000x1000 matrices of 0's & 1's
  # sIB   <- as.numeric(MsIB0) # This is the first time sample B indicator shows up! 9 million 0's & 1 million 1's!
  
  # RESUME ----
  # 'w' has nothing to do with the stratified random sample above; 'w' will be used just below where it says 'Mw'
  #w     <- rep((N/nA),(B*N))
  
  if (id_m %in% c(1,2,3)) {
    # Create the matrix of 7 columns of simulated data! Recall that these are 10 million elements each.
    My   <- matrix(y,B,N,byrow=T)
    Mx1  <- matrix(x1,B,N,byrow=T)
    Mx2  <- matrix(x2,B,N,byrow=T)
    
    # ADD Mx3 ----
    Mx3  <- matrix(x3,B,N,byrow=T)
    
    # ... ADD Mx4 ----
    Mx4  <- matrix(x4,B,N,byrow=T)
    
    MsIA <- matrix(sIA,B,N,byrow=T)
    MsIB <- matrix(sIB,B,N,byrow=T)
    Mw   <- matrix(w,B,N,byrow=T)
    
    M   <- cbind(My,  # will become dat[,1]
                 Mx1, # will become dat[,2]
                 Mx2, # will become dat[,3]
                 
                 # ADD Mx3 ----
                 Mx3, # will become dat[,4]
                 
                 # ... ADD Mx4  ----
                 Mx4, # will become dat[,5]
                 
                 MsIA,# will become dat[,6]
                 MsIB,# will become dat[,7]
                 Mw)  # will become dat[,8] 
    M   <- t(M)
    aM  <- as.numeric(M)
    
    # ... ADD 8 to Res so Mx4 fits ----
    # If you take 10 million elements and make 1000 matrices out of it, there will be 10,000 rows in each matrix.
    Res <- array(aM,c(N,8,B))
  } 
  
  if (id_m == 4) {
    # Create the matrix of 7 columns of simulated data! Recall that these are 10 million elements each.
    My   <- matrix(y,B,N,byrow=T)
    Mx1  <- matrix(x1,B,N,byrow=T)
    Mx2  <- matrix(x2,B,N,byrow=T)
    
    # ADD Mx3 ----
    Mx3  <- matrix(x3,B,N,byrow=T)
    
    # ... ADD 4.thru.21 ----
    Mx4 <- matrix(x4,B,N,byrow=T)
    Mx5 <- matrix(x5,B,N,byrow=T)
    Mx6 <- matrix(x6,B,N,byrow=T)
    Mx7 <- matrix(x7,B,N,byrow=T)
    Mx8 <- matrix(x8,B,N,byrow=T)
    Mx9 <- matrix(x9,B,N,byrow=T)
    Mx10 <- matrix(x10,B,N,byrow=T)
    Mx11 <- matrix(x11,B,N,byrow=T)
    Mx12 <- matrix(x12,B,N,byrow=T)
    Mx13 <- matrix(x13,B,N,byrow=T)
    Mx14 <- matrix(x14,B,N,byrow=T)
    Mx15 <- matrix(x15,B,N,byrow=T)
    Mx16 <- matrix(x16,B,N,byrow=T)
    Mx17 <- matrix(x17,B,N,byrow=T)
    Mx18 <- matrix(x18,B,N,byrow=T)
    Mx19 <- matrix(x19,B,N,byrow=T)
    Mx20 <- matrix(x20,B,N,byrow=T)
    
    MsIA <- matrix(sIA,B,N,byrow=T)
    MsIB <- matrix(sIB,B,N,byrow=T)
    Mw   <- matrix(w,B,N,byrow=T)
    
    M   <- cbind(My,   # will become dat[,1]
                 Mx1,  # will become dat[,2]
                 Mx2,  # will become dat[,3]
                 
                 # ADD Mx3 ----
                 Mx3,  # will become dat[,4]
                 
                 # ... ADD 4.thru.21 ----
                 Mx4,  # will become dat[,5]
                 Mx5,  # will become dat[,6]
                 Mx6,  # will become dat[,7]
                 Mx7,  # will become dat[,8]
                 Mx8,  # will become dat[,9]
                 Mx9,  # will become dat[,10]
                 Mx10, # will become dat[,11]
                 Mx11, # will become dat[,12]
                 Mx12, # will become dat[,13]
                 Mx13, # will become dat[,14]
                 Mx14, # will become dat[,15]
                 Mx15, # will become dat[,16]
                 Mx16, # will become dat[,17]
                 Mx17, # will become dat[,18]
                 Mx18, # will become dat[,19]
                 Mx19, # will become dat[,20]
                 Mx20, # will become dat[,21]
                 
                 MsIA, # will become dat[,22]
                 MsIB, # will become dat[,23]
                 Mw)   # will become dat[,24] 
    M   <- t(M)
    aM  <- as.numeric(M)
    
    # ... ADD 24 to Res so Mx4.thru.21 fits ----
    # If you take 10 million elements and make 1000 matrices out of it, there will be 10,000 rows in each matrix.
    Res <- array(aM,c(N,24,B))
  }
  
  # Population mean of Y
  theta0_1 <- mean(y)
  theta0_2 <- median(y)
  
  res <- list(Res,theta0_1,theta0_2)
  
  return(res)
}


FRES <- function(indat,modeling_method = "GAM",id_m){
  
  dat    <- indat[[1]]
  THETA0 <- c(indat[[2]],indat[[3]])
  
  # 1. The sample mean from sample A (Mean A) ----
  #res_MA    <- apply(dat,3,fM_A)
  #bias_MA   <- res_MA - THETA0
  #m_bias_MA <- apply(bias_MA,1,mean)
  #rb_MA     <- m_bias_MA / THETA0
  #var_MA    <- apply(res_MA,1,var)
  #se_MA     <- sqrt(var_MA)
  #rse_MA    <- se_MA / THETA0
  #mse_MA    <- m_bias_MA^2 + var_MA
  #rrmse_MA  <- sqrt(mse_MA) / THETA0
  #Res_MA    <- cbind(rb_MA,rse_MA,rrmse_MA)
  
  # 2. The naive estimator (sample mean) from sample B (Mean B) ----
  #res_MB    <- apply(dat,3,fM_B)
  #bias_MB   <- res_MB - THETA0
  #m_bias_MB <- apply(bias_MB,1,mean)
  #rb_MB     <- m_bias_MB / THETA0
  #var_MB    <- apply(res_MB,1,var)
  #se_MB     <- sqrt(var_MB)
  #rse_MB    <- se_MB / THETA0
  #mse_MB    <- m_bias_MB^2 + var_MB
  #rrmse_MB  <- sqrt(mse_MB) / THETA0
  #Res_MB    <- cbind(rb_MB,rse_MB,rrmse_MB)
  

  
  # 6. The PMIEG ----
  # * ADD f_ML instead of fGAM ----
  res_GAM    <- apply(dat,3,f_ML,
                      modeling_method = modeling_method,
                      id_m = id_m)
  bias_GAM   <- res_GAM - THETA0
  m_bias_GAM <- apply(bias_GAM,1,mean)
  rb_GAM     <- m_bias_GAM / THETA0             # rb_GAM
  var_GAM    <- apply(res_GAM,1,var)
  se_GAM     <- sqrt(var_GAM)
  rse_GAM    <- se_GAM / THETA0                 # rse_GAM
  mse_GAM    <- m_bias_GAM^2 + var_GAM
  rrmse_GAM  <- sqrt(mse_GAM) / THETA0          # rrmse_GAM
  Res_GAM    <- cbind(rb_GAM,rse_GAM,rrmse_GAM) # Res_GAM
  
  RES <- rbind(
    #Res_MA,
    #Res_MB,
    Res_GAM
  ) 
  RES <- round(RES,4)
  
  rownames(RES) <- c(
    #c('Mean_A','Median_A'),
    #c('Mean_B','Median_B'),
    # c('Mean_P','Domain Mean_P'),
    # c('Mean_PW','Domain Mean_PW'),
    # c('Mean_KS','Domain Mean_KS'),
    c('Mean_ML','Median_ML')
  )
  
  colnames(RES) <- c('RB','RSE','RRMSE')
  
  return(RES)
}
