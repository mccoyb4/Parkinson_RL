#---------------------------------------------------#
# B. McCoy 2017, based on Jahfari et al. 2016.
# R script calling on Stan RL_b_ag_al_regressors.stan for within- and between-subjects reinforcement 
# learning modelling of Parkinson's data.
#---------------------------------------------------#

library(rstan)
library("loo")
rm(list=ls())

datadir='~/Documents/RL2alpha/PARKTrain' # Raw data directory
extdir= '~/Documents/RL2alpha/scripts/STAN' # Stan file directory 
extout= '~/Documents/RL2alpha/Output' # Output directory for Stan .rdat file

# Load data
setwd(datadir)
data=list.files()

toMatch1 <- c("park_pd_on")
ppn1 = unique(grep(paste(toMatch1,collapse="|"),data, value=TRUE))

toMatch2 <- c("park_pd_off")
ppn2 = unique(grep(paste(toMatch2,collapse="|"),data, value=TRUE))

toMatch3 <- c("park_hc")
ppn3=unique(grep(paste(toMatch3,collapse="|"),data, value=TRUE))

# There are 46 subjects in total
run=1:length(ppn1) 
n_t=c()
n_s=length(ppn1)+length(ppn3)

Init=c()
Choice=c()
Reward=c()
Correct=c()
Subject=c()
Medication=c()
Disease=c()

# Note: run goes from 1->23 (pd on and pd off). hc_run goes from 24->46.
# run is used to index hc_run later so that correct subject number is used.

for (r in run)
{
  # PD ON
  setwd(datadir)
  setwd(ppn1[r])
  x1=list.files()
  dat1=read.table(x1[grep('_train',x1)],h=F)
  # c1: stim(1:6), c2: 0=correct, c3: 0=reward
  init=c(1,rep(0,(length(dat1[,1])-1)))
  Init=as.integer(c(Init,init))
  Subject=c(Subject,rep(r,length(dat1[,1])))
  medication=rep(1,(length(dat1[,1]))) # Medication=1 for pd on
  Medication=as.integer(c(Medication,medication))
  disease=rep(0,(length(dat1[,1]))) # Disease=0 for pd on
  Disease=as.integer(c(Disease,disease))
  
  choice = dat1[,1]
  # recode choice into options 1,3, and 5
  choice=ifelse(choice==12,1,choice)
  choice=ifelse(choice==34,3,choice)
  choice=ifelse(choice==56,5,choice)
  Choice=as.integer(c(Choice,choice))
  Correct=as.integer(c(Correct,1+dat1[,2])) # 1=correct, 2=incorrect
  Reward= as.integer(c(Reward,1-dat1[,3])) # 1=reward, 0=noreward
  
  # PD OFF
  setwd(datadir)
  setwd(ppn2[r])
  
  x2=list.files()
  dat2=read.table(x2[grep('_train',x2)],h=F)
  # c1: stim(1:6), c2: 0=correct, c3: 0=reward
  init2=c(1,rep(0,(length(dat2[,1])-1)))
  Init=as.integer(c(Init,init2))
  Subject=c(Subject,rep(r,length(dat2[,1])))
  medication2=rep(0,(length(dat2[,1]))) # Medication=0 for pd off
  Medication=as.integer(c(Medication,medication2))
  disease2=rep(0,(length(dat2[,1]))) # Disease=0 for pd off
  Disease=as.integer(c(Disease,disease2))
  
  choice2 = dat2[,1]
  # recode choice into options 1,3, and 5
  choice2=ifelse(choice2==12,1,choice2)
  choice2=ifelse(choice2==34,3,choice2)
  choice2=ifelse(choice2==56,5,choice2)
  Choice=as.integer(c(Choice,choice2))
  Correct=as.integer(c(Correct,1+dat2[,2])) # 1=correct, 2=incorrect
  Reward= as.integer(c(Reward,1-dat2[,3])) # 1=reward, 0=noreward
}

hc_run=(length(run)+1):(length(run)+length(ppn3)) 
for (r in run)
{
  # HC
  setwd(datadir)
  setwd(ppn3[r])
  
  x3=list.files()
  dat3=read.table(x3[grep('_train',x3)],h=F)
  # c1: stim(1:6), c2: 0=correct, c3: 0=reward
  init3=c(1,rep(0,(length(dat3[,1])-1)))
  Init=as.integer(c(Init,init3))
  Subject=c(Subject,rep(hc_run[r],length(dat3[,1])))
  medication3=rep(0,(length(dat3[,1]))) # Medication=0 for HC
  Medication=as.integer(c(Medication,medication3))
  disease3=rep(1,(length(dat3[,1]))) # Disease=1 for HC (meaning 'healthy')
  Disease=as.integer(c(Disease,disease3))
  
  choice3 = dat3[,1]
  # recode choice into options 1,3, and 5
  choice3=ifelse(choice3==12,1,choice3)
  choice3=ifelse(choice3==34,3,choice3)
  choice3=ifelse(choice3==56,5,choice3)
  Choice=as.integer(c(Choice,choice3))
  Correct=as.integer(c(Correct,1+dat3[,2])) # 1=correct, 2=incorrect
  Reward= as.integer(c(Reward,1-dat3[,3])) # 1=reward, 0=noreward

  setwd(datadir)
}

n_t=length(Correct)
n_s_pd=length(ppn1)
n_s_hc=length(ppn3)

stan_file=list(n_s=n_s,n_s_pd=n_s_pd, n_s_hc=n_s_hc,n_t=n_t,
               Choice=Choice,Correct=Correct,Reward=Reward, Medication=Medication, Disease=Disease, 
               Init=Init,Subject=Subject)

setwd(extout)
write.csv(stan_file,'stan_info.csv')

# go to script dir
setwd(extdir)

# define procedure
IT = 5000 # Number of samples per chain
WM = 2500 # Burn-in
CH= 3 # Number of chains
TH = 1
AD = .85 # adapt_delta, default = 0.8. Change to 0.95 if divergent transitions.
SS = 1 # stepsize, default = 2. Change to 1 if divergent transitions.
STANm='RL2_b_ag_al_regressors.stan' # define file name for the stan model

STAN<-stan(file=STANm,data=stan_file,iter=IT,warmup=WM, chains=CH, cores = 3, thin = TH, control = list(adapt_delta = AD, stepsize = SS))

# save full model
setwd(extout)
save(STAN,file=paste('RL_b_ag_al_regressors_','ch',CH,'it',IT,'wm',WM,'th',TH, 'ad', AD, 'ss',SS, 'N',n_s,'.Rdat',sep=''))

# Calculate BIC from LLH
setwd(extout)
LLH <- extract_log_lik(STAN)

k=3 # 3 free parameters
n_choices_file = read.csv("Nr_choices_per_subject.csv")
n_choices = n_choices_file$nr_trials

BIC = c()
for (i in 1:46){
  BIC <- c(BIC,-2*mean(LLH[,i])+k*log(n_choices[i]))}
mean_BIC = mean(BIC)

BIC=as.data.frame(BIC, col.names=NULL)
write.csv(BIC, 'BICs_b_ag_al.csv')

BIC_whole_model = -2*mean(LLH)+k*log(sum(n_choices))
write.csv(BIC_whole_model, 'BIC_model_b_ag_al.csv')

# AIC
AIC = c()
for (i in 1:46){
  AIC <- c(AIC,-2*mean(LLH[,i]) + 2*k)}
mean_AIC = mean(AIC)

AIC=as.data.frame(AIC, col.names=NULL)
write.csv(AIC, 'AICs_b_ag_al.csv')

AIC_whole_model = -2*mean(LLH) + 2*k
write.csv(AIC_whole_model, 'AIC_model_b_ag_al.csv')

