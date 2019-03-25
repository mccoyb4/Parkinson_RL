
# McCoy et al. 2019 #
# Code for lauching ShinyStan to view OpAL model fit #

install.packages("rstanarm")
library("rstanarm")

# Set this to appropriate directory in which .Rdat model output resides
extout= '~/Documents/PD_Github/Review/OpAL_evaluation' # ouput stan 

# Load fitted model
setwd(extout)
load('RL_bg_bl_ac_ag_al_OpAL_ch3it2000wm1000th1ad0.85ss1mt12N46.Rdat')

my_sso <- launch_shinystan(STAN)
