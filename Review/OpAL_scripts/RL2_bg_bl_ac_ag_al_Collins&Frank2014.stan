// RL_stan.stan script updated to use within- and between-subjects modelling as described in Sharp et al. (2016)

data{
  int<lower=1> n_s;                           // total # of subjects
  int<lower=1> n_s_pd;                        // # of Parkinson patients
  int<lower=1> n_s_hc;                        // # of healthy controls
  int<lower=1> n_t;                           // # of trials 
  int<lower=1,upper=5> Choice[n_t];           // choice options trial n_t (choice is 1,3 of 5) (All subjects)
  int<lower=1,upper=2> Correct[n_t];          // correct (=1, yes-correct)? trial n_t
  int<lower=0,upper=1> Reward[n_t];           // reward (=1, yes)? trial n_t
  int<lower=1,upper=n_s> Subject[n_t];        // subject number (n_s)
  int<lower=0,upper=1> Init[n_t];             // is this first trial of a subject? Should RL be initialized?

  // new binary indicators for disease and medication status
  // applying these to all beta, ag, & al parameters
  int<lower=0,upper=1> Disease[n_t];
  int<lower=0,upper=1> Medication[n_t];

}// end data

parameters{
  // group level mean parameters
  real mu_bg_pr;			  //betaG parameter
  real mu_bl_pr;        //betaL parameter
  real mu_ag_pr;		    //alphaG
  real mu_al_pr;   		  //alphaL
  real mu_ac_pr;        //alphaC

  // med and disease group level indicators
  real k_bg_med_pr;     //medication betaG param
  real k_bl_med_pr;     //medication betaL param
  real k_ag_med_pr;     //med alphaG param
  real k_al_med_pr;     //med alphaL param
  real k_ac_med_pr;     //med alphaC param

  real k_bg_dis_pr;      //disease betaG param
  real k_bl_dis_pr;      //disease betaL param
  real k_ag_dis_pr;     //disease alphaG param
  real k_al_dis_pr;     //disease alphaL param
  real k_ac_dis_pr;     //disease alphaC param

  // group level standard deviation
  real<lower=0> sd_bg;   		//betaG parameter
  real<lower=0> sd_bl;      //betaL parameter
  real<lower=0> sd_ag;   		//alphaG
  real<lower=0> sd_al;   		//alphaL
  real<lower=0> sd_ac;      //alphaC

  real<lower=0> sd_k_bg_med;   //med betaG
  real<lower=0> sd_k_bl_med;   //med betaL
  real<lower=0> sd_k_ag_med;  //med ag
  real<lower=0> sd_k_al_med;  //med al  
  real<lower=0> sd_k_ac_med;  //med ac
  
  // individual level parameters
  vector[n_s] bg_ind_pr[2];   			  //individual betaG parameter across medication status
  vector[n_s] bl_ind_pr[2];  
  vector[n_s] ag_ind_pr[2];   			  //alphaG
  vector[n_s] al_ind_pr[2];   			  //alphaL
  vector[n_s] ac_ind_pr[2];           //alphaC

  real k_bg_med_ind_pr[n_s_pd];   
  real k_bl_med_ind_pr[n_s_pd];  
  real k_ag_med_ind_pr[n_s_pd];    
  real k_al_med_ind_pr[n_s_pd];   
  real k_ac_med_ind_pr[n_s_pd];   

}//end paramters
	

transformed parameters{
  // group level mean parameters
  real<lower=0,upper=100> mu_bg; 				//betaG parameter
  real<lower=0,upper=100> mu_bl;        //betaL parameter
  real<lower=0,upper=1> mu_ag;   				//alphaG
  real<lower=0,upper=1> mu_al;   				//alphaL
  real<lower=0,upper=1> mu_ac;          //alphaC

  real<lower=-5,upper=5> k_bg_med;
  real<lower=-5,upper=5> k_bl_med;
  real<lower=-5,upper=5> k_ag_med;
  real<lower=-5,upper=5> k_al_med;
  real<lower=-5,upper=5> k_ac_med;

  real<lower=-5,upper=5> k_bg_dis;
  real<lower=-5,upper=5> k_bl_dis;
  real<lower=-5,upper=5> k_ag_dis;
  real<lower=-5,upper=5> k_al_dis;
  real<lower=-5,upper=5> k_ac_dis;

  // individual level parameters
  vector<lower=0,upper=100>[n_s] bg_ind[2];     //betaG parameter
  vector<lower=0,upper=100>[n_s] bl_ind[2];     //betaL parameter
  vector<lower=0,upper=1>[n_s] ag_ind[2];   		//alphaG
  vector<lower=0,upper=1>[n_s] al_ind[2];   		//alphaL
  vector<lower=0,upper=1>[n_s] ac_ind[2];       //alphaC

  real<lower=-5,upper=5> k_bg_med_ind[n_s_pd];
  real<lower=-5,upper=5> k_bl_med_ind[n_s_pd];
  real<lower=-5,upper=5> k_ag_med_ind[n_s_pd]; 
  real<lower=-5,upper=5> k_al_med_ind[n_s_pd];
  real<lower=-5,upper=5> k_ac_med_ind[n_s_pd];

  // group level mean parameters (probit)
  mu_bg  <-Phi_approx(mu_bg_pr) * 100;   	//betaG parameter
  mu_bl  <-Phi_approx(mu_bl_pr) * 100;    //betaG parameter
  mu_ag <-Phi_approx(mu_ag_pr);   				//alphaG
  mu_al <-Phi_approx(mu_al_pr);   				//alphaL
  mu_ac <-Phi_approx(mu_ac_pr);

  k_bg_med <-Phi_approx(k_bg_med_pr) * 10 - 5; 
  k_bl_med <-Phi_approx(k_bl_med_pr) * 10 - 5; 
  k_ag_med <-Phi_approx(k_ag_med_pr) * 10 - 5;
  k_al_med <-Phi_approx(k_al_med_pr) * 10 - 5;
  k_ac_med <-Phi_approx(k_ac_med_pr) * 10 - 5;

  k_bg_dis <-Phi_approx(k_bg_dis_pr) * 10 - 5; 
  k_bl_dis <-Phi_approx(k_bl_dis_pr) * 10 - 5; 
  k_ag_dis <-Phi_approx(k_ag_dis_pr) * 10 - 5; 
  k_al_dis <-Phi_approx(k_al_dis_pr) * 10 - 5; 
  k_ac_dis <-Phi_approx(k_ac_dis_pr) * 10 - 5; 

  // Individual level parameters (probit)

  // PD loop
  for (s in 1:n_s_pd){

    k_bg_med_ind[s]  <- Phi_approx(k_bg_med_pr + sd_k_bg_med * k_bg_med_ind_pr[s]) * 10 - 5;  
    k_bl_med_ind[s]  <- Phi_approx(k_bl_med_pr + sd_k_bl_med * k_bl_med_ind_pr[s]) * 10 - 5;    
    k_ag_med_ind[s]  <- Phi_approx(k_ag_med_pr + sd_k_ag_med * k_ag_med_ind_pr[s]) * 10 - 5;   
    k_al_med_ind[s]  <- Phi_approx(k_al_med_pr + sd_k_al_med * k_al_med_ind_pr[s]) * 10 - 5;   
    k_ac_med_ind[s]  <- Phi_approx(k_ac_med_pr + sd_k_ac_med * k_ac_med_ind_pr[s]) * 10 - 5; 

    for (m in 1:2){ // within-subject medication session

      if (m==1){    // off med
        bg_ind[m,s] <- Phi_approx(mu_bg_pr + sd_bg * bg_ind_pr[m,s]) * 100;
        bl_ind[m,s] <- Phi_approx(mu_bl_pr + sd_bl * bl_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + sd_al * al_ind_pr[m,s]); 
        ac_ind[m,s] <- Phi_approx(mu_ac_pr + sd_ac * ac_ind_pr[m,s]); 
      }

      else{       // on med
        bg_ind[m,s] <- Phi_approx(mu_bg_pr + k_bg_med_ind[s] + sd_bg * bg_ind_pr[m,s]) * 100;
        bl_ind[m,s] <- Phi_approx(mu_bl_pr + k_bl_med_ind[s] + sd_bl * bl_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + k_ag_med_ind[s] + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + k_al_med_ind[s] + sd_al * al_ind_pr[m,s]); 
        ac_ind[m,s] <- Phi_approx(mu_ac_pr + k_ac_med_ind[s] + sd_ac * ac_ind_pr[m,s]); 
      }
    }
  }

  // HC loop (m = 1 and 2, but only m=1 is updated in the model since they are off medication)
  
  for (s in (n_s_pd+1):(n_s_pd+n_s_hc)){

    for (m in 1:2){

      if (m==1){
        bg_ind[m,s] <- Phi_approx(mu_bg_pr + k_bg_dis_pr + sd_bg * bg_ind_pr[m,s]) * 100;
        bl_ind[m,s] <- Phi_approx(mu_bl_pr + k_bl_dis_pr + sd_bl * bl_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + k_ag_dis_pr + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + k_al_dis_pr + sd_al * al_ind_pr[m,s]);
        ac_ind[m,s] <- Phi_approx(mu_ac_pr + k_ac_dis_pr + sd_ac * ac_ind_pr[m,s]);

      }
      else{
        bg_ind[m,s] <- Phi_approx(mu_bg_pr + sd_bg * bg_ind_pr[m,s]) * 100;
        bl_ind[m,s] <- Phi_approx(mu_bl_pr + sd_bl * bl_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + sd_al * al_ind_pr[m,s]);
        ac_ind[m,s] <- Phi_approx(mu_ac_pr + sd_ac * ac_ind_pr[m,s]);  
      }      
    }
  }
  
} // end transformed parameters

model{
  // define general variables needed for subject loop
  int si;
  real prQ0[6];
  real prQ[6];
  real Qchoice[2];
  real G_a[6];
  real N_a[6];
  real Act;
  real epsilon;
  int a;
  int med;
  vector[2] pchoice;
  epsilon <- 0.00001;

  // set prior on group level mean parameters
  mu_bg_pr ~  normal(0,1);   			  //betaG parameter
  mu_bl_pr ~  normal(0,1);          //betaL parameter
  mu_ag_pr ~ normal(0,1);   				//alphaG
  mu_al_pr ~ normal(0,1);   				//alphaL
  mu_ac_pr ~ normal(0,1); 
  
  k_bg_med_pr ~ normal(0,1);        //med betaG
  k_bl_med_pr ~ normal(0,1);        //med betaL
  k_ag_med_pr ~ normal(0,1);        //med ag
  k_al_med_pr ~ normal(0,1);        //med al
  k_ac_med_pr ~ normal(0,1);

  k_bg_dis_pr ~ normal(0,1);
  k_bl_dis_pr ~ normal(0,1);
  k_ag_dis_pr ~ normal(0,1);
  k_al_dis_pr ~ normal(0,1);
  k_ac_dis_pr ~ normal(0,1);
 

  // set prior on group level standard deviations
  sd_bg ~  cauchy(0, 5);     		  //betaG parameter
  sd_bl ~  cauchy(0, 5);          //betaL parameter
  sd_ag ~ cauchy(0, 5);   				//alphaG
  sd_al ~ cauchy(0, 5);  				  //alphaL
  sd_ac ~ cauchy(0, 5); 

  sd_k_bg_med ~ cauchy(0, 5);       //med betaG
  sd_k_bl_med ~ cauchy(0, 5);       //med betaL
  sd_k_ag_med ~ cauchy(0, 5);       //med ag
  sd_k_al_med ~ cauchy(0, 5);       //med al
  sd_k_ac_med ~ cauchy(0, 5);

  // set prior for individual level parameters

  for (s in 1:n_s){

    bg_ind_pr[1,s]~ normal(0,1);           // off med / hc
    bl_ind_pr[1,s]~ normal(0,1);
    ag_ind_pr[1,s]~ normal(0,1);   		    
    al_ind_pr[1,s]~ normal(0,1);  
    ac_ind_pr[1,s]~ normal(0,1);  			 

    bg_ind_pr[2,s]~ normal(0,1);           // on med
    bl_ind_pr[2,s]~ normal(0,1); 
    ag_ind_pr[2,s]~ normal(0,1);          
    al_ind_pr[2,s]~ normal(0,1);  
    ac_ind_pr[2,s]~ normal(0,1);         

  } 

  k_bg_med_ind_pr ~ normal(0,1);   //vectorization
  k_bl_med_ind_pr ~ normal(0,1);
  k_ag_med_ind_pr ~ normal(0,1); 
  k_al_med_ind_pr ~ normal(0,1); 
  k_ac_med_ind_pr ~ normal(0,1); 
  
  // now start looping over subjects and trials
  for (t in 1:n_t)
  {
      // set initial values per subject and medication session. Resets values for each medication session while still retaining same subject number (within-subject manipulation).

      if (Init[t]==1){
            si<- Subject[t];
            for (v in 1:6)
              {
                prQ0[v] <- 0.5;
                prQ[v] <- 0.5;
                G_a[v] <- 1; // Chance level, according to appendix in Collins & Frank 2014
                N_a[v] <- 1; 

              }// end inital values loop
          // trial 1
          pchoice[1]<-0.5;
          pchoice[2]<-0.5;
        }

          med = Medication[t]+1; //coded as 0,1 in file. Changing to 1,2 for indexing here. 1 = off med/hc, 2 = on.

          // Choice parameter OPAL model (Cockburn 2014; Collins & Frank 2014):
          Qchoice[1] <- (bg_ind[med,si]*G_a[Choice[t]]) - (bl_ind[med,si]*N_a[Choice[t]]);
          Qchoice[2] <- (bg_ind[med,si]*G_a[(Choice[t]+1)]) - (bl_ind[med,si]*N_a[(Choice[t]+1)]);

          pchoice[1]    <- 1/(1+exp(Qchoice[2] - Qchoice[1]));
          pchoice[2]    <- 1-pchoice[1];
          pchoice[1]    <- epsilon/2+(1-epsilon)*pchoice[1];
          pchoice[2]    <- epsilon/2+(1-epsilon)*pchoice[2];

          Correct[t]~categorical(pchoice);
          a <- Correct[t]-1; //0=correct,1=incorrect  
          
          // Critic component (see Eq. 1 in Collins & Frank 2014)
          prQ[(Choice[t]+a)] <- prQ[(Choice[t]+a)] + (ac_ind[med,si]*(Reward[t]-prQ[(Choice[t]+a)]));

          // Actor component (see Eqs. 2 & 3 in Collins & Frank 2014)

          // Go pathway
          G_a[(Choice[t]+a)] <- G_a[(Choice[t]+a)] + ((ag_ind[med,si]*G_a[(Choice[t]+a)])*(Reward[t]-prQ[(Choice[t]+a)]));

          // No-Go pathway
          N_a[(Choice[t]+a)] <- N_a[(Choice[t]+a)] + ((al_ind[med,si]*N_a[(Choice[t]+a)])*(-1*(Reward[t]-prQ[(Choice[t]+a)])));

 
   }// end subject loop
}// end of model loop








