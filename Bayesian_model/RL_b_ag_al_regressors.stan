
// B. McCoy, 2017
// Stan script for within- and between-subjects reinforcement learning modelling of Parkinson's data 
// Free parameters: explore-exploit (beta), positive feedback learning (alpha gain), negative feedback learning (alpha loss)
// Informed by Jahfari et al. 2016 and Sharp et al. 2016

data{
  int<lower=1> n_s;                           // total # of subjects
  int<lower=1> n_s_pd;                        // # of Parkinson patients
  int<lower=1> n_s_hc;                        // # of healthy controls
  int<lower=1> n_t;                           // # of trials 
  int<lower=1,upper=5> Choice[n_t];           // choice options trial n_t (choice is 1,3 of 5) (all subjects)
  int<lower=1,upper=2> Correct[n_t];          // correct (=1, yes-correct) for trial n_t
  int<lower=0,upper=1> Reward[n_t];           // reward (=1, yes) for trial n_t
  int<lower=1,upper=n_s> Subject[n_t];        // subject number (n_s)
  int<lower=0,upper=1> Init[n_t];             // is this first trial of a subject? Should RL be initialized?

  // Binary indicators for Disease and Medication status
  // Applying these to all beta, ag, and al parameters
  int<lower=0,upper=1> Disease[n_t];
  int<lower=0,upper=1> Medication[n_t];

}// end data

parameters{
  // group level mean parameters
  real mu_b_pr;			    //beta (inverse gain)
  real mu_ag_pr;		    //ag
  real mu_al_pr;   		  //al

  // med and disease group level indicators
  real k_b_med_pr;      //med beta 
  real k_ag_med_pr;     //med ag 
  real k_al_med_pr;     //med al

  real k_b_dis_pr;      //disease beta 
  real k_ag_dis_pr;     //disease ag 
  real k_al_dis_pr;     //disease al 

  // group level standard deviation
  real<lower=0> sd_b;   		//beta
  real<lower=0> sd_ag;   		//ag
  real<lower=0> sd_al;   		//al

  real<lower=0> sd_k_b_med;   //med beta
  real<lower=0> sd_k_ag_med;  //med ag
  real<lower=0> sd_k_al_med;  //med al  
  
  // individual level parameters
  vector[n_s] b_ind_pr[2];   			 
  vector[n_s] ag_ind_pr[2];   			 
  vector[n_s] al_ind_pr[2];   			 

  real k_b_med_ind_pr[n_s_pd];    
  real k_ag_med_ind_pr[n_s_pd];    
  real k_al_med_ind_pr[n_s_pd];     

}//end paramters
	

transformed parameters{
  // group level mean parameters
  real<lower=0,upper=100> mu_b; 				
  real<lower=0,upper=1> mu_ag;  
  real<lower=0,upper=1> mu_al;  		

  real<lower=-5,upper=5> k_b_med;
  real<lower=-5,upper=5> k_ag_med;
  real<lower=-5,upper=5> k_al_med;

  real<lower=-5,upper=5> k_b_dis;
  real<lower=-5,upper=5> k_ag_dis;
  real<lower=-5,upper=5> k_al_dis;

  // individual level parameters
  vector<lower=0,upper=100>[n_s] b_ind[2];     	
  vector<lower=0,upper=1>[n_s] ag_ind[2];   		
  vector<lower=0,upper=1>[n_s] al_ind[2];   		

  real<lower=-5,upper=5> k_b_med_ind[n_s_pd];
  real<lower=-5,upper=5> k_ag_med_ind[n_s_pd]; 
  real<lower=-5,upper=5> k_al_med_ind[n_s_pd];

  // group level mean parameters (probit)
  mu_b  <-Phi_approx(mu_b_pr) * 100;   	
  mu_ag <-Phi_approx(mu_ag_pr);   			
  mu_al <-Phi_approx(mu_al_pr);   			

  k_b_med <-Phi_approx(k_b_med_pr) * 10 - 5; 
  k_ag_med <-Phi_approx(k_ag_med_pr) * 10 - 5;
  k_al_med <-Phi_approx(k_al_med_pr) * 10 - 5;

  k_b_dis <-Phi_approx(k_b_dis_pr) * 10 - 5; 
  k_ag_dis <-Phi_approx(k_ag_dis_pr) * 10 - 5; 
  k_al_dis <-Phi_approx(k_al_dis_pr) * 10 - 5; 

  // Individual level parameters (probit)

  // PD loop
  for (s in 1:n_s_pd){

    k_b_med_ind[s]  <- Phi_approx(k_b_med_pr + sd_k_b_med * k_b_med_ind_pr[s]) * 10 - 5;    
    k_ag_med_ind[s]  <- Phi_approx(k_ag_med_pr + sd_k_ag_med * k_ag_med_ind_pr[s]) * 10 - 5;   
    k_al_med_ind[s]  <- Phi_approx(k_al_med_pr + sd_k_al_med * k_al_med_ind_pr[s]) * 10 - 5;   

    for (m in 1:2){ // within-subject medication session

      if (m==1){    // off med
        b_ind[m,s] <- Phi_approx(mu_b_pr + sd_b * b_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + sd_al * al_ind_pr[m,s]); 
      }

      else{       // on med
        b_ind[m,s] <- Phi_approx(mu_b_pr + k_b_med_ind[s] + sd_b * b_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + k_ag_med_ind[s] + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + k_al_med_ind[s] + sd_al * al_ind_pr[m,s]); 
      }
    }
  }

  // HC loop (m = 1 and 2, but only m=1 is updated in the model since they are off medication)
  
  for (s in (n_s_pd+1):(n_s_pd+n_s_hc)){

    for (m in 1:2){

      if (m==1){
        b_ind[m,s] <- Phi_approx(mu_b_pr + k_b_dis_pr + sd_b * b_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + k_ag_dis_pr + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + k_al_dis_pr + sd_al * al_ind_pr[m,s]);

      }
      else{
        b_ind[m,s] <- Phi_approx(mu_b_pr + sd_b * b_ind_pr[m,s]) * 100;
        ag_ind[m,s] <- Phi_approx(mu_ag_pr + sd_ag * ag_ind_pr[m,s]);
        al_ind[m,s] <- Phi_approx(mu_al_pr + sd_al * al_ind_pr[m,s]); 
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
  real epsilon;
  int a;
  int med;
  real alpha;
  vector[2] pchoice;
  epsilon <- 0.00001;

  // set prior on group level mean parameters
  mu_b_pr ~  normal(0,1);   			 
  mu_ag_pr ~ normal(0,1);   			
  mu_al_pr ~ normal(0,1);   				
  
  k_b_med_pr ~ normal(0,1);    
  k_ag_med_pr ~ normal(0,1);       
  k_al_med_pr ~ normal(0,1);     

  k_b_dis_pr ~ normal(0,1);
  k_ag_dis_pr ~ normal(0,1);
  k_al_dis_pr ~ normal(0,1);
 

  // set prior on group level standard deviations
  sd_b ~  cauchy(0, 5);     	
  sd_ag ~ cauchy(0, 5);   		
  sd_al ~ cauchy(0, 5);  			

  sd_k_b_med ~ cauchy(0, 5);      
  sd_k_ag_med ~ cauchy(0, 5);    
  sd_k_al_med ~ cauchy(0, 5);  

  // set prior for individual level parameters

  for (s in 1:n_s){

    b_ind_pr[1,s]~ normal(0,1);           // off med / hc
    ag_ind_pr[1,s]~ normal(0,1);   		    
    al_ind_pr[1,s]~ normal(0,1);   			 

    b_ind_pr[2,s]~ normal(0,1);           // on med
    ag_ind_pr[2,s]~ normal(0,1);          
    al_ind_pr[2,s]~ normal(0,1);           

  } 

  k_b_med_ind_pr ~ normal(0,1);   //vectorization
  k_ag_med_ind_pr ~ normal(0,1); 
  k_al_med_ind_pr ~ normal(0,1); 
  
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

              }// end inital values loop
          // trial 1
          pchoice[1]<-0.5;
          pchoice[2]<-0.5;
        }

          med = Medication[t]+1; //coded as 0,1 in file. Changing to 1,2 for indexing here. 1 = off med/hc, 2 = on.
   
          Qchoice[1]    <- prQ[Choice[t]]; 
          Qchoice[2]    <- prQ[(Choice[t]+1)];
          pchoice[1]    <- 1/(1+exp(b_ind[med,si]*(Qchoice[2]-Qchoice[1])));
          pchoice[2]    <- 1-pchoice[1];
          pchoice[1]    <- epsilon/2+(1-epsilon)*pchoice[1];
          pchoice[2]    <- epsilon/2+(1-epsilon)*pchoice[2];

          Correct[t]~categorical(pchoice);
          a <- Correct[t]-1; //0=correct,1=incorrect	

          // reinforcement
          alpha <- Reward[t]*ag_ind[med,si]+(1-Reward[t])*al_ind[med,si];
          prQ[(Choice[t]+a)] <- prQ[(Choice[t]+a)] + alpha*(Reward[t]-prQ[(Choice[t]+a)]);

 
   }// end subject loop
}// end of model loop

generated quantities{

  // See Github code from Ahn Young for similar setup : https://github.com/youngahn/hBayesDM/blob/master/exec/prl_rp.stan 
  
  real log_lik[n_s];        // For log likelihood calculation (on a per subject basis)
  real RPE[n_t];            // For reward prediction error calculation (on a subject and per trial basis)
  real Qval_chosen[n_t];    // For cue-value of chosen stimulus
  real Qval_unchosen[n_t];  // For cue-value of unchosen stimulus

  real Qval_update[n_t];    // For updated Q-value of chosen stimulus
  real RPE_update[n_t];     // For the weighted RPE update of chosen stimulus

  { // local section, saves time and space

    // Setting up variables for trial-by-trial modelling, log_lik and RPE calculations.
    int si;
    real prQ0[6];
    real prQ[6];
    real Qchoice[2];
    real epsilon;
    int a;
    int b;
    int med;
    real alpha;
    vector[2] pchoice;
    epsilon <- 0.00001;
    
    for (t in 1:n_t) {

      // Almost the same setup as in model{} block, just with log_lik and RPE calculations added (& trial-to-trial regressors for Q-values).

      if (Init[t]==1){
        si<- Subject[t];
        for (v in 1:6)
        {
          prQ0[v] <- 0.5;
          prQ[v] <- 0.5;

        }// end inital values 

        // trial 1
        pchoice[1]<-0.5;
        pchoice[2]<-0.5;

        # Initialise log_lik to 0 for each participant        
        log_lik[si] = 0;
      }

      med = Medication[t]+1; //coded as 0,1 in file. Changing to 1,2 for indexing here. 1 = off med/hc, 2 = on.

      Qchoice[1]    <- prQ[Choice[t]]; 
      Qchoice[2]    <- prQ[(Choice[t]+1)];
      pchoice[1]    <- 1/(1+exp(b_ind[med,si]*(Qchoice[2]-Qchoice[1])));
      pchoice[2]    <- 1-pchoice[1];
      pchoice[1]    <- epsilon/2+(1-epsilon)*pchoice[1];
      pchoice[2]    <- epsilon/2+(1-epsilon)*pchoice[2];


      // Log likelihood of the softmax choice. Gets updated on the subject level, given trial-to-trial choices.
      log_lik[si] = log_lik[si] + categorical_lpmf( Correct[t] | pchoice);

      a <- Correct[t]-1; //0=correct,1=incorrect  
      
      // Model regressors - store values before being updated

      if (a==0){
        b=1;
      }
      else{
        b=0;
      }

      RPE[t] =  Reward[t]-prQ[(Choice[t]+a)];  // Reward prediction error
      Qval_chosen[t] = prQ[(Choice[t]+a)];     // Q-value chosen stimulus
      Qval_unchosen[t] = prQ[(Choice[t]+b)];   // Q-value unchosen stimulus


      // Reinforcement - update Q-values based on current outcome
      alpha <- Reward[t]*ag_ind[med,si]+(1-Reward[t])*al_ind[med,si];
      prQ[(Choice[t]+a)] <- prQ[(Choice[t]+a)] + alpha * (Reward[t]-prQ[(Choice[t]+a)]); // last part is the RPE

      Qval_update[t] = prQ[(Choice[t]+a)];
      RPE_update[t] = alpha * (Reward[t]-prQ[(Choice[t]+a)]);

   }
  }  
}








