
1. Requirements:
   - Download RStudio (download from https://www.rstudio.com/products/rstudio/download/)
   - rstanarm package (installed and loaded in Open_model_ShinyStan.R script)
   - Download the model output .Rdat file, located on figure at https://doi.org/10.6084/m9.figshare.7887032.v1. 
     Too large to upload on Github.

2. Change extout directory to point to model output directory.

3. Run Open_model_ShinyStan.R. This launches a web-based application.

4. Click on 'Diagnose' tab at top. This provides 5 sub-tabs: 'NUTS(plots)', 'HMC/NUTS(stats)', 'R_hat,n_eff,se_mean', 
   'Autocorrelatio'n', and 'PPcheck'.

5. Under NUTS(plots), if 'All chains' = 0, this shows all sampling chains. Otherwise, 1-3 can be chosen. 

6. Under NUTS(plots), 'Parameter' can be used to load several diagnostics per parameter. Parameters of interest are 
   the medication and disease difference parameters; this all begin with the letter 'k_'. On the group level, these 
   are: k_bg_med, k_bl_med, k_ag_med, k_al_med, and k_ac_med for medication differences for the 5 free parameters, and 
   '_dis' in place of '_med' for the equivalent disease differences.

7. In particular, medication differences in the alpha_gain parameter (k_ag_med) show chain sampling issues 
   (in the top left plot) since they continue to move around the whole parameter space without convergencing across 
   samples. The second row of plots shows divergent transitions as red dots (146 here). Divergent transitions indicate 
   that sampler misses certain features of the target distribution and returns biased estimates.

8. Under the main 'R_hat,n_eff,se_mean' tab, there are three diagnostics of the model fit. The threshold for these can 
   be adjusted on the right and is automatically set at a default level to indicate problems. The long lists of 
   parameters under each of the neff/N, mcse/sd, and R_hat columns shows that there were issues in estimating these 
   parameters according to the diagnostic of interest. In particular, the R_hat > 1.1 of the log-posterior parameter 
   suggests that the entire model is not an adequate model fit.

9. A separate plot (OpAL_evaluation_plot.pdf) included in the Github folder shows a brief overview of some of these 
   issues, with an example of non-uniform multi-peaked distributions for individual alpha_gain medication differences 
   parameters (k_ag_med_ind).