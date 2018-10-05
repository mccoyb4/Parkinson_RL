
# Based on Github PoldrackLab script: https://github.com/poldracklab/CNP_task_analysis/blob/master/CNP_analysis.py

from nipype.interfaces import fsl
from nipype.algorithms.modelgen import orth
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import Function
from preproc_functions import natural_sort
import nibabel as nib
import pandas as pd
import numpy as np
import glob
import fnmatch
import sys
import os
from IPython import embed as shell

# The following are commandline arguments specified by bash script.
sub_id = str(sys.argv[1])
hc_or_pd = str(sys.argv[2]) # group : 'hc', 'pd'
pd_on_off = str(sys.argv[3]) # 'on', 'off'
smooth = str(sys.argv[4]) # use smoothed output or not? 0 = no smooth, 1 = smooth
ae_or_lisa = str(sys.argv[5]) # 'ae' or 'lisa' server

subjects = [sub_id]

if ae_or_lisa=='lisa':
	run_number = str(sys.argv[6])

if hc_or_pd == 'hc':

	#subjects = ['sub-108','sub-111','sub-112','sub-113','sub-114','sub-115','sub-116','sub-117','sub-118','sub-119','sub-120','sub-121','sub-123','sub-124','sub-126','sub-127','sub-128','sub-129','sub-130','sub-131','sub-132','sub-133']

	# Name directories for input and output
	if ae_or_lisa == 'ae':

		datadir='/home/shared/2016/Parkinson/data/hc'
		fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/hc_syn-sdc'
		if smooth == '0':
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/hc"
		elif smooth == '1':
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/hc"

	elif ae_or_lisa == 'lisa':

		datadir='/nfs/bromccoy/Parkinson/data/hc'
		fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/hc_syn-sdc'
		if smooth == '0':
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/hc"
		elif smooth == '1':
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/hc"

elif hc_or_pd == 'pd':

	# subjects = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-219','sub-220','sub-221','sub-222','sub-223','sub-224']

	# Name directories for input and output
	if ae_or_lisa == 'ae':

		datadir='/home/shared/2016/Parkinson/data/pd'
		if smooth == '0':
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd"
		elif smooth == '1':
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd"

		if pd_on_off == 'on':
			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
		elif pd_on_off == 'off':
			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_off_syn-sdc'

	elif ae_or_lisa == 'lisa':

		datadir='/nfs/bromccoy/Parkinson/data/pd'
		if smooth == '0':
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd"
		elif smooth == '1':
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/pd"

		if pd_on_off == 'on':
			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
		elif pd_on_off == 'off':
			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_off_syn-sdc'

for sb in range(len(subjects)):

	if ae_or_lisa == 'ae':
		if subjects[sb] not in ('sub-203','sub-209', 'sub-213', 'sub-215', 'sub-115'):
			runs = ["1","2"]
		elif subjects[sb] in ('sub-203','sub-209', 'sub-215'):
			runs=["1"]
		elif subjects[sb] in ('sub-213','sub-115'):
			runs=["2"]

	elif ae_or_lisa == 'lisa':
		runs=run_number

	for run_nr in runs:

		if hc_or_pd == 'hc':

			# 5 analysis folders
			sub_rpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"feedback_RPE_ortho")
			sub_arpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"feedback_aRPE_ortho")
			sub_qval_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"feedback_qval_ortho")
			sub_gf_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"gf_RPE_ortho")
			sub_qval_diff_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"feedback_qval_diff_ortho")
			
			sub_stats_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"feedback","stats")
			sub_ev_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"feedback","workflow","l1design")
			events_filename = os.path.join(datadir,subjects[sb],"func",subjects[sb]+"_task-train_run-%s_events.tsv"%(run_nr))

		elif hc_or_pd == 'pd':

			# 5 analysis folders
			sub_rpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"feedback_RPE_ortho")
			sub_arpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"feedback_aRPE_ortho")		    	
			sub_qval_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"feedback_qval_ortho")
			sub_gf_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"gf_RPE_ortho")	
			sub_qval_diff_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"feedback_qval_diff_ortho")		
			
			sub_stats_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"feedback","stats")
			sub_ev_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"feedback","workflow","l1design")
			events_filename = os.path.join(datadir,subjects[sb],pd_on_off,"func",subjects[sb]+"_task-train_run-%s_events.tsv"%(run_nr))	

		if not os.path.exists(sub_rpe_workflow_dir):
			os.makedirs(sub_rpe_workflow_dir)

		if not os.path.exists(sub_arpe_workflow_dir):
			os.makedirs(sub_arpe_workflow_dir)

		if not os.path.exists(sub_qval_workflow_dir):
			os.makedirs(sub_qval_workflow_dir)

		if not os.path.exists(sub_gf_workflow_dir):
			os.makedirs(sub_gf_workflow_dir)

		if not os.path.exists(sub_qval_diff_workflow_dir):
			os.makedirs(sub_qval_diff_workflow_dir)

		# All information from events.tsv
		events = pd.read_csv(events_filename, sep="\t")

		# Mask : Whole brain MNI mask file (converted to native space)
		maskfile = os.path.join(workflow_dir,"masks","mni2func_mask_dil.nii.gz")

		## LOOP OVER TRIALS (CONTRASTS) ##

		contrast_unsorted_files = []
		for txt_file in glob.glob(sub_ev_dir + '/ev_trial_*.txt'):
			contrast_unsorted_files.append(txt_file)

		### 1. Prepare Feedback EV (EV_feedback)

		EV_feedback, EV_goed, EV_fout = [[] for i in range(3)]

		# Need to sort contrast_unsorted_files so that the trial numbers are correct

		contrast_sorted_files = natural_sort(contrast_unsorted_files)
		
		for i in range(len(contrast_sorted_files)):
			
			for txt_file in glob.glob(contrast_sorted_files[i]):

				if fnmatch.fnmatch(txt_file, '*_Goed_*'):
					EV_feedback.append(1)
					EV_goed.append(1)
					EV_fout.append(0)
				elif fnmatch.fnmatch(txt_file, '*_Fout_*'):
					EV_feedback.append(-1)
					EV_fout.append(1)
					EV_goed.append(0)
				else:
					EV_feedback.append(0) # omissions (will exclude these copes later)
					EV_goed.append(0)
					EV_fout.append(0)

		EV_feedback = np.array(EV_feedback)
		EV_goed = np.array(EV_goed)
		EV_fout = np.array(EV_fout)

		### 2. Prepare RPE EV (EV_rpe, then EV_rpe_demeaned)

		# Mean-centred
		EV_rpe = np.array(events['RPE'])
		rpe_mean = EV_rpe[~np.isnan(EV_rpe)].mean() # get mean of all non-nan values
		EV_rpe_demeaned = EV_rpe - rpe_mean 

		EV_rpe_array = np.array([EV_feedback,EV_rpe_demeaned])
		EV_rpe_df = pd.DataFrame(EV_rpe_array.T, columns=['feedback','rpe_demeaned'])  

		EV_rpe_design_df = EV_rpe_df[EV_rpe_df['feedback']!=0] # exclude omissions

		# Orthogonalized
		rpeorth_demeaned = orth(EV_feedback,EV_rpe_demeaned)
		EV_rpeorth_array = np.array([EV_feedback,rpeorth_demeaned])
		EV_rpeorth_df = pd.DataFrame(EV_rpeorth_array.T, columns=['feedback','rpeorth_demeaned'])  

		EV_rpeorth_design_df = EV_rpeorth_df[EV_rpeorth_df['feedback']!=0] # exclude omissions

		# for goed/fout analysis
		gf_rpe_array = np.array([EV_goed,EV_fout,EV_rpe_demeaned])
		gf_rpe_df = pd.DataFrame(gf_rpe_array.T, columns=['goed','fout','rpe_demeaned'])  
		gf_rpe_design_df = gf_rpe_df[(gf_rpe_df['goed']==1) | (gf_rpe_df['fout']==1)]  # exclude omissions


		### 3. Prepare RPE update (alpha*RPE) EV (EV_arpe, then EV_arpe_demeaned)

		# Mean-centred
		EV_arpe = np.array(events['RPE_update'])
		arpe_mean = EV_arpe[~np.isnan(EV_arpe)].mean() # get mean of all non-nan values
		EV_arpe_demeaned = EV_arpe - arpe_mean 

		EV_arpe_array = np.array([EV_feedback,EV_arpe_demeaned])
		EV_arpe_df = pd.DataFrame(EV_arpe_array.T, columns=['feedback','arpe_demeaned'])  

		EV_arpe_design_df = EV_arpe_df[EV_arpe_df['feedback']!=0] # exclude omissions

		# Orthogonalized
		arpeorth_demeaned = orth(EV_feedback,EV_arpe_demeaned)
		EV_arpeorth_array = np.array([EV_feedback,arpeorth_demeaned])
		EV_arpeorth_df = pd.DataFrame(EV_arpeorth_array.T, columns=['feedback','arpeorth_demeaned'])  

		EV_arpeorth_design_df = EV_arpeorth_df[EV_arpeorth_df['feedback']!=0] # exclude omissions


		### 4. Prepare updated Q-val EV (EV_qval_update, then EV_qval_demeaned)

		# Mean-centred
		EV_qval_update=np.array(events['Qval_update'])
		qval_update_mean = EV_qval_update[~np.isnan(EV_qval_update)].mean() # get mean of all non-nan values
		EV_qval_update_demeaned = EV_qval_update - qval_update_mean 

		EV_qval_array = np.array([EV_feedback,EV_qval_update_demeaned])
		EV_qval_df = pd.DataFrame(EV_qval_array.T, columns=['feedback','qval_demeaned'])  

		EV_qval_design_df = EV_qval_df[EV_qval_df['feedback']!=0] # exclude omissions

		# Orthogonalized
		qvalorth_demeaned = orth(EV_feedback,EV_qval_update_demeaned)
		EV_qvalorth_array = np.array([EV_feedback,qvalorth_demeaned])
		EV_qvalorth_df = pd.DataFrame(EV_qvalorth_array.T, columns=['feedback','qvalorth_demeaned'])  

		EV_qvalorth_design_df = EV_qvalorth_df[EV_qvalorth_df['feedback']!=0] # exclude omissions

		
		### 5. Prepare Qval difference EV [chosen - unchosen] 
		
		# Mean-centered
		EV_qval_diff = events['Qval_chosen']-events['Qval_unchosen']
		qval_diff_mean = EV_qval_diff[~np.isnan(EV_qval_diff)].mean()
		EV_qval_diff_demeaned = EV_qval_diff - qval_diff_mean 

		EV_qval_diff_array = np.array([EV_feedback,EV_qval_diff_demeaned])
		EV_qval_diff_df = pd.DataFrame(EV_qval_diff_array.T, columns=['feedback','qval_diff_demeaned'])  

		EV_qval_diff_design_df = EV_qval_diff_df[EV_qval_diff_df['feedback']!=0] # exclude omissions


		# Orthogonalized
		qval_difforth_demeaned = orth(EV_feedback,EV_qval_diff_demeaned)
		EV_qval_difforth_array = np.array([EV_feedback,qval_difforth_demeaned])
		EV_qval_difforth_df = pd.DataFrame(EV_qval_difforth_array.T, columns=['feedback','qval_difforth_demeaned'])  

		EV_qval_difforth_design_df = EV_qval_difforth_df[EV_qval_difforth_df['feedback']!=0] # exclude omissions


		# Get Cope numbers (i.e. number of trials). Exclude copes that are omission trials.

		contrasts=np.arange(1,len(contrast_unsorted_files)+1)

		copes,varcopes, tstats = [[] for i in range(3)]

		EV_indices = EV_rpe_design_df.index
		
		for c,contrast in enumerate(contrasts):

			if c in EV_indices:

				copes.append(os.path.join(sub_stats_dir,'%s%i.nii.gz'%('cope',contrast)))
				varcopes.append(os.path.join(sub_stats_dir,'%s%i.nii.gz'%('varcope',contrast)))
				tstats.append(os.path.join(sub_stats_dir,'%s%i.nii.gz'%('tstat',contrast)))

		###### NIPYPE WORKFLOW 1: RPE as mean-centered covariate ######

		Parkflow_rpe = Workflow(name='workflow')
		Parkflow_rpe.base_dir = sub_rpe_workflow_dir

		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')

		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')

		feedback_tcont = ['group_mean', 'T',['reg1','reg2'],[1,0]]
		rpe_pos_tcont = ['rpe+', 'T',['reg1','reg2'],[0,1]]
		rpe_neg_tcont = ['rpe-', 'T',['reg1','reg2'],[0,-1]]
		
		multregmodel.inputs.contrasts = [feedback_tcont, rpe_pos_tcont, rpe_neg_tcont]
		multregmodel.inputs.regressors = dict(reg1=list(EV_rpe_design_df['feedback']),reg2=list(EV_rpeorth_design_df['rpeorth_demeaned']))

		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_rpe.base_dir,'stats'))

		Parkflow_rpe.connect([(copemerge,FE,[('merged_file','cope_file')]),
				(varcopemerge,FE,[('merged_file','var_cope_file')]),
				(multregmodel,FE,[('design_mat','design_file'),
								('design_con','t_con_file'),
								('design_grp','cov_split_file')]),
				])

		Parkflow_rpe.write_graph(graph2use='colored')
		Parkflow_rpe.run()


		###### NIPYPE WORKFLOW 2: RPE_update (alpha*RPE) as mean-centered covariate ######

		Parkflow_arpe = Workflow(name='workflow')
		Parkflow_arpe.base_dir = sub_arpe_workflow_dir

		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')

		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')
		
		feedback_tcont = ['group_mean', 'T',['reg1','reg2'],[1,0]]
		arpe_pos_tcont = ['arpe+', 'T',['reg1','reg2'],[0,1]]
		arpe_neg_tcont = ['arpe-', 'T',['reg1','reg2'],[0,-1]]
		
		multregmodel.inputs.contrasts = [feedback_tcont, arpe_pos_tcont, arpe_neg_tcont] 
		multregmodel.inputs.regressors = dict(reg1=list(EV_arpe_design_df['feedback']),reg2=list(EV_arpeorth_design_df['arpeorth_demeaned']))


		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_arpe.base_dir,'stats'))

		Parkflow_arpe.connect([(copemerge,FE,[('merged_file','cope_file')]),
				(varcopemerge,FE,[('merged_file','var_cope_file')]),
				(multregmodel,FE,[('design_mat','design_file'),
								('design_con','t_con_file'),
								('design_grp','cov_split_file')]),
				])

		Parkflow_arpe.write_graph(graph2use='colored')
		Parkflow_arpe.run()


		###### NIPYPE WORKFLOW 3: Qval_update (qval(t-1) + alpha*RPE) as mean-centered covariate ######

		Parkflow_qval = Workflow(name='workflow')
		Parkflow_qval.base_dir = sub_qval_workflow_dir

		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')

		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')
		
		feedback_tcont = ['group_mean', 'T',['reg1','reg2'],[1,0]]
		qval_pos_tcont = ['qval+', 'T',['reg1','reg2'],[0,1]]
		qval_neg_tcont = ['qval-', 'T',['reg1','reg2'],[0,-1]]
		
		multregmodel.inputs.contrasts = [feedback_tcont, qval_pos_tcont, qval_neg_tcont]
		multregmodel.inputs.regressors = dict(reg1=list(EV_qval_design_df['feedback']),reg2=list(EV_qvalorth_design_df['qvalorth_demeaned']))


		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_qval.base_dir,'stats'))

		Parkflow_qval.connect([(copemerge,FE,[('merged_file','cope_file')]),
				(varcopemerge,FE,[('merged_file','var_cope_file')]),
				(multregmodel,FE,[('design_mat','design_file'),
								('design_con','t_con_file'),
								('design_grp','cov_split_file')]),
				])

		Parkflow_qval.write_graph(graph2use='colored')
		Parkflow_qval.run()


		###### NIPYPE WORKFLOW 4: Separate Goed/Fout EVs with RPE as mean-centered covariate ######

		Parkflow_gf = Workflow(name='workflow')
		Parkflow_gf.base_dir = sub_gf_workflow_dir

		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')

		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')
		
		goed_fout_tcont = ['goed_fout', 'T',['reg1','reg2','reg3'],[1,-1,0]]
		fout_goed_tcont = ['fout_goed', 'T',['reg1','reg2','reg3'],[-1,1,0]]
		goed = ['goed', 'T',['reg1','reg2','reg3'],[1,0,0]]
		fout = ['fout', 'T',['reg1','reg2','reg3'],[0,1,0]] 
		rpe = ['rpe', 'T',['reg1','reg2','reg3'],[0,0,1]] 
		
		multregmodel.inputs.contrasts = [goed_fout_tcont, fout_goed_tcont, goed, fout, rpe]
		multregmodel.inputs.regressors = dict(reg1=list(gf_rpe_design_df['goed']),reg2=list(gf_rpe_design_df['fout']),reg3=list(EV_rpe_design_df['rpe_demeaned']))

		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_gf.base_dir,'stats'))

		Parkflow_gf.connect([(copemerge,FE,[('merged_file','cope_file')]),
				(varcopemerge,FE,[('merged_file','var_cope_file')]),
				(multregmodel,FE,[('design_mat','design_file'),
								('design_con','t_con_file'),
								('design_grp','cov_split_file')]),
				])

		Parkflow_gf.write_graph(graph2use='colored')
		Parkflow_gf.run()



		###### NIPYPE WORKFLOW 5: Qval_diff [chosen - unchosen Q-val] as mean-centered covariate ######

		Parkflow_qval_diff = Workflow(name='workflow')
		Parkflow_qval_diff.base_dir = sub_qval_diff_workflow_dir

		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')
		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')
		
		feedback_tcont = ['group_mean', 'T',['reg1','reg2'],[1,0]]
		qval_diff_pos_tcont = ['qval_diff+', 'T',['reg1','reg2'],[0,1]]
		qval_diff_neg_tcont = ['qval_diff-', 'T',['reg1','reg2'],[0,-1]]
		
		multregmodel.inputs.contrasts = [feedback_tcont, qval_diff_pos_tcont, qval_diff_neg_tcont]
		multregmodel.inputs.regressors = dict(reg1=list(EV_qval_diff_design_df['feedback']),reg2=list(EV_qval_difforth_design_df['qval_difforth_demeaned']))

		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_qval_diff.base_dir,'stats'))

		Parkflow_qval_diff.connect([(copemerge,FE,[('merged_file','cope_file')]),
				(varcopemerge,FE,[('merged_file','var_cope_file')]),
				(multregmodel,FE,[('design_mat','design_file'),
								('design_con','t_con_file'),
								('design_grp','cov_split_file')]),
				])

		Parkflow_qval_diff.write_graph(graph2use='colored')
		Parkflow_qval_diff.run()