
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

			# analysis folders
			sub_qval_diff_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"stimulus_QvalDiff")
			sub_choice_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"choice_QvalDiff")

			sub_stats_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"stimulus","stats")
			sub_ev_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"stimulus","workflow","l1design")
			events_filename = os.path.join(datadir,subjects[sb],"func",subjects[sb]+"_task-train_run-%s_events.tsv"%(run_nr))

		elif hc_or_pd == 'pd':

			# analysis folders
			sub_qval_diff_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"stimulus_QvalDiff")
			sub_choice_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"choice_QvalDiff")
			
			sub_stats_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"stimulus","stats")
			sub_ev_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"stimulus","workflow","l1design")
			events_filename = os.path.join(datadir,subjects[sb],pd_on_off,"func",subjects[sb]+"_task-train_run-%s_events.tsv"%(run_nr))	

		if not os.path.exists(sub_qval_diff_workflow_dir):
			os.makedirs(sub_qval_diff_workflow_dir)

		if not os.path.exists(sub_choice_workflow_dir):
			os.makedirs(sub_choice_workflow_dir)

		# All information from events.tsv
		events = pd.read_csv(events_filename, sep="\t")

		# Mask : Whole brain MNI mask file (converted to native space)
		maskfile = os.path.join(workflow_dir,"masks","mni2func_mask_dil.nii.gz")

		## LOOP OVER TRIALS (CONTRASTS) ##

		contrast_unsorted_files = []
		for txt_file in glob.glob(sub_ev_dir + '/ev_trial_*.txt'):
			contrast_unsorted_files.append(txt_file)

		### 1. Prepare Stimulus EV (EV_stimulus)

		EV_stimulus = []

		# Need to sort contrast_unsorted_files so that the trial numbers are correct

		contrast_sorted_files = natural_sort(contrast_unsorted_files)
		
		for i in range(len(contrast_sorted_files)):
			
			for txt_file in glob.glob(contrast_sorted_files[i]):

				# append 1 for all response trials (and 0 for omissions)
				if fnmatch.fnmatch(txt_file, '*_Goed_*'):
					EV_stimulus.append(1) 
				elif fnmatch.fnmatch(txt_file, '*_Fout_*'):
					EV_stimulus.append(1)
				else:
					EV_stimulus.append(0) # omissions (will exclude these copes later)

		EV_stimulus = np.array(EV_stimulus)

		### 2. Prepare Choice EVs (EV_correct_choice, EV_incorrect_choice)

		EV_correct_choice, EV_incorrect_choice = [[] for i in range(2)]
		
		for i in range(len(contrast_sorted_files)):
			
			for txt_file in glob.glob(contrast_sorted_files[i]):

				# Append 1 for all response trials (and 0 for omissions)
				if (fnmatch.fnmatch(txt_file, '*_choice_A_*') or fnmatch.fnmatch(txt_file, '*_choice_C_*') or fnmatch.fnmatch(txt_file, '*_choice_E_*')):
					EV_correct_choice.append(1) 
					EV_incorrect_choice.append(0) 
				elif (fnmatch.fnmatch(txt_file, '*_choice_B_*') or fnmatch.fnmatch(txt_file, '*_choice_D_*') or fnmatch.fnmatch(txt_file, '*_choice_F_*')):
					EV_incorrect_choice.append(1)
					EV_correct_choice.append(0) 
				else:
					EV_correct_choice.append(0) # omissions (will exclude these copes later)
					EV_incorrect_choice.append(0)

		EV_correct_choice = np.array(EV_correct_choice)
		EV_incorrect_choice = np.array(EV_incorrect_choice)

		
		### 3. Prepare Qval difference EV [chosen - unchosen] 
		
		# Mean-centered
		EV_qval_diff = events['Qval_chosen']-events['Qval_unchosen']
		qval_diff_mean = EV_qval_diff[~np.isnan(EV_qval_diff)].mean()
		EV_qval_diff_demeaned = EV_qval_diff - qval_diff_mean 

		EV_qval_diff_array = np.array([EV_stimulus,EV_qval_diff_demeaned])
		EV_qval_diff_df = pd.DataFrame(EV_qval_diff_array.T, columns=['stimulus','qval_diff_demeaned'])  

		EV_qval_diff_design_df = EV_qval_diff_df[EV_qval_diff_df['stimulus']!=0] # exclude omissions

		choice_qval_diff_array = np.array([EV_correct_choice,EV_incorrect_choice,EV_qval_diff_demeaned])
		choice_qval_diff_df = pd.DataFrame(choice_qval_diff_array.T, columns=['correct_choice','incorrect_choice','qval_diff_demeaned'])  
		choice_qval_diff_design_df = choice_qval_diff_df[(choice_qval_diff_df['correct_choice']==1) | (choice_qval_diff_df['incorrect_choice']==1)]  # exclude omissions


		# Get Cope numbers (i.e. number of trials). Exclude copes that are omission trials.

		contrasts=np.arange(1,len(contrast_unsorted_files)+1)

		copes,varcopes, tstats = [[] for i in range(3)]

		EV_indices = EV_qval_diff_design_df.index
		
		for c,contrast in enumerate(contrasts):

			if c in EV_indices:

				copes.append(os.path.join(sub_stats_dir,'%s%i.nii.gz'%('cope',contrast)))
				varcopes.append(os.path.join(sub_stats_dir,'%s%i.nii.gz'%('varcope',contrast)))
				tstats.append(os.path.join(sub_stats_dir,'%s%i.nii.gz'%('tstat',contrast)))


		###### NIPYPE WORKFLOW 1: Qval_diff [chosen - unchosen Q-val] as mean-centered covariate ######

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
		
		stimulus_tcont = ['group_mean', 'T',['reg1','reg2'],[1,0]]
		qval_diff_pos_tcont = ['qval_diff+', 'T',['reg1','reg2'],[0,1]]
		qval_diff_neg_tcont = ['qval_diff-', 'T',['reg1','reg2'],[0,-1]]
		
		multregmodel.inputs.contrasts = [stimulus_tcont, qval_diff_pos_tcont, qval_diff_neg_tcont]
		multregmodel.inputs.regressors = dict(reg1=list(EV_qval_diff_design_df['stimulus']),reg2=list(EV_qval_diff_design_df['qval_diff_demeaned']))

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


		###### NIPYPE WORKFLOW 2: Separate correct_choice/incorrect_choice EVs with QvalDiff as mean-centered covariate ######

		# Parkflow_choice = Workflow(name='workflow')
		# Parkflow_choice.base_dir = sub_choice_workflow_dir

		# # Create nodes

		# copemerge = Node(interface=fsl.Merge(
		# 	dimension='t',
		# 	in_files=copes),
		# 	name='copemerge')
		# varcopemerge = Node(interface=fsl.Merge(
		# 	dimension='t',
		# 	in_files=varcopes),
		# 	name='varcopemerge')

		# multregmodel = Node(interface=fsl.MultipleRegressDesign(
		# 	contrasts=[],
		# 	regressors={}),
		# 	name='multregmodel')
		
		# correct = ['correct', 'T',['reg1','reg2','reg3'],[1,0,0]]
		# incorrect = ['incorrect', 'T',['reg1','reg2','reg3'],[0,1,0]] 
		# qval_diff_pos = ['qval_diff_pos', 'T',['reg1','reg2','reg3'],[0,0,1]] 
		# qval_diff_neg = ['qval_diff_neg', 'T',['reg1','reg2','reg3'],[0,0,-1]] 
		# correct_incorrect_tcont = ['correct_incorrect', 'T',['reg1','reg2','reg3'],[1,-1,0]]
		# incorrect_correct_tcont = ['incorrect_correct', 'T',['reg1','reg2','reg3'],[-1,1,0]]
	
		# multregmodel.inputs.contrasts = [correct,incorrect,qval_diff_pos,qval_diff_neg,correct_incorrect_tcont,incorrect_correct_tcont]
		# multregmodel.inputs.regressors = dict(reg1=list(choice_qval_diff_design_df['correct_choice']),reg2=list(choice_qval_diff_design_df['incorrect_choice']),reg3=list(choice_qval_diff_design_df['qval_diff_demeaned']))

		# FE=Node(interface=fsl.FLAMEO(
		# 	run_mode='fe',
		# 	mask_file=maskfile),
		# 	name='FE',
		# 	stats_dir=os.path.join(Parkflow_choice.base_dir,'stats'))

		# Parkflow_choice.connect([(copemerge,FE,[('merged_file','cope_file')]),
		# 		(varcopemerge,FE,[('merged_file','var_cope_file')]),
		# 		(multregmodel,FE,[('design_mat','design_file'),
		# 						('design_con','t_con_file'),
		# 						('design_grp','cov_split_file')]),
		# 		])

		# Parkflow_choice.write_graph(graph2use='colored')
		# Parkflow_choice.run()

