
# Based on Github PoldrackLab script: https://github.com/poldracklab/CNP_task_analysis/blob/master/CNP_analysis.py

from nipype.interfaces import fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import os
#from IPython import embed as shell

sub_id = str(sys.argv[1])
hc_or_pd = str(sys.argv[2]) # group : 'hc', 'pd'
pd_on_off = str(sys.argv[3]) # 'on', 'off'
smooth = str(sys.argv[4]) # use smoothed output or not? 0 = no smooth, 1 = smooth
ae_or_lisa = str(sys.argv[5]) # 'ae' or 'lisa' server

subjects = [sub_id]

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

	if hc_or_pd == 'hc':

		sub_rpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","across_runs","feedback_RPE_ortho")
		sub_arpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","across_runs","feedback_aRPE_ortho")
		sub_qval_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","across_runs","feedback_qval_ortho")
		sub_gf_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","across_runs","gf_RPE_ortho")
		sub_qval_diff_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","across_runs","feedback_qval_diff_ortho")

		# RPE dirs
		sub_run1_rpe_dir = os.path.join(workflow_dir,subjects[sb],"train","run-1","feedback_RPE_ortho","workflow",'FE')
		sub_run2_rpe_dir = os.path.join(workflow_dir,subjects[sb],"train","run-2","feedback_RPE_ortho","workflow",'FE')
		
		# alpha RPE dirs
		sub_run1_arpe_dir = os.path.join(workflow_dir,subjects[sb],"train","run-1","feedback_aRPE_ortho","workflow",'FE')
		sub_run2_arpe_dir = os.path.join(workflow_dir,subjects[sb],"train","run-2","feedback_aRPE_ortho","workflow",'FE')

		# Qval update dirs
		sub_run1_qval_dir = os.path.join(workflow_dir,subjects[sb],"train","run-1","feedback_qval_ortho","workflow",'FE')
		sub_run2_qval_dir = os.path.join(workflow_dir,subjects[sb],"train","run-2","feedback_qval_ortho","workflow",'FE')

		# Goed/Fout RPE dirs
		sub_run1_gf_dir = os.path.join(workflow_dir,subjects[sb],"train","run-1","gf_RPE_ortho","workflow",'FE')
		sub_run2_gf_dir = os.path.join(workflow_dir,subjects[sb],"train","run-2","gf_RPE_ortho","workflow",'FE')

		# Qval_diff dirs
		sub_run1_qval_diff_dir = os.path.join(workflow_dir,subjects[sb],"train","run-1","feedback_qval_diff_ortho","workflow",'FE')
		sub_run2_qval_diff_dir = os.path.join(workflow_dir,subjects[sb],"train","run-2","feedback_qval_diff_ortho","workflow",'FE')


	elif hc_or_pd == 'pd':

		sub_rpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","across_runs","feedback_RPE_ortho")
		sub_arpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","across_runs","feedback_aRPE_ortho")
		sub_qval_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","across_runs","feedback_qval_ortho")
		sub_gf_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","across_runs","gf_RPE_ortho")
		sub_qval_diff_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","across_runs","feedback_qval_diff_ortho")

		# RPE dirs
		sub_run1_rpe_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-1","feedback_RPE_ortho","workflow",'FE')
		sub_run2_rpe_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-2","feedback_RPE_ortho","workflow",'FE')
		
		# alpha RPE dirs
		sub_run1_arpe_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-1","feedback_aRPE_ortho","workflow",'FE')
		sub_run2_arpe_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-2","feedback_aRPE_ortho","workflow",'FE')

		# Qval update dirs
		sub_run1_qval_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-1","feedback_qval_ortho","workflow",'FE')
		sub_run2_qval_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-2","feedback_qval_ortho","workflow",'FE')

		# Goed/Fout RPE dirs
		sub_run1_gf_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-1","gf_RPE_ortho","workflow",'FE')
		sub_run2_gf_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-2","gf_RPE_ortho","workflow",'FE')

		# Qval_diff dirs
		sub_run1_qval_diff_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-1","feedback_qval_diff_ortho","workflow",'FE')
		sub_run2_qval_diff_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-2","feedback_qval_diff_ortho","workflow",'FE')


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

	feat_rpe_dirs = [sub_run1_rpe_dir,sub_run2_rpe_dir]
	feat_arpe_dirs = [sub_run1_arpe_dir,sub_run2_arpe_dir]
	feat_qval_dirs = [sub_run1_qval_dir,sub_run2_qval_dir]
	feat_gf_dirs = [sub_run1_gf_dir,sub_run2_gf_dir]
	feat_qval_diff_dirs = [sub_run1_qval_diff_dir,sub_run2_qval_diff_dir]

	## ACROSS-RUN MASK ##
	maskfile = os.path.join(workflow_dir,"masks","mni2func_mask_dil.nii.gz")

	## 1. RPE workflow ##

	contrasts = [1,2,3] # Feedback, RPE+, RPE-

	# Loop over contrasts #

	for contrast in contrasts: 

		copes = [os.path.join(x,'stats','%s%i.nii.gz'%('cope',contrast)) for x in feat_rpe_dirs]
		varcopes = [os.path.join(x,'stats','%s%i.nii.gz'%('varcope',contrast)) for x in feat_rpe_dirs]
		tstats = [os.path.join(x,'stats','%s%i.nii.gz'%('tstat',contrast)) for x in feat_rpe_dirs]

		# Create workflow
		Parkflow_rpe_across_runs = Workflow(name='workflow')
		Parkflow_rpe_across_runs.base_dir = os.path.join(sub_rpe_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_rpe_across_runs.base_dir):
			os.makedirs(Parkflow_rpe_across_runs.base_dir)
	
		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')
		tstatmerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=tstats),
			name='tstatmerge')		
		level2model = Node(interface=fsl.L2Model(
			num_copes=len(copes)),
			name='l2model')
		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_rpe_across_runs.base_dir,'stats'))
		
		Parkflow_rpe_across_runs.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(level2model,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
				])

		Parkflow_rpe_across_runs.write_graph(graph2use='colored')
		Parkflow_rpe_across_runs.run()


	## 2. RPE_update (alpha * RPE) workflow ##

	contrasts = [1,2,3] # Feedback, aRPE+, aRPE-

	for contrast in contrasts: 

		copes = [os.path.join(x,'stats','%s%i.nii.gz'%('cope',contrast)) for x in feat_arpe_dirs]
		varcopes = [os.path.join(x,'stats','%s%i.nii.gz'%('varcope',contrast)) for x in feat_arpe_dirs]
		tstats = [os.path.join(x,'stats','%s%i.nii.gz'%('tstat',contrast)) for x in feat_arpe_dirs]


		# Create workflow
		Parkflow_arpe_across_runs = Workflow(name='workflow')
		Parkflow_arpe_across_runs.base_dir = os.path.join(sub_arpe_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_arpe_across_runs.base_dir):
			os.makedirs(Parkflow_arpe_across_runs.base_dir)
	
		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')
		tstatmerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=tstats),
			name='tstatmerge')		
		level2model = Node(interface=fsl.L2Model(
			num_copes=len(copes)),
			name='l2model')
		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_arpe_across_runs.base_dir,'stats'))

		
		Parkflow_arpe_across_runs.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(level2model,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
						])

		Parkflow_arpe_across_runs.write_graph(graph2use='colored')
		Parkflow_arpe_across_runs.run()


	## 3. Qval_update (qval(t-1) + alpha*RPE) workflow ##

	contrasts = [1,2,3] # Feedback, Qval+, Qval-

	for contrast in contrasts: 

		copes = [os.path.join(x,'stats','%s%i.nii.gz'%('cope',contrast)) for x in feat_qval_dirs]
		varcopes = [os.path.join(x,'stats','%s%i.nii.gz'%('varcope',contrast)) for x in feat_qval_dirs]
		tstats = [os.path.join(x,'stats','%s%i.nii.gz'%('tstat',contrast)) for x in feat_qval_dirs]

		# Create workflow
		Parkflow_qval_across_runs = Workflow(name='workflow')
		Parkflow_qval_across_runs.base_dir = os.path.join(sub_qval_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_qval_across_runs.base_dir):
			os.makedirs(Parkflow_qval_across_runs.base_dir)
	
		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')
		tstatmerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=tstats),
			name='tstatmerge')		
		level2model = Node(interface=fsl.L2Model(
			num_copes=len(copes)),
			name='l2model')
		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_qval_across_runs.base_dir,'stats'))

		
		Parkflow_qval_across_runs.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(level2model,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
						])

		Parkflow_qval_across_runs.write_graph(graph2use='colored')
		Parkflow_qval_across_runs.run()


	## 4. Goed/Fout RPE workflow ##

	contrasts = [1,2,3,4,5] # Goed>Fout, Fout>Goed, Goed only, Fout only, RPE

	# Loop over contrasts #

	for contrast in contrasts: 

		copes = [os.path.join(x,'stats','%s%i.nii.gz'%('cope',contrast)) for x in feat_gf_dirs]
		varcopes = [os.path.join(x,'stats','%s%i.nii.gz'%('varcope',contrast)) for x in feat_gf_dirs]
		tstats = [os.path.join(x,'stats','%s%i.nii.gz'%('tstat',contrast)) for x in feat_gf_dirs]

		# Create workflow
		Parkflow_gf_across_runs = Workflow(name='workflow')
		Parkflow_gf_across_runs.base_dir = os.path.join(sub_gf_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_gf_across_runs.base_dir):
			os.makedirs(Parkflow_gf_across_runs.base_dir)
	
		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')
		tstatmerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=tstats),
			name='tstatmerge')		
		level2model = Node(interface=fsl.L2Model(
			num_copes=len(copes)),
			name='l2model')
		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_gf_across_runs.base_dir,'stats'))

		
		Parkflow_gf_across_runs.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(level2model,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
				])

		Parkflow_gf_across_runs.write_graph(graph2use='colored')
		Parkflow_gf_across_runs.run()


	## 5. Qval_diff (chosen - unchosen value) workflow ##

	contrasts = [1,2,3] # Feedback, Qval_diff+, Qval_diff-

	for contrast in contrasts: 

		copes = [os.path.join(x,'stats','%s%i.nii.gz'%('cope',contrast)) for x in feat_qval_diff_dirs]
		varcopes = [os.path.join(x,'stats','%s%i.nii.gz'%('varcope',contrast)) for x in feat_qval_diff_dirs]
		tstats = [os.path.join(x,'stats','%s%i.nii.gz'%('tstat',contrast)) for x in feat_qval_diff_dirs]

		# Create workflow
		Parkflow_qval_diff_across_runs = Workflow(name='workflow')
		Parkflow_qval_diff_across_runs.base_dir = os.path.join(sub_qval_diff_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_qval_diff_across_runs.base_dir):
			os.makedirs(Parkflow_qval_diff_across_runs.base_dir)
	
		# Create nodes

		copemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=copes),
			name='copemerge')
		varcopemerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=varcopes),
			name='varcopemerge')
		tstatmerge = Node(interface=fsl.Merge(
			dimension='t',
			in_files=tstats),
			name='tstatmerge')		
		level2model = Node(interface=fsl.L2Model(
			num_copes=len(copes)),
			name='l2model')
		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=maskfile),
			name='FE',
			stats_dir=os.path.join(Parkflow_qval_diff_across_runs.base_dir,'stats'))

		
		Parkflow_qval_diff_across_runs.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(level2model,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
						])

		Parkflow_qval_diff_across_runs.write_graph(graph2use='colored')
		Parkflow_qval_diff_across_runs.run()
