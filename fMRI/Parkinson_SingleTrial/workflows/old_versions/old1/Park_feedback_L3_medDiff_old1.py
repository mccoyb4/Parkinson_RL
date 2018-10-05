
from nipype.interfaces import fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import os
#from IPython import embed as shell

sub_id = str(sys.argv[1])
smooth = str(sys.argv[2]) # use smoothed output or not? 0 = no smooth, 1 = smooth
ae_or_lisa = str(sys.argv[3]) # 'ae' or 'lisa' server

subjects = [sub_id]

# subjects = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-219','sub-220','sub-221','sub-222','sub-223','sub-224']

# Name directories for input and output
if ae_or_lisa == 'ae':

	datadir='/home/shared/2016/Parkinson/data/pd'
	fmriprep_dirs = ['/home/shared/2016/Parkinson/fmriprep_preproc/pd_on_syn-sdc','/home/shared/2016/Parkinson/fmriprep_preproc/pd_off_syn-sdc']

	if smooth == '0':
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth/pd"
	elif smooth == '1':
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth/pd"

elif ae_or_lisa == 'lisa':

	datadir='/nfs/bromccoy/Parkinson/data/pd'
	fmriprep_dirs = ['/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_on_syn-sdc','/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_off_syn-sdc']

	if smooth == '0':
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth/pd"
	elif smooth == '1':
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth/pd"


for sb in range(len(subjects)):

	# Set up med_diff workflow directory
	sub_rpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],"med_diff","feedback_RPE")
	sub_arpe_workflow_dir = os.path.join(workflow_dir,subjects[sb],"med_diff","feedback_aRPE")
	sub_qval_workflow_dir = os.path.join(workflow_dir,subjects[sb],"med_diff","feedback_qval")

	if not os.path.exists(sub_rpe_workflow_dir):
		os.makedirs(sub_rpe_workflow_dir)
	if not os.path.exists(sub_arpe_workflow_dir):
		os.makedirs(sub_arpe_workflow_dir)	
	if not os.path.exists(sub_qval_workflow_dir):
		os.makedirs(sub_qval_workflow_dir)	

	if subjects[sb] not in ('sub-203','sub-209','sub-213','sub-215'):
	
		# ON
		sub_on_rpe_dir = os.path.join(workflow_dir,subjects[sb],"on","train","across_runs","feedback_RPE")
		sub_on_arpe_dir = os.path.join(workflow_dir,subjects[sb],"on","train","across_runs","feedback_aRPE")	
		sub_on_qval_dir = os.path.join(workflow_dir,subjects[sb],"on","train","across_runs","feedback_qval")

		# OFF
		sub_off_rpe_dir = os.path.join(workflow_dir,subjects[sb],"off","train","across_runs","feedback_RPE")
		sub_off_arpe_dir = os.path.join(workflow_dir,subjects[sb],"off","train","across_runs","feedback_aRPE")
		sub_off_qval_dir = os.path.join(workflow_dir,subjects[sb],"off","train","across_runs","feedback_qval")

		# Combine
		feat_rpe_dirs = [sub_on_rpe_dir,sub_off_rpe_dir]
		feat_arpe_dirs = [sub_on_arpe_dir,sub_off_arpe_dir]
		feat_qval_dirs = [sub_on_qval_dir,sub_off_qval_dir]

	elif subjects[sb] in ('sub-203','sub-209','sub-215'):

		# ON
		sub_on_rpe_dir = os.path.join(workflow_dir,subjects[sb],"on","train","run-1","feedback_RPE")
		sub_on_arpe_dir = os.path.join(workflow_dir,subjects[sb],"on","train","run-1","feedback_aRPE")	
		sub_on_qval_dir = os.path.join(workflow_dir,subjects[sb],"on","train","run-1","feedback_qval")	

		# OFF
		sub_off_rpe_dir = os.path.join(workflow_dir,subjects[sb],"off","train","run-1","feedback_RPE")
		sub_off_arpe_dir = os.path.join(workflow_dir,subjects[sb],"off","train","run-1","feedback_aRPE")
		sub_off_qval_dir = os.path.join(workflow_dir,subjects[sb],"off","train","run-1","feedback_qval")

		# Combine
		cope_rpe_dirs = [sub_on_rpe_dir,sub_off_rpe_dir]		
		cope_arpe_dirs = [sub_on_arpe_dir,sub_off_arpe_dir]
		cope_qval_dirs = [sub_on_qval_dir,sub_off_qval_dir]

	elif subjects[sb] in ('sub-213'):

		# ON
		sub_on_rpe_dir = os.path.join(workflow_dir,subjects[sb],"on","train","run-2","feedback_RPE")
		sub_on_arpe_dir = os.path.join(workflow_dir,subjects[sb],"on","train","run-2","feedback_aRPE")	
		sub_on_qval_dir = os.path.join(workflow_dir,subjects[sb],"on","train","run-2","feedback_qval")	

		# OFF
		sub_off_rpe_dir = os.path.join(workflow_dir,subjects[sb],"off","train","run-2","feedback_RPE")
		sub_off_arpe_dir = os.path.join(workflow_dir,subjects[sb],"off","train","run-2","feedback_aRPE")
		sub_off_qval_dir = os.path.join(workflow_dir,subjects[sb],"off","train","run-2","feedback_qval")

		# Combine
		cope_rpe_dirs = [sub_on_rpe_dir,sub_off_rpe_dir]		
		cope_arpe_dirs = [sub_on_arpe_dir,sub_off_arpe_dir]
		cope_qval_dirs = [sub_on_qval_dir,sub_off_qval_dir]

	## CREATE ACROSS-SESSION MASK ##

	sessmaskfile_rpe = os.path.join(sub_rpe_workflow_dir,"mask.nii.gz")
	sessmaskfile_arpe = os.path.join(sub_arpe_workflow_dir,"mask.nii.gz")
	sessmaskfile_qval = os.path.join(sub_qval_workflow_dir,"mask.nii.gz")

	mask = np.zeros([60,71,60,2]) # always 2 sessions: on and off
	k=0

	for i in range(2): # 2 sessions : on and off 
		if subjects[sb] not in ('sub-203','sub-209', 'sub-213','sub-215'):
			maskfile = os.path.join(feat_rpe_dirs[i],"mask.nii.gz")
		elif subjects[sb] in ('sub-203','sub-209','sub-215'):
			maskfile = os.path.join(fmriprep_dirs[i],"fmriprep", subjects[sb], "func", subjects[sb] + "_task-train_run-1_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz")
		elif subjects[sb] in ('sub-213'):	
			maskfile = os.path.join(fmriprep_dirs[i],"fmriprep", subjects[sb], "func", subjects[sb] + "_task-train_run-2_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz")

		if os.path.exists(maskfile):
			masksub = nib.load(maskfile)
			data = masksub.get_data()
			mask[:,:,:,k] = data
			k += 1

	mask = mask[:,:,:,:k]

	mask = np.mean(mask,axis=3)
	mask = np.where(mask>0.8,1,0)
	maskimg = nib.Nifti1Image(mask,affine=masksub.affine,header=masksub.header)
	maskimg.to_filename(sessmaskfile_rpe)
	maskimg.to_filename(sessmaskfile_arpe)
	maskimg.to_filename(sessmaskfile_qval)

	## 1. RPE workflow ##

	contrasts = [1,2,3] # Feedback, RPE+, RPE-

	# Loop over contrasts #

	for contrast in contrasts: 

		if subjects[sb] not in ('sub-203','sub-209', 'sub-213','sub-215'):
			# collecting copes from feat directories (for those subs that had across_runs)
			copes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_rpe_dirs]
			varcopes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_rpe_dirs]
			tstats = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_rpe_dirs]

		elif subjects[sb] in ('sub-203','sub-209', 'sub-213','sub-215'):
			# appending remaining copes for those subs that had only one run
			copes = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)) for x in cope_rpe_dirs]
			varcopes = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast))for x in cope_rpe_dirs]
			tstats = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('tstat',contrast))for x in cope_rpe_dirs]


		# Create workflow
		Parkflow_rpe_across_sess = Workflow(name='workflow')
		Parkflow_rpe_across_sess.base_dir = os.path.join(sub_rpe_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_rpe_across_sess.base_dir):
			os.makedirs(Parkflow_rpe_across_sess.base_dir)
	
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
		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')
		
		multregmodel.inputs.contrasts = [['on-off', 'T',['reg1','reg2'],[1,-1]], ['off-on', 'T',['reg1','reg2'],[-1,1]]] 
		multregmodel.inputs.regressors = dict(reg1=[1, 0],reg2=[0, 1])

		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=sessmaskfile_rpe),
			name='FE',
			stats_dir=os.path.join(Parkflow_rpe_across_sess.base_dir,'stats'))

		
		Parkflow_rpe_across_sess.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(multregmodel,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
						])

		Parkflow_rpe_across_sess.write_graph(graph2use='colored')
		Parkflow_rpe_across_sess.run()


	## 2. RPE_update (alpha * RPE) workflow ##

	contrasts = [1,2,3] # Feedback, aRPE+, aRPE-

	for contrast in contrasts: 

		if subjects[sb] not in ('sub-203','sub-209', 'sub-213','sub-215'):
			# collecting copes from feat directories (for those subs that had across_runs)
			copes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_arpe_dirs]
			varcopes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_arpe_dirs]
			tstats = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_arpe_dirs]

		elif subjects[sb] in ('sub-203','sub-209', 'sub-213','sub-215'):
			# appending remaining copes for those subs that had only one run
			copes = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)) for x in cope_arpe_dirs]
			varcopes = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast))for x in cope_arpe_dirs]
			tstats = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('tstat',contrast))for x in cope_arpe_dirs]

		# Create workflow
		Parkflow_arpe_across_sess = Workflow(name='workflow')
		Parkflow_arpe_across_sess.base_dir = os.path.join(sub_arpe_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_arpe_across_sess.base_dir):
			os.makedirs(Parkflow_arpe_across_sess.base_dir)
	
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
		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')
		
		multregmodel.inputs.contrasts = [['on-off', 'T',['reg1','reg2'],[1,-1]], ['off-on', 'T',['reg1','reg2'],[-1,1]]] 
		multregmodel.inputs.regressors = dict(reg1=[1, 0],reg2=[0, 1])

		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=sessmaskfile_arpe),
			name='FE',
			stats_dir=os.path.join(Parkflow_arpe_across_sess.base_dir,'stats'))

		
		Parkflow_arpe_across_sess.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(multregmodel,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
						])

		Parkflow_arpe_across_sess.write_graph(graph2use='colored')
		Parkflow_arpe_across_sess.run()

	## 3. Qval_update workflow ##

	contrasts = [1,2,3] # Feedback, Qval+, Qval-

	for contrast in contrasts: 

		if subjects[sb] not in ('sub-203','sub-209', 'sub-213','sub-215'):
			# collecting copes from feat directories (for those subs that had across_runs)
			copes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_qval_dirs]
			varcopes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_qval_dirs]
			tstats = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_qval_dirs]

		elif subjects[sb] in ('sub-203','sub-209', 'sub-213','sub-215'):
			# appending remaining copes for those subs that had only one run
			copes = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)) for x in cope_qval_dirs]
			varcopes = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast))for x in cope_qval_dirs]
			tstats = [os.path.join(x,'workflow','FE','stats','%s%i.nii.gz'%('tstat',contrast))for x in cope_qval_dirs]

		# Create workflow
		Parkflow_qval_across_sess = Workflow(name='workflow')
		Parkflow_qval_across_sess.base_dir = os.path.join(sub_qval_workflow_dir,"cope"+str(contrast))

		if not os.path.exists(Parkflow_qval_across_sess.base_dir):
			os.makedirs(Parkflow_qval_across_sess.base_dir)
	
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
		multregmodel = Node(interface=fsl.MultipleRegressDesign(
			contrasts=[],
			regressors={}),
			name='multregmodel')
		
		multregmodel.inputs.contrasts = [['on-off', 'T',['reg1','reg2'],[1,-1]], ['off-on', 'T',['reg1','reg2'],[-1,1]]] 
		multregmodel.inputs.regressors = dict(reg1=[1, 0],reg2=[0, 1])

		FE=Node(interface=fsl.FLAMEO(
			run_mode='fe',
			mask_file=sessmaskfile_qval),
			name='FE',
			stats_dir=os.path.join(Parkflow_qval_across_sess.base_dir,'stats'))

		
		Parkflow_qval_across_sess.connect([(copemerge,FE,[('merged_file','cope_file')]),
						(varcopemerge,FE,[('merged_file','var_cope_file')]),
						(multregmodel,FE,[('design_mat','design_file'),
										('design_con','t_con_file'),
										('design_grp','cov_split_file')]),
						])

		Parkflow_qval_across_sess.write_graph(graph2use='colored')
		Parkflow_qval_across_sess.run()
