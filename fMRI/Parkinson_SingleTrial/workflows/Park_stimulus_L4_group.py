
from nipype.interfaces import fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import os
#from IPython import embed as shell

# Adjust these variables to run required group analysis
hc_or_pd = str(sys.argv[1]) #'pd'
pd_on_off = str(sys.argv[2]) #'on'
smooth = str(sys.argv[3]) #'0' # use smoothed output or not? 0 = no smooth, 1 = smooth
ae_or_lisa = str(sys.argv[4]) #'ae'

if hc_or_pd == 'hc':

	subjects = ['sub-108', 'sub-111', 'sub-113', 'sub-115', 'sub-116', 'sub-118', 'sub-119', 'sub-123', 'sub-124', 'sub-126', 'sub-129', 'sub-130', 'sub-131', 'sub-132', 'sub-133']
	# ['sub-108','sub-111','sub-112','sub-113','sub-114','sub-115','sub-116','sub-117','sub-118','sub-119','sub-120','sub-121','sub-123','sub-124','sub-126','sub-127','sub-128','sub-129','sub-130','sub-131','sub-132','sub-133']

	# Name directories for input and output
	if ae_or_lisa == 'ae':

		fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/hc_syn-sdc'

		if smooth == '0':
			group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/hc/group"
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/hc"
		elif smooth == '1':
			group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/hc/group"
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/hc"

	elif ae_or_lisa == 'lisa':

		fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/hc_syn-sdc'

		if smooth == '0':
			group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/hc/group"
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/hc"
		elif smooth == '1':
			group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/hc/group"
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/hc"
	
elif hc_or_pd == 'pd':

	subjects = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-219','sub-220','sub-221','sub-222','sub-223','sub-224'] 

	# Name directories for input and output

	if pd_on_off == 'on':

		if ae_or_lisa == 'ae':

			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
			if smooth == '0':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd/group/on"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd"
			elif smooth == '1':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd/group/on"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd"

		elif ae_or_lisa == 'lisa':

			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
			if smooth == '0':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd/group/on"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd"
			elif smooth == '1':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/pd/group/on"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/pd"
		
	elif pd_on_off == 'off':

		if ae_or_lisa == 'ae':
		
			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_off_syn-sdc'
			if smooth == '0':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd/group/off"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd"
			elif smooth == '1':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd/group/off"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd"

		elif ae_or_lisa == 'lisa':
			
			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_off_syn-sdc'
			if smooth == '0':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd/group/off"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd"
			elif smooth == '1':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/pd/group/off"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/pd"


if not os.path.exists(group_dir):
	os.makedirs(group_dir)		

feat_qval_diff_dirs_all_subs, feat_choice_dirs_all_subs = [[] for i in range(2)]
cope_qval_diff_dirs_all_subs, cope_choice_dirs_all_subs = [[] for i in range(2)]

for sb in range(len(subjects)):

	if hc_or_pd == 'hc':
		sub_workflow_dir = os.path.join(workflow_dir,subjects[sb])
	elif hc_or_pd == 'pd':
		sub_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off)

	if subjects[sb] not in ('sub-203','sub-209', 'sub-213', 'sub-215', 'sub-115'):

		sub_qval_diff_feat_dir = os.path.join(sub_workflow_dir,"train","across_runs","stimulus_QvalDiff")
		sub_choice_feat_dir = os.path.join(sub_workflow_dir,"train","across_runs","choice_QvalDiff")

		feat_qval_diff_dirs_all_subs.append(sub_qval_diff_feat_dir)
		feat_choice_dirs_all_subs.append(sub_choice_feat_dir)
	
	elif subjects[sb] in ('sub-203','sub-209','sub-215'):
	
		sub_qval_diff_cope_dir = os.path.join(sub_workflow_dir,"train","run-1","stimulus_QvalDiff")
		sub_choice_cope_dir = os.path.join(sub_workflow_dir,"train","run-1","choice_QvalDiff")

		cope_qval_diff_dirs_all_subs.append(sub_qval_diff_cope_dir)
		cope_choice_dirs_all_subs.append(sub_choice_cope_dir)


	elif subjects[sb] in ('sub-213','sub-115'):

		sub_qval_diff_cope_dir = os.path.join(sub_workflow_dir,"train","run-2","stimulus_QvalDiff")
		sub_choice_cope_dir = os.path.join(sub_workflow_dir,"train","run-2","choice_QvalDiff")

		cope_qval_diff_dirs_all_subs.append(sub_qval_diff_cope_dir)	
		cope_choice_dirs_all_subs.append(sub_choice_cope_dir)	

groupmaskfile = os.path.join(workflow_dir,"masks","mni2func_mask_dil_erode1_bin.nii.gz")

## Qval_diff WORKFLOW ##

contrasts = [1, 2, 3] 

for contrast in contrasts: 

	# collecting copes from feat directories (for those subs that had across_runs)
	copes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_qval_diff_dirs_all_subs]
	varcopes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_qval_diff_dirs_all_subs]
	tstats = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_qval_diff_dirs_all_subs]

	# appending remaining copes for those subs that had only one run
	for i in range(len(cope_qval_diff_dirs_all_subs)):
		copes.append(os.path.join(cope_qval_diff_dirs_all_subs[i],'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)))
		varcopes.append(os.path.join(cope_qval_diff_dirs_all_subs[i],'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast)))
		tstats.append(os.path.join(cope_qval_diff_dirs_all_subs[i],'workflow','FE','stats','%s%i.nii.gz'%('tstat',contrast)))

	# Create workflow
	Parkflow_group_qval_diff = Workflow(name='workflow')
	Parkflow_group_qval_diff.base_dir = os.path.join(group_dir,"stimulus_QvalDiff","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_qval_diff.base_dir):
		os.makedirs(Parkflow_group_qval_diff.base_dir)

	# Create nodes

	copemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=copes),
		name='copemerge')
	varcopemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=varcopes),
		name='varcopemerge')
	level2model = Node(interface=fsl.L2Model(
		num_copes=len(copes)),
		name='l2model')
	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile),
		infer_outliers=True,
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_qval_diff.base_dir,'stats',))

	
	Parkflow_group_qval_diff.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(level2model,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_qval_diff.write_graph(graph2use='colored')
	Parkflow_group_qval_diff.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_qval_diff.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_qval_diff_dirs_all_subs) + len(cope_qval_diff_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()


## Choice WORKFLOW ##

# contrasts = [1, 2, 3, 4, 5, 6] 

# for contrast in contrasts: 

# 	# collecting copes from feat directories (for those subs that had across_runs)
# 	copes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_choice_dirs_all_subs]
# 	varcopes = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_choice_dirs_all_subs]
# 	tstats = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_choice_dirs_all_subs]

# 	# appending remaining copes for those subs that had only one run
# 	for i in range(len(cope_qval_diff_dirs_all_subs)):
# 		copes.append(os.path.join(cope_choice_dirs_all_subs[i],'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)))
# 		varcopes.append(os.path.join(cope_choice_dirs_all_subs[i],'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast)))
# 		tstats.append(os.path.join(cope_choice_dirs_all_subs[i],'workflow','FE','stats','%s%i.nii.gz'%('tstat',contrast)))

# 	# Create workflow
# 	Parkflow_group_choice = Workflow(name='workflow')
# 	Parkflow_group_choice.base_dir = os.path.join(group_dir,"choice_QvalDiff","cope"+str(contrast))

# 	if not os.path.exists(Parkflow_group_choice.base_dir):
# 		os.makedirs(Parkflow_group_choice.base_dir)

# 	# Create nodes

# 	copemerge = Node(interface=fsl.Merge(
# 		dimension='t',
# 		in_files=copes),
# 		name='copemerge')
# 	varcopemerge = Node(interface=fsl.Merge(
# 		dimension='t',
# 		in_files=varcopes),
# 		name='varcopemerge')
# 	level2model = Node(interface=fsl.L2Model(
# 		num_copes=len(copes)),
# 		name='l2model')
# 	flame1=Node(interface=fsl.FLAMEO(
# 		run_mode='flame1',
# 		mask_file=groupmaskfile),
# 		#infer_outliers=True,
# 		name='flame1',
# 		stats_dir=os.path.join(Parkflow_group_choice.base_dir,'stats',))

	
# 	Parkflow_group_choice.connect([(copemerge,flame1,[('merged_file','cope_file')]),
# 					(varcopemerge,flame1,[('merged_file','var_cope_file')]),
# 					(level2model,flame1,[('design_mat','design_file'),
# 									('design_con','t_con_file'),
# 									('design_grp','cov_split_file')])
# 					,
# 					])

# 	Parkflow_group_choice.write_graph(graph2use='colored')
# 	Parkflow_group_choice.run()

# 	# Cluster correction
# 	os.chdir(os.path.join(Parkflow_group_choice.base_dir,'workflow','flame1','stats'))

# 	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_choice_dirs_all_subs) + len(cope_choice_dirs_all_subs)-1)
# 	smooth = os.popen(smoothcmd).read().split("\n")
# 	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

# 	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
# 	clusterout = os.popen(clustercmd).read()
# 	f1=open('thres_cluster_2.3_fwe_table.txt','w+')
# 	f1.write(clusterout)
# 	f1.close()
