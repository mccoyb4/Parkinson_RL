
from nipype.interfaces import fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
#from preproc_functions import natural_sort
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import os
from IPython import embed as shell

# Adjust smooth variables to run required group analysis
smooth = '1' # use smoothed output or not? 0 = no smooth, 1 = smooth
hc_v_onoff = 'on' # 'on' is hc vs pd on, 'off' is hc vs pd off

subjects_hc = ['sub-108','sub-111','sub-112','sub-113','sub-114','sub-115','sub-116','sub-117','sub-118','sub-119','sub-120','sub-121','sub-123','sub-124','sub-126','sub-127','sub-128','sub-129','sub-130','sub-131','sub-132','sub-133']
subjects_pd = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-219','sub-220','sub-221','sub-222','sub-223','sub-224']

datadir_hc='/home/shared/2016/Parkinson/data/hc'
datadir_pd='/home/shared/2016/Parkinson/data/pd'

all_subjects = subjects_hc + subjects_pd

if smooth == '0':

	group_dir_hc = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/hc/group"
	workflow_dir_hc = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/hc"

	group_dir_pd = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd/group"
	workflow_dir_pd = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd"

	group_diff_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/group_diff"

elif smooth == '1':

	group_dir_hc = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/hc/group/"
	workflow_dir_hc = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/hc"		

	group_dir_pd = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd/group/"
	workflow_dir_pd = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd"	

	group_diff_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/group_diff"
	
if not os.path.exists(group_dir_hc):
	os.makedirs(group_dir_hc)
if not os.path.exists(group_dir_pd):
	os.makedirs(group_dir_pd)
if not os.path.exists(group_diff_dir):
	os.makedirs(group_diff_dir)

feat_rpe_dirs_all_subs_hc, feat_arpe_dirs_all_subs_hc, feat_qval_dirs_all_subs_hc = [[] for i in range(3)]
feat_rpe_dirs_all_subs_pd, feat_arpe_dirs_all_subs_pd, feat_qval_dirs_all_subs_pd = [[] for i in range(3)]

cope_rpe_dirs_all_subs_hc, cope_arpe_dirs_all_subs_hc, cope_qval_dirs_all_subs_hc = [[] for i in range(3)]
cope_rpe_dirs_all_subs_pd, cope_arpe_dirs_all_subs_pd, cope_qval_dirs_all_subs_pd = [[] for i in range(3)]

feat_and_cope_rpe_dirs_hc, feat_and_cope_arpe_dirs_hc, feat_and_cope_qval_dirs_hc = [[] for i in range(3)]
feat_and_cope_rpe_dirs_pd, feat_and_cope_arpe_dirs_pd, feat_and_cope_qval_dirs_pd = [[] for i in range(3)]

for sb in range(len(subjects_hc)):

	sub_workflow_dir_hc = os.path.join(workflow_dir_hc,subjects_hc[sb])

	if subjects_hc[sb] not in ('sub-115'):
		sub_rpe_feat_dir = os.path.join(sub_workflow_dir_hc,"train","across_runs","feedback_RPE")
		sub_arpe_feat_dir = os.path.join(sub_workflow_dir_hc,"train","across_runs","feedback_aRPE")
		sub_qval_feat_dir = os.path.join(sub_workflow_dir_hc,"train","across_runs","feedback_qval")

		feat_and_cope_rpe_dirs_hc.append(sub_rpe_feat_dir)
		feat_and_cope_arpe_dirs_hc.append(sub_arpe_feat_dir)
		feat_and_cope_qval_dirs_hc.append(sub_qval_feat_dir)

	elif subjects_hc[sb] in ('sub-115'):
		sub_rpe_cope_dir = os.path.join(sub_workflow_dir_hc,"train","run-2","feedback_RPE")
		sub_arpe_cope_dir = os.path.join(sub_workflow_dir_hc,"train","run-2","feedback_aRPE")
		sub_qval_cope_dir = os.path.join(sub_workflow_dir_hc,"train","run-2","feedback_qval")

		feat_and_cope_rpe_dirs_hc.append(sub_rpe_cope_dir)
		feat_and_cope_arpe_dirs_hc.append(sub_arpe_cope_dir)
		feat_and_cope_qval_dirs_hc.append(sub_qval_cope_dir)


for sb in range(len(subjects_pd)):

	sub_workflow_dir_pd = os.path.join(workflow_dir_pd,subjects_pd[sb],hc_v_onoff)

	if subjects_pd[sb] not in ('sub-203','sub-209', 'sub-213', 'sub-215'):
		sub_rpe_feat_dir = os.path.join(sub_workflow_dir_pd,"train","across_runs","feedback_RPE")
		sub_arpe_feat_dir = os.path.join(sub_workflow_dir_pd,"train","across_runs","feedback_aRPE")
		sub_qval_feat_dir = os.path.join(sub_workflow_dir_pd,"train","across_runs","feedback_qval")

		feat_and_cope_rpe_dirs_pd.append(sub_rpe_feat_dir)
		feat_and_cope_arpe_dirs_pd.append(sub_arpe_feat_dir)
		feat_and_cope_qval_dirs_pd.append(sub_qval_feat_dir)

	elif subjects_pd[sb] in ('sub-203','sub-209','sub-215'):
		sub_rpe_cope_dir = os.path.join(sub_workflow_dir_pd,"train","run-1","feedback_RPE")
		sub_arpe_cope_dir = os.path.join(sub_workflow_dir_pd,"train","run-1","feedback_aRPE")
		sub_qval_cope_dir = os.path.join(sub_workflow_dir_pd,"train","run-1","feedback_qval")

		feat_and_cope_rpe_dirs_pd.append(sub_rpe_cope_dir)
		feat_and_cope_arpe_dirs_pd.append(sub_arpe_cope_dir)
		feat_and_cope_qval_dirs_pd.append(sub_qval_cope_dir)

	elif subjects_pd[sb] in ('sub-213'):
		sub_rpe_cope_dir = os.path.join(sub_workflow_dir_pd,"train","run-2","feedback_RPE")
		sub_arpe_cope_dir = os.path.join(sub_workflow_dir_pd,"train","run-2","feedback_aRPE")
		sub_qval_cope_dir = os.path.join(sub_workflow_dir_pd,"train","run-2","feedback_qval")

		feat_and_cope_rpe_dirs_pd.append(sub_rpe_cope_dir)
		feat_and_cope_arpe_dirs_pd.append(sub_arpe_cope_dir)
		feat_and_cope_qval_dirs_pd.append(sub_qval_cope_dir)

# Concatenate folder names across HC and PD
feat_and_cope_rpe_dirs_all_subs = feat_and_cope_rpe_dirs_hc + feat_and_cope_rpe_dirs_pd
feat_and_cope_arpe_dirs_all_subs = feat_and_cope_arpe_dirs_hc + feat_and_cope_arpe_dirs_pd
feat_and_cope_qval_dirs_all_subs = feat_and_cope_qval_dirs_hc + feat_and_cope_qval_dirs_pd

groupmaskfile = os.path.join(workflow_dir_hc,"masks","mni2func_mask_dil_erode1_bin.nii.gz")

HC_EV = list(np.ones(len(subjects_hc))) + list(np.zeros(len(subjects_pd)))
PD_EV = list(np.zeros(len(subjects_hc))) + list(np.ones(len(subjects_pd)))
group_list = list(np.ones(len(subjects_hc))) + list(2*np.ones(len(subjects_pd)))
group_list = [int(x) for x in group_list] #changing elements to int

## Feedback and RPE workflow ##

# HC v PD

contrasts = [1,2,3]

for contrast in contrasts: 

	copes, varcopes = [[] for i in range(2)]

	for sb in range(len(all_subjects)):
		if all_subjects[sb] not in ('sub-203','sub-209', 'sub-213','sub-215','sub-115'):

			copes.append(os.path.join(feat_and_cope_rpe_dirs_all_subs[sb],'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz'))
			varcopes.append(os.path.join(feat_and_cope_rpe_dirs_all_subs[sb],'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz'))

		elif all_subjects[sb] in ('sub-203','sub-209', 'sub-213','sub-215','sub-115'):
			
			copes.append(os.path.join(feat_and_cope_rpe_dirs_all_subs[sb],'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)))
			varcopes.append(os.path.join(feat_and_cope_rpe_dirs_all_subs[sb],'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast)))
		
	# Create workflow
	Parkflow_group_rpe = Workflow(name='workflow')
	Parkflow_group_rpe.base_dir = os.path.join(group_diff_dir,"hc_pd"+hc_v_onoff,"feedback_RPE","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_rpe.base_dir):
		os.makedirs(Parkflow_group_rpe.base_dir)

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

	hcminpd_tcont = ['hc-pd'+hc_v_onoff, 'T',['reg1','reg2'],[1,-1]]
	pdminhc_tcont = ['pd'+hc_v_onoff+'-hc', 'T',['reg1','reg2'],[-1,1]]
	
	multregmodel.inputs.contrasts = [hcminpd_tcont, pdminhc_tcont]
	multregmodel.inputs.regressors = dict(reg1=HC_EV,reg2=PD_EV)
	multregmodel.inputs.groups = group_list

	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_rpe.base_dir,'stats'))

	# Use level2model if not using covariate
	Parkflow_group_rpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(multregmodel,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_rpe.write_graph(graph2use='colored')
	Parkflow_group_rpe.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_rpe.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_and_cope_rpe_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_zstat1_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()	

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_zstat2_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()


## Feedback and alpha*RPE workflow ##

# HC v PD

contrasts = [1,2,3]

for contrast in contrasts: 

	copes, varcopes = [[] for i in range(2)]

	for sb in range(len(all_subjects)):
		if all_subjects[sb] not in ('sub-203','sub-209', 'sub-213','sub-215','sub-115'):

			copes.append(os.path.join(feat_and_cope_arpe_dirs_all_subs[sb],'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz'))
			varcopes.append(os.path.join(feat_and_cope_arpe_dirs_all_subs[sb],'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz'))

		elif all_subjects[sb] in ('sub-203','sub-209', 'sub-213','sub-215','sub-115'):
			
			copes.append(os.path.join(feat_and_cope_arpe_dirs_all_subs[sb],'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)))
			varcopes.append(os.path.join(feat_and_cope_arpe_dirs_all_subs[sb],'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast)))
		
	# Create workflow
	Parkflow_group_arpe = Workflow(name='workflow')
	Parkflow_group_arpe.base_dir = os.path.join(group_diff_dir,"hc_pd"+hc_v_onoff,"feedback_aRPE","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_arpe.base_dir):
		os.makedirs(Parkflow_group_arpe.base_dir)

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

	hcminpd_tcont = ['hc-pd'+hc_v_onoff, 'T',['reg1','reg2'],[1,-1]]
	pdminhc_tcont = ['pd'+hc_v_onoff+'-hc', 'T',['reg1','reg2'],[-1,1]]
	
	multregmodel.inputs.contrasts = [hcminpd_tcont, pdminhc_tcont]
	multregmodel.inputs.regressors = dict(reg1=HC_EV,reg2=PD_EV)
	multregmodel.inputs.groups = group_list

	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_arpe.base_dir,'stats'))

	# Use level2model if not using covariate
	Parkflow_group_arpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(multregmodel,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_arpe.write_graph(graph2use='colored')
	Parkflow_group_arpe.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_arpe.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_and_cope_arpe_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_zstat1_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()	

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_zstat2_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()



## Feedback and qval workflow ##

# HC v PD

contrasts = [1,2,3]

for contrast in contrasts: 

	copes, varcopes = [[] for i in range(2)]

	for sb in range(len(all_subjects)):
		if all_subjects[sb] not in ('sub-203','sub-209', 'sub-213','sub-215','sub-115'):

			copes.append(os.path.join(feat_and_cope_qval_dirs_all_subs[sb],'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz'))
			varcopes.append(os.path.join(feat_and_cope_qval_dirs_all_subs[sb],'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz'))

		elif all_subjects[sb] in ('sub-203','sub-209', 'sub-213','sub-215','sub-115'):
			
			copes.append(os.path.join(feat_and_cope_qval_dirs_all_subs[sb],'workflow','FE','stats','%s%i.nii.gz'%('cope',contrast)))
			varcopes.append(os.path.join(feat_and_cope_qval_dirs_all_subs[sb],'workflow','FE','stats','%s%i.nii.gz'%('varcope',contrast)))
		
	# Create workflow
	Parkflow_group_qval = Workflow(name='workflow')
	Parkflow_group_qval.base_dir = os.path.join(group_diff_dir,"hc_pd"+hc_v_onoff,"feedback_qval","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_qval.base_dir):
		os.makedirs(Parkflow_group_qval.base_dir)

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

	hcminpd_tcont = ['hc-pd'+hc_v_onoff, 'T',['reg1','reg2'],[1,-1]]
	pdminhc_tcont = ['pd'+hc_v_onoff+'-hc', 'T',['reg1','reg2'],[-1,1]]
	
	multregmodel.inputs.contrasts = [hcminpd_tcont, pdminhc_tcont]
	multregmodel.inputs.regressors = dict(reg1=HC_EV,reg2=PD_EV)
	multregmodel.inputs.groups = group_list

	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_qval.base_dir,'stats'))

	# Use level2model if not using covariate
	Parkflow_group_qval.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(multregmodel,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_qval.write_graph(graph2use='colored')
	Parkflow_group_qval.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_qval.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_and_cope_qval_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_zstat1_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()	

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_zstat2_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()


