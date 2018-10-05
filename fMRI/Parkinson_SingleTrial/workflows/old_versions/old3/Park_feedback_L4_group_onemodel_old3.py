
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

	subjects = ['sub-111','sub-112','sub-113','sub-114','sub-115','sub-116','sub-117','sub-118','sub-119','sub-120','sub-121','sub-123','sub-124','sub-126','sub-127','sub-128','sub-129','sub-130','sub-131','sub-132','sub-133']	#['sub-108','sub-111','sub-112','sub-113','sub-114','sub-115','sub-116','sub-117','sub-118','sub-119','sub-120','sub-121','sub-123','sub-124','sub-126','sub-127','sub-128','sub-129','sub-130','sub-131','sub-132','sub-133']

	# Name directories for input and output
	if ae_or_lisa == 'ae':

		fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/hc_syn-sdc'

		if smooth == '0':
			group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_3/hc/group"
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_3/hc"
		elif smooth == '1':
			group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_3/hc/group"
			workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_3/hc"

	elif ae_or_lisa == 'lisa':

		fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/hc_syn-sdc'

		if smooth == '0':
			group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_3/hc/group"
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_3/hc"
		elif smooth == '1':
			group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_3/hc/group"
			workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_3/hc"
	
elif hc_or_pd == 'pd':

	subjects = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-219','sub-220','sub-221','sub-222','sub-223','sub-224'] 

	# Name directories for input and output

	if pd_on_off == 'on':

		if ae_or_lisa == 'ae':

			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
			if smooth == '0':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_3/pd/group/on"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_3/pd"
			elif smooth == '1':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_3/pd/group/on"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_3/pd"

		elif ae_or_lisa == 'lisa':

			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
			if smooth == '0':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_3/pd/group/on"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_3/pd"
			elif smooth == '1':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_3/pd/group/on"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_3/pd"
		
	elif pd_on_off == 'off':

		if ae_or_lisa == 'ae':
		
			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_off_syn-sdc'
			if smooth == '0':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_3/pd/group/off"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_3/pd"
			elif smooth == '1':
				group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_3/pd/group/off"
				workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_3/pd"

		elif ae_or_lisa == 'lisa':
			
			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_off_syn-sdc'
			if smooth == '0':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_3/pd/group/off"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_3/pd"
			elif smooth == '1':
				group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_3/pd/group/off"
				workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_3/pd"


if not os.path.exists(group_dir):
	os.makedirs(group_dir)		

feat_rpe_dirs_all_subs, feat_arpe_dirs_all_subs, feat_qval_dirs_all_subs = [[] for i in range(3)]
cope_rpe_dirs_all_subs, cope_arpe_dirs_all_subs, cope_qval_dirs_all_subs = [[] for i in range(3)]

for sb in range(len(subjects)):

	if hc_or_pd == 'hc':
		sub_workflow_dir = os.path.join(workflow_dir,subjects[sb])
	elif hc_or_pd == 'pd':
		sub_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off)

	if subjects[sb] not in ('sub-203','sub-209', 'sub-213', 'sub-215', 'sub-115'):

		sub_rpe_feat_dir = os.path.join(sub_workflow_dir,"train","across_runs","feedback_RPE")
		sub_arpe_feat_dir = os.path.join(sub_workflow_dir,"train","across_runs","feedback_aRPE")
		sub_qval_feat_dir = os.path.join(sub_workflow_dir,"train","across_runs","feedback_qval")

		feat_rpe_dirs_all_subs.append(sub_rpe_feat_dir)
		feat_arpe_dirs_all_subs.append(sub_arpe_feat_dir)	
		feat_qval_dirs_all_subs.append(sub_qval_feat_dir)	
	
	elif subjects[sb] in ('sub-203','sub-209','sub-215'):
	
		sub_rpe_cope_dir = os.path.join(sub_workflow_dir,"train","run-1","feedback_RPE")
		sub_arpe_cope_dir = os.path.join(sub_workflow_dir,"train","run-1","feedback_aRPE")
		sub_qval_cope_dir = os.path.join(sub_workflow_dir,"train","run-1","feedback_qval")

		cope_rpe_dirs_all_subs.append(sub_rpe_cope_dir)		
		cope_arpe_dirs_all_subs.append(sub_arpe_cope_dir)	
		cope_qval_dirs_all_subs.append(sub_qval_cope_dir)

	elif subjects[sb] in ('sub-213','sub-115'):

		sub_rpe_cope_dir = os.path.join(sub_workflow_dir,"train","run-2","feedback_RPE")
		sub_arpe_cope_dir = os.path.join(sub_workflow_dir,"train","run-2","feedback_aRPE")
		sub_qval_cope_dir = os.path.join(sub_workflow_dir,"train","run-2","feedback_qval")

		cope_rpe_dirs_all_subs.append(sub_rpe_cope_dir)		
		cope_arpe_dirs_all_subs.append(sub_arpe_cope_dir)	
		cope_qval_dirs_all_subs.append(sub_qval_cope_dir)	

groupmaskfile = os.path.join(workflow_dir,"masks","mni2func_mask_dil.nii.gz")


#### RPE : PREPARE COPES ###
copes, varcopes = [[] for i in range(2)]
EV_feedback, EV_rpe_pos,EV_rpe_neg = [[] for i in range(3)]

for x in feat_rpe_dirs_all_subs:
	
	# FEEDBACK COPE (cope 1)
	copes.append(os.path.join(x,'cope1','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope1','workflow','FE','stats','varcope1.nii.gz'))
	EV_feedback.append(1)
	EV_rpe_pos.append(0)
	EV_rpe_neg.append(0)

	# RPE+ COPE (cope 2)
	copes.append(os.path.join(x,'cope2','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope2','workflow','FE','stats','varcope1.nii.gz'))
	EV_feedback.append(0)
	EV_rpe_pos.append(1)
	EV_rpe_neg.append(0)

	# RPE- COPE (cope 3)
	copes.append(os.path.join(x,'cope3','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope3','workflow','FE','stats','varcope1.nii.gz'))
	EV_feedback.append(0)
	EV_rpe_pos.append(0)
	EV_rpe_neg.append(1)


# appending remaining copes for those subs that had only one run
for i in range(len(cope_rpe_dirs_all_subs)):

	# FEEDBACK COPE (cope 1)
	copes.append(os.path.join(cope_rpe_dirs_all_subs[i],'workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(cope_rpe_dirs_all_subs[i],'workflow','FE','stats','varcope1.nii.gz'))
	EV_feedback.append(1)
	EV_rpe_pos.append(0)
	EV_rpe_neg.append(0)

	# RPE+ COPE (cope 2)
	copes.append(os.path.join(cope_rpe_dirs_all_subs[i],'workflow','FE','stats','cope2.nii.gz'))
	varcopes.append(os.path.join(cope_rpe_dirs_all_subs[i],'workflow','FE','stats','varcope2.nii.gz'))
	EV_feedback.append(0)
	EV_rpe_pos.append(1)
	EV_rpe_neg.append(0)

	# RPE- COPE (cope 3)
	copes.append(os.path.join(cope_rpe_dirs_all_subs[i],'workflow','FE','stats','cope3.nii.gz'))
	varcopes.append(os.path.join(cope_rpe_dirs_all_subs[i],'workflow','FE','stats','varcope3.nii.gz'))
	EV_feedback.append(0)
	EV_rpe_pos.append(0)
	EV_rpe_neg.append(1)


# RPE WORKFLOW
Parkflow_group_rpe = Workflow(name='workflow')
Parkflow_group_rpe.base_dir = os.path.join(group_dir,"feedback_RPE","one_model")

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

feedback_tcont = ['feedback', 'T',['reg1','reg2','reg3'],[1,0,0]]
rpe_pos_tcont = ['rpe+', 'T',['reg1','reg2','reg3'],[0,1,0]]
rpe_neg_tcont = ['rpe-', 'T',['reg1','reg2','reg3'],[0,0,1]]
rpe_pos_feed_tcont = ['rpe+>feedback', 'T',['reg1','reg2','reg3'],[-1,1,0]]
rpe_pos_neg_tcont = ['rpe+>rpe-', 'T',['reg1','reg2','reg3'],[0,1,-1]]
rpe_neg_pos_tcont = ['rpe->rpe+', 'T',['reg1','reg2','reg3'],[0,-1,1]]
rpe_neg_feed_tcont = ['rpe->feedback', 'T',['reg1','reg2','reg3'],[-1,0,1]]

multregmodel.inputs.contrasts = [feedback_tcont, rpe_pos_tcont, rpe_neg_tcont,rpe_pos_feed_tcont,rpe_pos_neg_tcont,rpe_neg_pos_tcont,rpe_neg_feed_tcont]
multregmodel.inputs.regressors = dict(reg1=list(EV_feedback),reg2=list(EV_rpe_pos),reg3=list(EV_rpe_neg))

flame12=Node(interface=fsl.FLAMEO(
	run_mode='flame12',
	mask_file=groupmaskfile,
	infer_outliers=True),
	name='flame12',
	stats_dir=os.path.join(Parkflow_group_rpe.base_dir,'stats'))


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

smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_rpe_dirs_all_subs) + len(cope_rpe_dirs_all_subs)-1)
smooth = os.popen(smoothcmd).read().split("\n")
smoothn = [x.split(' ')[1] for x in smooth[:-1]]

clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat1.txt','w+')
f1.write(clusterout)
f1.close()

clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat2.txt','w+')
f1.write(clusterout)
f1.close()

clustercmd = 'cluster -i zstat3 -c cope3 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat3 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat3.txt','w+')
f1.write(clusterout)
f1.close()


#### RPE_update : PREPARE COPES ###
# EV_feedback, EV_rpe_pos, EV_rpe_neg are the same as in RPE case (just applying '1's to the correct listings of copes)
copes, varcopes = [[] for i in range(2)]

for x in feat_arpe_dirs_all_subs:
	
	# FEEDBACK COPE (cope 1)
	copes.append(os.path.join(x,'cope1','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope1','workflow','FE','stats','varcope1.nii.gz'))

	# aRPE+ COPE (cope 2)
	copes.append(os.path.join(x,'cope2','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope2','workflow','FE','stats','varcope1.nii.gz'))

	# aRPE- COPE (cope 3)
	copes.append(os.path.join(x,'cope3','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope3','workflow','FE','stats','varcope1.nii.gz'))

# appending remaining copes for those subs that had only one run
for i in range(len(cope_arpe_dirs_all_subs)):

	# FEEDBACK COPE (cope 1)
	copes.append(os.path.join(cope_arpe_dirs_all_subs[i],'workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(cope_arpe_dirs_all_subs[i],'workflow','FE','stats','varcope1.nii.gz'))

	# RPE+ COPE (cope 2)
	copes.append(os.path.join(cope_arpe_dirs_all_subs[i],'workflow','FE','stats','cope2.nii.gz'))
	varcopes.append(os.path.join(cope_arpe_dirs_all_subs[i],'workflow','FE','stats','varcope2.nii.gz'))

	# RPE- COPE (cope 3)
	copes.append(os.path.join(cope_arpe_dirs_all_subs[i],'workflow','FE','stats','cope3.nii.gz'))
	varcopes.append(os.path.join(cope_arpe_dirs_all_subs[i],'workflow','FE','stats','varcope3.nii.gz'))


# RPE_update (alpha * RPE) WORKFLOW
Parkflow_group_arpe = Workflow(name='workflow')
Parkflow_group_arpe.base_dir = os.path.join(group_dir,"feedback_aRPE","one_model")

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

feedback_tcont = ['feedback', 'T',['reg1','reg2','reg3'],[1,0,0]]
arpe_pos_tcont = ['arpe+', 'T',['reg1','reg2','reg3'],[0,1,0]]
arpe_neg_tcont = ['arpe-', 'T',['reg1','reg2','reg3'],[0,0,1]]
arpe_pos_feed_tcont = ['arpe+>feedback', 'T',['reg1','reg2','reg3'],[-1,1,0]]
arpe_pos_neg_tcont = ['arpe+>arpe-', 'T',['reg1','reg2','reg3'],[0,1,-1]]
arpe_neg_pos_tcont = ['arpe->arpe+', 'T',['reg1','reg2','reg3'],[0,-1,1]]
arpe_neg_feed_tcont = ['arpe->feedback', 'T',['reg1','reg2','reg3'],[-1,0,1]]

multregmodel.inputs.contrasts = [feedback_tcont, arpe_pos_tcont, arpe_neg_tcont,arpe_pos_feed_tcont,arpe_pos_neg_tcont,arpe_neg_pos_tcont,arpe_neg_feed_tcont]
multregmodel.inputs.regressors = dict(reg1=list(EV_feedback),reg2=list(EV_rpe_pos),reg3=list(EV_rpe_neg)) # just '1's for the right copes, so can use same EV_rpe_pos/neg lists here

flame12=Node(interface=fsl.FLAMEO(
	run_mode='flame12',
	mask_file=groupmaskfile,
	infer_outliers=True),
	name='flame12',
	stats_dir=os.path.join(Parkflow_group_arpe.base_dir,'stats'))


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

smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_arpe_dirs_all_subs) + len(cope_arpe_dirs_all_subs)-1)
smooth = os.popen(smoothcmd).read().split("\n")
smoothn = [x.split(' ')[1] for x in smooth[:-1]]

clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat1.txt','w+')
f1.write(clusterout)
f1.close()

clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat2.txt','w+')
f1.write(clusterout)
f1.close()

clustercmd = 'cluster -i zstat3 -c cope3 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat3 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat3.txt','w+')
f1.write(clusterout)
f1.close()



#### Qval_update : PREPARE COPES ###
# EV_feedback, EV_rpe_pos, EV_rpe_neg are the same as in RPE case (just applying '1's to the correct listings of copes)
copes, varcopes = [[] for i in range(2)]

for x in feat_qval_dirs_all_subs:
	
	# FEEDBACK COPE (cope 1)
	copes.append(os.path.join(x,'cope1','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope1','workflow','FE','stats','varcope1.nii.gz'))

	# qval+ COPE (cope 2)
	copes.append(os.path.join(x,'cope2','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope2','workflow','FE','stats','varcope1.nii.gz'))

	# qval- COPE (cope 3)
	copes.append(os.path.join(x,'cope3','workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(x,'cope3','workflow','FE','stats','varcope1.nii.gz'))

# appending remaining copes for those subs that had only one run
for i in range(len(cope_qval_dirs_all_subs)):

	# FEEDBACK COPE (cope 1)
	copes.append(os.path.join(cope_qval_dirs_all_subs[i],'workflow','FE','stats','cope1.nii.gz'))
	varcopes.append(os.path.join(cope_qval_dirs_all_subs[i],'workflow','FE','stats','varcope1.nii.gz'))

	# qval+ COPE (cope 2)
	copes.append(os.path.join(cope_qval_dirs_all_subs[i],'workflow','FE','stats','cope2.nii.gz'))
	varcopes.append(os.path.join(cope_qval_dirs_all_subs[i],'workflow','FE','stats','varcope2.nii.gz'))

	# qval- COPE (cope 3)
	copes.append(os.path.join(cope_qval_dirs_all_subs[i],'workflow','FE','stats','cope3.nii.gz'))
	varcopes.append(os.path.join(cope_qval_dirs_all_subs[i],'workflow','FE','stats','varcope3.nii.gz'))


# Qval_update WORKFLOW
Parkflow_group_qval = Workflow(name='workflow')
Parkflow_group_qval.base_dir = os.path.join(group_dir,"feedback_qval","one_model")

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

feedback_tcont = ['feedback', 'T',['reg1','reg2','reg3'],[1,0,0]]
qval_pos_tcont = ['qval+', 'T',['reg1','reg2','reg3'],[0,1,0]]
qval_neg_tcont = ['qval-', 'T',['reg1','reg2','reg3'],[0,0,1]]
qval_pos_feed_tcont = ['qval+>feedback', 'T',['reg1','reg2','reg3'],[-1,1,0]]
qval_pos_neg_tcont = ['qval+>qval-', 'T',['reg1','reg2','reg3'],[0,1,-1]]
qval_neg_pos_tcont = ['qval->qval+', 'T',['reg1','reg2','reg3'],[0,-1,1]]
qval_neg_feed_tcont = ['qval->feedback', 'T',['reg1','reg2','reg3'],[-1,0,1]]

multregmodel.inputs.contrasts = [feedback_tcont, qval_pos_tcont, qval_neg_tcont,qval_pos_feed_tcont,qval_pos_neg_tcont,qval_neg_pos_tcont,qval_neg_feed_tcont]
multregmodel.inputs.regressors = dict(reg1=list(EV_feedback),reg2=list(EV_rpe_pos),reg3=list(EV_rpe_neg)) # just '1's for the right copes, so can use same EV_rpe_pos/neg lists here

flame12=Node(interface=fsl.FLAMEO(
	run_mode='flame12',
	mask_file=groupmaskfile,
	infer_outliers=True),
	name='flame12',
	stats_dir=os.path.join(Parkflow_group_qval.base_dir,'stats'))


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

smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_qval_dirs_all_subs) + len(cope_qval_dirs_all_subs)-1)
smooth = os.popen(smoothcmd).read().split("\n")
smoothn = [x.split(' ')[1] for x in smooth[:-1]]

clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat1.txt','w+')
f1.write(clusterout)
f1.close()

clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat2.txt','w+')
f1.write(clusterout)
f1.close()

clustercmd = 'cluster -i zstat3 -c cope3 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat3 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
clusterout = os.popen(clustercmd).read()
f1=open('thres_cluster_2.3_fwe_table_zstat3.txt','w+')
f1.write(clusterout)
f1.close()
