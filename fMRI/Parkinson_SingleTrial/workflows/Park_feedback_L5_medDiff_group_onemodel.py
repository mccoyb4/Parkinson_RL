
from nipype.interfaces import fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import os
#from IPython import embed as shell

# Adjust smooth variables to run required group analysis
smooth = str(sys.argv[1]) #'0' # use smoothed output or not? 0 = no smooth, 1 = smooth
ae_or_lisa = str(sys.argv[2]) # 'ae' or 'lisa'

subjects = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-219','sub-220','sub-221','sub-222','sub-223','sub-224']

if ae_or_lisa == 'ae':
	
	datadir='/home/shared/2016/Parkinson/data/pd'

	if smooth == '0':
		group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd/group/med_diff"
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd"
	elif smooth == '1':
		group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd/group/med_diff"
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_4/pd"		

elif ae_or_lisa == 'lisa':

	datadir='/nfs/bromccoy/Parkinson/data/pd'

	if smooth == '0':
		group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd/group/med_diff"
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd"
	elif smooth == '1':
		group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/pd/group/med_diff"
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_4/pd"	

		
if not os.path.exists(group_dir):
	os.makedirs(group_dir)

feat_rpe_dirs_all_subs, feat_arpe_dirs_all_subs, feat_qval_dirs_all_subs = [[] for i in range(3)]

k_beta,k_ag,k_al = [[] for i in range(3)]

for sb in range(len(subjects)):

	sub_workflow_dir = os.path.join(workflow_dir,subjects[sb],"med_diff")

	sub_rpe_feat_dir = os.path.join(sub_workflow_dir,"feedback_RPE")
	sub_arpe_feat_dir = os.path.join(sub_workflow_dir,"feedback_aRPE")
	sub_qval_feat_dir = os.path.join(sub_workflow_dir,"feedback_qval")

	feat_rpe_dirs_all_subs.append(sub_rpe_feat_dir)
	feat_arpe_dirs_all_subs.append(sub_arpe_feat_dir)
	feat_qval_dirs_all_subs.append(sub_qval_feat_dir)

	# Read RL beta, alpha gain and alpha loss difference parameters (can use as covariate later)
	# Can get events file from any session/run of this subject
	events_filename = os.path.join(datadir,subjects[sb],"on","func",subjects[sb]+"_task-train_run-1_events.tsv")	
	events = pd.read_csv(events_filename, sep="\t")
	k_beta.append(events['b_med_diff'].iloc[0])
	k_ag.append(events['ag_med_diff'].iloc[0])
	k_al.append(events['al_med_diff'].iloc[0])

k_beta=np.array(k_beta)
k_ag=np.array(k_ag)
k_al=np.array(k_al)

k_beta_demeaned = k_beta - k_beta[~np.isnan(k_beta)].mean()
k_ag_demeaned = k_ag - k_ag[~np.isnan(k_ag)].mean()
k_al_demeaned = k_al - k_al[~np.isnan(k_al)].mean()

groupmaskfile = os.path.join(workflow_dir,"masks","mni2func_mask_erode1.nii.gz")


# Prepare copes : ON-OFF
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

## Feedback and RPE workflow ##

# ON - OFF

# Create workflow
Parkflow_group_rpe = Workflow(name='workflow')
Parkflow_group_rpe.base_dir = os.path.join(group_dir,"on_off","feedback_RPE","one_model")

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

flame1=Node(interface=fsl.FLAMEO(
	run_mode='flame1',
	mask_file=groupmaskfile),
	#infer_outliers=True),
	name='flame1',
	stats_dir=os.path.join(Parkflow_group_rpe.base_dir,'stats'))


Parkflow_group_rpe.connect([(copemerge,flame1,[('merged_file','cope_file')]),
				(varcopemerge,flame1,[('merged_file','var_cope_file')]),
				(multregmodel,flame1,[('design_mat','design_file'),
								('design_con','t_con_file'),
								('design_grp','cov_split_file')])
				,
				])

Parkflow_group_rpe.write_graph(graph2use='colored')
Parkflow_group_rpe.run()

# Cluster correction
os.chdir(os.path.join(Parkflow_group_rpe.base_dir,'workflow','flame1','stats'))

smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_rpe_dirs_all_subs)-1)
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

