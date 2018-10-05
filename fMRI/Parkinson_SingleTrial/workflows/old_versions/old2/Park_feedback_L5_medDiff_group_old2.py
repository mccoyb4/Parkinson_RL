
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

# Put 'sub-219' back in!
subjects = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-220','sub-221','sub-222','sub-223','sub-224']

if ae_or_lisa == 'ae':
	
	datadir='/home/shared/2016/Parkinson/data/pd'

	if smooth == '0':
		group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_2/pd/group/med_diff"
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_2/pd"
	elif smooth == '1':
		group_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_2/pd/group/med_diff"
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/smooth_2/pd"		

elif ae_or_lisa == 'lisa':

	datadir='/nfs/bromccoy/Parkinson/data/pd'

	if smooth == '0':
		group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_2/pd/group/med_diff"
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_2/pd"
	elif smooth == '1':
		group_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_2/pd/group/med_diff"
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/smooth_2/pd"	

		
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

## CREATE ACROSS-SUBJECT MASK ##

groupmaskfile = os.path.join(group_dir,"groupmask.nii.gz")

mask = np.zeros([60,71,60,len(feat_rpe_dirs_all_subs)]) # mask is same for rpe and arpe files
k=0

for sb in subjects:

	maskfile = os.path.join(sub_rpe_feat_dir,"mask.nii.gz")

	if os.path.exists(maskfile):
		masksub = nib.load(maskfile)
		data = masksub.get_data()
		mask[:,:,:,k] = data
		k += 1

mask = mask[:,:,:,:k]

mask = np.mean(mask,axis=3)
mask = np.where(mask>0.8,1,0)
maskimg = nib.Nifti1Image(mask,affine=masksub.affine,header=masksub.header)
maskimg.to_filename(groupmaskfile)


## Feedback and RPE workflow ##

# ON - OFF
contrasts = [1,2,3]

for contrast in contrasts: 

	copes_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_rpe_dirs_all_subs]
	varcopes_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_rpe_dirs_all_subs]
	tstats_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_rpe_dirs_all_subs]

	# Create workflow
	Parkflow_group_rpe = Workflow(name='workflow')
	Parkflow_group_rpe.base_dir = os.path.join(group_dir,"on_off","feedback_RPE","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_rpe.base_dir):
		os.makedirs(Parkflow_group_rpe.base_dir)

	# Create nodes

	copemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=copes_on_off),
		name='copemerge')
	varcopemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=varcopes_on_off),
		name='varcopemerge')
	tstatmerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=tstats_on_off),
		name='tstatmerge')		
	level2model = Node(interface=fsl.L2Model(
		num_copes=len(copes_on_off)),
		name='l2model')

	multregmodel = Node(interface=fsl.MultipleRegressDesign(
		contrasts=[],
		regressors={}),
		name='multregmodel')
	
	multregmodel.inputs.contrasts = [["cope"+str(contrast)+'_mean', 'T',['reg1'],[1]],['k_ag', 'T',['reg2'],[1]]] 
	multregmodel.inputs.regressors = dict(reg1=list(np.ones(len(copes_on_off))),reg2=list(k_ag_demeaned))


	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_rpe.base_dir,'stats',))

	# Use level2model if not using covariate
	Parkflow_group_rpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(level2model,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	# Use multregmodel if using covariate
	# Parkflow_group_rpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
	# 			(varcopemerge,flame12,[('merged_file','var_cope_file')]),
	# 			(multregmodel,flame12,[('design_mat','design_file'),
	# 							('design_con','t_con_file'),
	# 							('design_grp','cov_split_file')])
	# 			,
	# 			])

	Parkflow_group_rpe.write_graph(graph2use='colored')
	Parkflow_group_rpe.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_rpe.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_rpe_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_mean_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()	

	# # Cluster correction for k_ag covariate
	# clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	# clusterout = os.popen(clustercmd).read()
	# f1=open('thres_cluster_k_ag_2.3_fwe_table.txt','w+')
	# f1.write(clusterout)
	# f1.close()

	# # Cluster correction for k_al covariate
	# clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	# clusterout = os.popen(clustercmd).read()
	# f1=open('thres_cluster_k_al_2.3_fwe_table.txt','w+')
	# f1.write(clusterout)
	# f1.close()


# OFF - ON
contrasts = [1,2,3]

for contrast in contrasts: 

	copes_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope2.nii.gz') for x in feat_rpe_dirs_all_subs]
	varcopes_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope2.nii.gz') for x in feat_rpe_dirs_all_subs]
	tstats_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat2.nii.gz') for x in feat_rpe_dirs_all_subs]

	# Create workflow
	Parkflow_group_rpe = Workflow(name='workflow')
	Parkflow_group_rpe.base_dir = os.path.join(group_dir,"off_on","feedback_RPE","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_rpe.base_dir):
		os.makedirs(Parkflow_group_rpe.base_dir)

	# Create nodes

	copemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=copes_off_on),
		name='copemerge')
	varcopemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=varcopes_off_on),
		name='varcopemerge')
	tstatmerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=tstats_off_on),
		name='tstatmerge')		
	
	level2model = Node(interface=fsl.L2Model(
		num_copes=len(copes_off_on)),
		name='l2model')

	multregmodel = Node(interface=fsl.MultipleRegressDesign(
		contrasts=[],
		regressors={}),
		name='multregmodel')

	multregmodel.inputs.contrasts = [["cope"+str(contrast)+'_mean', 'T',['reg1'],[1]],['k_ag', 'T',['reg2'],[1]]] 
	multregmodel.inputs.regressors = dict(reg1=list(np.ones(len(copes_off_on))),reg2=list(k_ag_demeaned))

	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_rpe.base_dir,'stats',))

	# Use level2model model if not using covariate
	Parkflow_group_rpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(level2model,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	# Use multregmodel model if using covariate
	# Parkflow_group_rpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
	# 		(varcopemerge,flame12,[('merged_file','var_cope_file')]),
	# 		(multregmodel,flame12,[('design_mat','design_file'),
	# 						('design_con','t_con_file'),
	# 						('design_grp','cov_split_file')])
	# 		,
	# 		])

	Parkflow_group_rpe.write_graph(graph2use='colored')
	Parkflow_group_rpe.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_rpe.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_rpe_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	# Cluster correction for mean of this contrast
	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_mean_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()	

	# Cluster correction for k_ag covariate
	# clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	# clusterout = os.popen(clustercmd).read()
	# f1=open('thres_cluster_k_ag_2.3_fwe_table.txt','w+')
	# f1.write(clusterout)
	# f1.close()

	# # Cluster correction for k_al covariate
	# clustercmd = 'cluster -i zstat2 -c cope2 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat2 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	# clusterout = os.popen(clustercmd).read()
	# f1=open('thres_cluster_k_al_2.3_fwe_table.txt','w+')
	# f1.write(clusterout)
	# f1.close()


## RPE_update (alpha * RPE) workflow ##

# ON - OFF
contrasts = [1,2,3]

for contrast in contrasts: 

	copes_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_arpe_dirs_all_subs]
	varcopes_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_arpe_dirs_all_subs]
	tstats_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_arpe_dirs_all_subs]

	# Create workflow
	Parkflow_group_arpe = Workflow(name='workflow')
	Parkflow_group_arpe.base_dir = os.path.join(group_dir,"on_off","feedback_aRPE","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_arpe.base_dir):
		os.makedirs(Parkflow_group_arpe.base_dir)

	# Create nodes

	copemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=copes_on_off),
		name='copemerge')
	varcopemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=varcopes_on_off),
		name='varcopemerge')
	tstatmerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=tstats_on_off),
		name='tstatmerge')		
	level2model = Node(interface=fsl.L2Model(
		num_copes=len(copes_on_off)),
		name='l2model')
	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_arpe.base_dir,'stats',))

	
	Parkflow_group_arpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(level2model,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_arpe.write_graph(graph2use='colored')
	Parkflow_group_arpe.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_arpe.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_arpe_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()	

# OFF - ON
contrasts = [1,2,3]

for contrast in contrasts: 

	copes_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope2.nii.gz') for x in feat_arpe_dirs_all_subs]
	varcopes_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope2.nii.gz') for x in feat_arpe_dirs_all_subs]
	tstats_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat2.nii.gz') for x in feat_arpe_dirs_all_subs]

	# Create workflow
	Parkflow_group_arpe = Workflow(name='workflow')
	Parkflow_group_arpe.base_dir = os.path.join(group_dir,"off_on","feedback_aRPE","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_arpe.base_dir):
		os.makedirs(Parkflow_group_arpe.base_dir)

	# Create nodes

	copemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=copes_off_on),
		name='copemerge')
	varcopemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=varcopes_off_on),
		name='varcopemerge')
	tstatmerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=tstats_off_on),
		name='tstatmerge')		
	level2model = Node(interface=fsl.L2Model(
		num_copes=len(copes_off_on)),
		name='l2model')
	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_arpe.base_dir,'stats',))

	
	Parkflow_group_arpe.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(level2model,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_arpe.write_graph(graph2use='colored')
	Parkflow_group_arpe.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_arpe.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_arpe_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()


## Qval_update workflow ##

# ON - OFF
contrasts = [1,2,3]

for contrast in contrasts: 

	copes_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope1.nii.gz') for x in feat_qval_dirs_all_subs]
	varcopes_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope1.nii.gz') for x in feat_qval_dirs_all_subs]
	tstats_on_off = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat1.nii.gz') for x in feat_qval_dirs_all_subs]

	# Create workflow
	Parkflow_group_qval = Workflow(name='workflow')
	Parkflow_group_qval.base_dir = os.path.join(group_dir,"on_off","feedback_qval","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_qval.base_dir):
		os.makedirs(Parkflow_group_qval.base_dir)

	# Create nodes

	copemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=copes_on_off),
		name='copemerge')
	varcopemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=varcopes_on_off),
		name='varcopemerge')
	tstatmerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=tstats_on_off),
		name='tstatmerge')		
	level2model = Node(interface=fsl.L2Model(
		num_copes=len(copes_on_off)),
		name='l2model')
	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_qval.base_dir,'stats',))

	
	Parkflow_group_qval.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(level2model,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_qval.write_graph(graph2use='colored')
	Parkflow_group_qval.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_qval.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_qval_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()	

# OFF - ON
contrasts = [1,2,3]

for contrast in contrasts: 

	copes_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','cope2.nii.gz') for x in feat_qval_dirs_all_subs]
	varcopes_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','varcope2.nii.gz') for x in feat_qval_dirs_all_subs]
	tstats_off_on = [os.path.join(x,'%s%i'%('cope',contrast),'workflow','FE','stats','tstat2.nii.gz') for x in feat_qval_dirs_all_subs]

	# Create workflow
	Parkflow_group_qval = Workflow(name='workflow')
	Parkflow_group_qval.base_dir = os.path.join(group_dir,"off_on","feedback_qval","cope"+str(contrast))

	if not os.path.exists(Parkflow_group_qval.base_dir):
		os.makedirs(Parkflow_group_qval.base_dir)

	# Create nodes

	copemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=copes_off_on),
		name='copemerge')
	varcopemerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=varcopes_off_on),
		name='varcopemerge')
	tstatmerge = Node(interface=fsl.Merge(
		dimension='t',
		in_files=tstats_off_on),
		name='tstatmerge')		
	level2model = Node(interface=fsl.L2Model(
		num_copes=len(copes_off_on)),
		name='l2model')
	flame12=Node(interface=fsl.FLAMEO(
		run_mode='flame12',
		mask_file=groupmaskfile,
		infer_outliers=True),
		name='flame12',
		stats_dir=os.path.join(Parkflow_group_qval.base_dir,'stats',))

	
	Parkflow_group_qval.connect([(copemerge,flame12,[('merged_file','cope_file')]),
					(varcopemerge,flame12,[('merged_file','var_cope_file')]),
					(level2model,flame12,[('design_mat','design_file'),
									('design_con','t_con_file'),
									('design_grp','cov_split_file')])
					,
					])

	Parkflow_group_qval.write_graph(graph2use='colored')
	Parkflow_group_qval.run()

	# Cluster correction
	os.chdir(os.path.join(Parkflow_group_qval.base_dir,'workflow','flame12','stats'))

	smoothcmd = 'smoothest -r res4d -d %i -m mask'%(len(feat_qval_dirs_all_subs)-1)
	smooth = os.popen(smoothcmd).read().split("\n")
	smoothn = [x.split(' ')[1] for x in smooth[:-1]]

	clustercmd = 'cluster -i zstat1 -c cope1 -t 2.3 -p 0.01 -d %s --volume=%s --othresh=thresh_cluster_2.3_fwe_zstat1 --connectivity=26 --mm'%(smoothn[0],smoothn[1])
	clusterout = os.popen(clustercmd).read()
	f1=open('thres_cluster_2.3_fwe_table.txt','w+')
	f1.write(clusterout)
	f1.close()
