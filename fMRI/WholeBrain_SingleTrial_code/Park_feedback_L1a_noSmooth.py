from __future__ import division, print_function, absolute_import

# Based on Github PoldrackLab script: https://github.com/poldracklab/CNP_task_analysis/blob/master/CNP_analysis.py
# Also based on Github Knapen lab PRF : https://github.com/tknapen/PRF_MB_7T/tree/master/nPRF

# Outputs to single_trial_analysis/no_smooth_4 

from nipype.interfaces.fsl import model, ExtractROI, SliceTimer, Smooth, FEATModel, FEAT, Level1Design, FILMGLS, maths
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import afni
from nipype.interfaces.base import Bunch
from bids.grabbids import BIDSLayout
from preproc_functions import savgol_filter, average_signal, percent_signal_change, natural_sort
import json
import nibabel as nib
import pandas as pd
import numpy as np
import sys
import os
#from IPython import embed as shell

# The following are commandline arguments specified by bash script.
sub_id = str(sys.argv[1])
hc_or_pd = str(sys.argv[2]) # group : 'hc', 'pd'
pd_on_off = str(sys.argv[3]) # 'on', 'off'
run_nr = str(sys.argv[4]) # # '1' or '2'
ae_or_lisa = str(sys.argv[5]) # 'ae' or 'lisa' server

subjects = [sub_id]

if ae_or_lisa == 'ae':
	with open('analysis_settings.json') as f:
	    json_s = f.read()
	    analysis_info = json.loads(json_s)
elif ae_or_lisa == 'lisa':
	with open('/home/bromccoy/bash_scripts/Park_pipeline/Park/Parkinson_SingleTrial/workflows/analysis_settings.json') as f:
	    json_s = f.read()
	    analysis_info = json.loads(json_s)


if hc_or_pd == 'hc':

	#subjects = ['sub-108','sub-111','sub-112','sub-113','sub-114','sub-115','sub-116','sub-117','sub-118','sub-119','sub-120','sub-121','sub-123','sub-124','sub-126','sub-127','sub-128','sub-129','sub-130','sub-131','sub-132','sub-133']

	# Name directories for input and output
	if ae_or_lisa == 'ae':

		datadir='/home/shared/2016/Parkinson/data/hc'
		fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/hc_syn-sdc'
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/hc"

	elif ae_or_lisa == 'lisa':

		datadir='/nfs/bromccoy/Parkinson/data/hc'
		fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/hc_syn-sdc'
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/hc"

elif hc_or_pd == 'pd':

	# subjects = ['sub-201','sub-202','sub-203','sub-204','sub-205','sub-206','sub-207','sub-208','sub-209','sub-210','sub-211','sub-212','sub-213','sub-214','sub-215','sub-216','sub-217','sub-219','sub-220','sub-221','sub-222','sub-223','sub-224']

	# Name directories for input and output
	if ae_or_lisa=='ae':

		datadir='/home/shared/2016/Parkinson/data/pd'
		workflow_dir = "/home/shared/2016/Parkinson/single_trial_analysis/no_smooth_4/pd"

		if pd_on_off == 'on':
			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
		elif pd_on_off == 'off':
			fmriprep_dir = '/home/shared/2016/Parkinson/fmriprep_preproc/pd_off_syn-sdc'

	elif ae_or_lisa=='lisa':
		
		datadir='/nfs/bromccoy/Parkinson//data/pd'
		workflow_dir = "/nfs/bromccoy/Parkinson/single_trial_analysis/no_smooth_4/pd"

		if pd_on_off == 'on':
			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_on_syn-sdc'
		elif pd_on_off == 'off':
			fmriprep_dir = '/nfs/bromccoy/Parkinson/fmriprep_preproc/pd_off_syn-sdc'

## Loop across all subjects specified by bash script ##
for sb in range(len(subjects)):


	if hc_or_pd == 'hc':

	    sub_workflow_dir = os.path.join(workflow_dir,subjects[sb],"train","run-"+run_nr,"feedback")
	    bold_filename = os.path.join(datadir,subjects[sb],"func",subjects[sb]+"_task-train_run-%s_bold.nii.gz"%(run_nr))
	    events_filename = os.path.join(datadir,subjects[sb],"func",subjects[sb]+"_task-train_run-%s_events.tsv"%(run_nr))

	elif hc_or_pd == 'pd':

	    sub_workflow_dir = os.path.join(workflow_dir,subjects[sb],pd_on_off,"train","run-"+run_nr,"feedback")
	    bold_filename = os.path.join(datadir,subjects[sb],pd_on_off,"func",subjects[sb]+"_task-train_run-%s_bold.nii.gz"%(run_nr))
	    events_filename = os.path.join(datadir,subjects[sb],pd_on_off,"func",subjects[sb]+"_task-train_run-%s_events.tsv"%(run_nr))	    	

	if not os.path.exists(sub_workflow_dir):
	    os.makedirs(sub_workflow_dir)

	from nipype.caching import Memory
	mem = Memory(base_dir=sub_workflow_dir)

	events = pd.read_csv(events_filename, sep="\t")

	confounds = pd.read_csv(os.path.join(fmriprep_dir, "fmriprep", 
	                                    subjects[sb], "func", 
	                                    subjects[sb]+"_task-train_run-%s_bold_confounds.tsv"%(run_nr)),
	                                    sep="\t", na_values="n/a")

	# Note: onsets from events files are already with first 2 TRs removed (= 4.3 secs)

	# Get design matrix info
	trials, feedback_onset_per_trial, feedback_duration_per_trial = [[] for i in range(3)]

	for i in range(len(events)):
	    trials.append("trial_"+str(i+1)+"_pair_"+str(events.Pair[i])+"_choice_"+str(events.Choice[i])+"_feedback_"+str(events.Outcome[i]))
	    feedback_onset_per_trial.append([events.Feedback_Onset[i]])
	    feedback_duration_per_trial.append([events.Feedback_duration[i]])

	info = [Bunch(conditions=trials,
					onsets=feedback_onset_per_trial,
					durations=feedback_duration_per_trial,
					regressors=[list(confounds.FramewiseDisplacement.fillna(0)[analysis_info['remove_TRs']:]),
								list(confounds.X[analysis_info['remove_TRs']:]),
								list(confounds.Y[analysis_info['remove_TRs']:]),
								list(confounds.Z[analysis_info['remove_TRs']:]),
								list(confounds.RotX[analysis_info['remove_TRs']:]),
								list(confounds.RotY[analysis_info['remove_TRs']:]),
								list(confounds.RotZ[analysis_info['remove_TRs']:]),
								list(confounds.aCompCor00[analysis_info['remove_TRs']:]),
								list(confounds.aCompCor01[analysis_info['remove_TRs']:]),
								list(confounds.aCompCor02[analysis_info['remove_TRs']:]),
								list(confounds.aCompCor03[analysis_info['remove_TRs']:]),
								list(confounds.aCompCor04[analysis_info['remove_TRs']:]),
								list(confounds.aCompCor05[analysis_info['remove_TRs']:])],
					regressor_names=['FramewiseDisplacement',
									'X',
									'Y',
									'Z',
									'RotX',
									'RotY',
									'RotZ',
									'aCompCor00',
									'aCompCor01',
									'aCompCor02',
									'aCompCor03',
									'aCompCor04',
									'aCompCor05'])
	        ]

	# Preparing single trial contrasts (t-stats)
	contrasts = []
	for i in range(len(events)):
	    contrasts.append(['trial_'+str(i+1),'T', ["trial_"+str(i+1)+"_pair_"+str(events.Pair[i])+"_choice_"+str(events.Choice[i])+"_feedback_"+str(events.Outcome[i])],[1]])


	####### NIPYPE PIPELINE #######

	skip = mem.cache(ExtractROI)
	skip(in_file=os.path.join(fmriprep_dir, "fmriprep", 
	                        subjects[sb], "func", 
	                        subjects[sb]+"_task-train_run-%s_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(run_nr)),
	    t_min=analysis_info['remove_TRs'], t_size=-1, 
	    roi_file=os.path.join(sub_workflow_dir, subjects[sb] + "_task-train_run-" + run_nr +
	                    "_bold_space-MNI152NLin2009cAsym_preproc_roi.nii.gz"))

	identity = Node(IdentityInterface(
            fields=['sg_input']), name='identity')

	identity.inputs.sg_input = os.path.join(sub_workflow_dir, subjects[sb] + "_task-train_run-" + run_nr + "_bold_space-MNI152NLin2009cAsym_preproc_roi.nii.gz")

	sgfilter = Node(Function(input_names=['in_file', 'window_length', 'polyorder'],
                  output_names=['sg_file'],
                  function=savgol_filter),
                  name='sgfilter', iterfield=['in_file'])

	sgfilter.inputs.window_length = analysis_info['sg_filter_window_length']   	
	sgfilter.inputs.polyorder = analysis_info['sg_filter_order']


	l1 = Node(SpecifyModel(
		subject_info = info,
	    input_units='secs',
	    time_repetition=analysis_info['TR'],
	    high_pass_filter_cutoff=analysis_info['highpass_filter'] # set to -1 for no filtering as already have done SG filtering
	), name='l1')


	l1model = Node(Level1Design(
	    interscan_interval=analysis_info['TR'],
	    bases={'dgamma': {'derivs': True}},
	    model_serial_correlations=True,
	    contrasts=contrasts
	), name='l1design')

	l1featmodel = Node(FEATModel(), name='l1model')

	filmgls= Node(FILMGLS(
	    autocorr_noestimate = True,
	    results_dir=os.path.join(sub_workflow_dir,'stats')      
	), name='filmgls')

	Parkflow = Workflow(name='workflow')
	Parkflow.base_dir = sub_workflow_dir

	Parkflow.connect([(identity, sgfilter, [('sg_input', 'in_file')]),
			 (sgfilter, l1, [('sg_file', 'functional_runs')]),
			 (l1, l1model, [('session_info', 'session_info')]),
			 (l1model, l1featmodel, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
			 (sgfilter, filmgls, [('sg_file', 'in_file')]),
			 (l1featmodel, filmgls, [('design_file', 'design_file'),('con_file','tcon_file'),('fcon_file','fcon_file')])
			 ])

	Parkflow.write_graph(graph2use='colored')
	Parkflow.run('MultiProc', plugin_args={'n_procs': 16})

	# Remove ROI nifti from skip function, this file takes up a lot of space
	os.remove(os.path.join(sub_workflow_dir, subjects[sb] + "_task-train_run-" + run_nr + "_bold_space-MNI152NLin2009cAsym_preproc_roi.nii.gz"))

