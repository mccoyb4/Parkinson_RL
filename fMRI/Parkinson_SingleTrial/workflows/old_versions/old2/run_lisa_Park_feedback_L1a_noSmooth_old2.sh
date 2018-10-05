#PBS -lwalltime=48:00:00 -lnodes=1:mem64gb

module load eb Python/2.7.14-foss-2017b
module load eb fsl/5.08
PROJ_PATH=/nfs/bromccoy

SUBJ_ID=$PBS_ARRAYID
SUBJ_NR="sub-$SUBJ_ID"

python /home/bromccoy/bash_scripts/Park_pipeline/Park/Parkinson_SingleTrial/workflows/old_versions/old2/Park_feedback_L1a_noSmooth_old2.py "$SUBJ_NR" 'pd' 'off' 'lisa'

