#PBS -lwalltime=8:00:00 -lnodes=1:mem64gb

module load eb Python/2.7.14-foss-2017b
module load eb fsl/5.08
PROJ_PATH=/nfs/bromccoy

python /home/bromccoy/bash_scripts/Park_pipeline/Park/Parkinson_SingleTrial/workflows/old_versions/old2/Park_feedback_L5_medDiff_group_old2.py '0' 'lisa'
