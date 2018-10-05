#!/bin/bash
#PBS -lwalltime=30:00:00 -lnodes=1:mem64gb

module load eb Python/2.7.14-foss-2017b
module load eb fsl/5.08
PROJ_PATH=/nfs/bromccoy

T_START=${PBS_ARRAYID[0]}
T_END=${PBS_ARRAYID[${#PBS_ARRAYID[@]}-1]}

for ((i=$T_START; i<=$T_END; i++)) ; do
(
	python /home/bromccoy/bash_scripts/Park_pipeline/Park/Parkinson_SingleTrial/workflows/Park_feedback_L1a_noSmooth.py "sub-""$i" "hc" "off" "2" "lisa"
) &
done
wait