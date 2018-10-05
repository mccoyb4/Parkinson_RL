
subs=('sub-108')

for s in "${subs[@]}";
do
    python Park_feedback_L1a_Smooth_old3.py $s 'hc' 'off' 'ae'
done
