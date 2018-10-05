
subs=('sub-219')

for s in "${subs[@]}";
do
    python Park_feedback_L1a_noSmooth_old2.py $s 'pd' 'off' 'ae'
done
