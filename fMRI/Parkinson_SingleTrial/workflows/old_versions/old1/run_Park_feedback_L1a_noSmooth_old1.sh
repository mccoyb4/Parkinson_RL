
subs=('sub-201')

for s in "${subs[@]}";
do
    python Park_feedback_L1a_noSmooth_old1.py $s 'pd' 'off' 'single-trial' 'ae'
done