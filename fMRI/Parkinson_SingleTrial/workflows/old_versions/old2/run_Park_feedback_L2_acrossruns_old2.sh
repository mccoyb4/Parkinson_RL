
subs=('sub-201' 'sub-202' 'sub-204' 'sub-205' 'sub-206' 'sub-207' 'sub-208' 'sub-210' 'sub-211' 'sub-212' 'sub-214' 'sub-216' 'sub-217' 'sub-219' 'sub-220' 'sub-221' 'sub-222' 'sub-223' 'sub-224')

for s in "${subs[@]}";
do
    python Park_feedback_L2_acrossruns_old2.py $s 'pd' 'on' '1' 'ae'
done