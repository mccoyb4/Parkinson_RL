
subs=('sub-201' 'sub-202' 'sub-203' 'sub-204' 'sub-205' 'sub-206' 'sub-207' 'sub-208' 'sub-209' 'sub-210' 'sub-211' 'sub-212' 'sub-213' 'sub-214' 'sub-215' 'sub-216' 'sub-217' 'sub-220' 'sub-221' 'sub-222' 'sub-223' 'sub-224')

# put 'sub-219' back in

for s in "${subs[@]}";
do
    python Park_feedback_L3_medDiff_old2.py $s '0' 'ae'
done