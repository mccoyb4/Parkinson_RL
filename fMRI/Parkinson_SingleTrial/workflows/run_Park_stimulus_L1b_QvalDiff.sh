
subs=('sub-108' 'sub-111' 'sub-113' 'sub-115' 'sub-116' 'sub-118' 'sub-119' 'sub-123' 'sub-124' 'sub-126' 'sub-129' 'sub-130' 'sub-131' 'sub-132' 'sub-133') 
# 'sub-112' 'sub-114' 'sub-117' 'sub-120' 'sub-121' 'sub-127' 'sub-128'
#subs=('sub-201' 'sub-202' 'sub-203' 'sub-204' 'sub-205' 'sub-206' 'sub-207' 'sub-208' 'sub-209' 'sub-210' 'sub-211' 'sub-212' 'sub-213' 'sub-214' 'sub-215' 'sub-216' 'sub-217' 'sub-219' 'sub-220' 'sub-221' 'sub-222' 'sub-223' 'sub-224')

for s in "${subs[@]}";
do
    python Park_stimulus_L1b_QvalDiff.py $s 'hc' 'off' '1' 'ae' &
done