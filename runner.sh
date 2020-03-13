#!/usr/bin/env bash

conda install pandas
conda install numpy
conda install scipy

FIRST="first"
SECOND="second"

for file in 'T-oder-<mask>-2ltr2f_topk500_fixspacesTrue.bz2' \
            '<mask>-2ltr2f_topk500_fixspacesTrue.bz2' \
            '<mask>-T-2ltr2f_topk500_fixspacesTrue.bz2' \
            '<mask>-und-T-2ltr2f_topk500_fixspacesTrue.bz2' \
            'T-<mask>-2ltr2f_topk500_fixspacesTrue.bz2' \
            'T-und-<mask>-2ltr2f_topk500_fixspacesTrue.bz2' \
            '<mask>-oder-T-2ltr2f_topk500_fixspacesTrue.bz2'
do
  python3 main.py  --first-directory $FIRST --second-directory $SECOND \
--low-bound 0.7 0.8 0.85 0.9 0.95  --high-bound 0.99 0.995 0.999 0.9995 0.9999 \
 --threshold  10 20 50 100 150 200 300 400 500 \
 -f $file \
 -o result.txt &
done