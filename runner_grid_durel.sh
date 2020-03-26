python3 count_durel_score.py \
--low-bound 0.5 0.6 0.7 0.8 \
--high-bound 0.93 0.95 0.97 \
 --threshold 100 300 500 \
 --model tfidf count \
 --first-subst \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 --second-subst \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
--output ~/Desktop/1.csv &

python3 count_durel_score.py \
--low-bound 0.5 0.6 0.7 0.8 \
--high-bound 0.93 0.95 0.97 \
 --threshold 100 300 500 \
 --model tfidf count \
 --first-subst \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
--second-subst \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint36.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
--output ~/Desktop/2.csv &

python3 count_durel_score.py \
--low-bound 0.5 0.6 0.7 0.8 \
--high-bound 0.93 0.95 0.97 \
 --threshold 100 300 500 \
 --model tfidf count \
 --first-subst \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
 --second-subst \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
 --output ~/Desktop/3.csv &

 python3 count_durel_score.py \
--low-bound 0.5 0.6 0.7 0.8 \
--high-bound 0.93 0.95 0.97 \
 --threshold 100 300 500 \
 --model tfidf count \
 --first-subst \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
 --second-subst \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-small-xlmfinetuned_checkpoint12.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-oder-T-2ltr2f_topk150_fixspacesTrue.npz \
 /home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-T-2ltr2f_topk150_fixspacesTrue.npz \
--output ~/Desktop/4.csv &

python3 count_durel_score.py \
--low-bound 0.5 0.6 0.7 0.8 \
--high-bound 0.93 0.95 0.97 \
 --threshold 100 300 500 \
 --model tfidf count \
 --first-subst \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_1-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-oder-T-2ltr2f_topk500_fixspacesTrue.bz2 \
--second-subst \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/\<mask\>\<mask\>-und-T-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/T-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/T-oder-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/ezhik/dta_2-limitNone-maxexperwordNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint7.pt/T-und-\<mask\>\<mask\>-2ltr2f_topk150_fixspacesTrue.npz \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-oder-T-2ltr2f_topk500_fixspacesTrue.bz2 \
--output ~/Desktop/5.csv &

python3 count_durel_score.py \
--low-bound 0.5 0.6 0.7 0.8 \
--high-bound 0.93 0.95 0.97 \
 --threshold 100 300 500 \
 --model tfidf count \
 --first-subst \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-T-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-und-T-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/T-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/T-oder-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/T-und-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
--second-subst \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-T-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/\<mask\>-und-T-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/T-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/T-oder-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint12.pt/T-und-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
/home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
--output ~/Desktop/6.csv &

python3 count_durel_score.py \
--low-bound 0.5 0.6 0.7 0.8 \
--high-bound 0.93 0.95 0.97 \
 --threshold 100 300 500 \
 --model tfidf count \
 --first-subst \
 /home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-T-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-und-T-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-oder-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_1-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-und-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
 --second-subst \
 /home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-T-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-und-T-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-oder-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
 /home/y.kozhevnikov/Desktop/dta_2-limitNone/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/T-und-\<mask\>-2ltr2f_topk500_fixspacesTrue.bz2 \
--output ~/Desktop/7.csv &