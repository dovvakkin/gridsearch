rm -rf answer
mkdir -p answer/task1
mkdir -p answer/task2

python3 make_pred_fixed_dual_tfidf.py --threshold 500 --low-bound 0.65 --high-bound 0.999 --first-subst ~/Desktop/english_1-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-and-T-2ltr2f_topk500_fixspacesTrue.bz2 --second-subst ~/Desktop/english_2-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-and-T-2ltr2f_topk500_fixspacesTrue.bz2 --target-words target_words/eng.txt --output answer/task2/english.txt &&\
python3 make_pred_fixed_dual_tfidf.py --threshold 500 --low-bound 0.65 --high-bound 0.999 --first-subst ~/Desktop/german_1-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-und-T-2ltr2f_topk500_fixspacesTrue.bz2 --second-subst ~/Desktop/german_2-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-und-T-2ltr2f_topk500_fixspacesTrue.bz2 --target-words target_words/ger.txt --output answer/task2/german.txt && \
python3 make_pred_fixed_dual_tfidf.py --threshold 500 --low-bound 0.65 --high-bound 0.999 --first-subst ~/Desktop/latin_1-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-et-T-2ltr2f_topk500_fixspacesTrue.bz2 --second-subst ~/Desktop/latin_2-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-et-T-2ltr2f_topk500_fixspacesTrue.bz2 --target-words target_words/lat.txt  --output answer/task2/latin.txt &&\
python3 make_pred_fixed_dual_tfidf.py --threshold 500 --low-bound 0.65 --high-bound 0.999 --first-subst ~/Desktop/swedish_1-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-och-T-2ltr2f_topk500_fixspacesTrue.bz2 --second-subst ~/Desktop/swedish_2-limitNone-maxexperword1000/modelcheckpoints_alldata-large-xlmfinetuned_checkpoint26.pt/\<mask\>-och-T-2ltr2f_topk500_fixspacesTrue.bz2 --target-words target_words/swe.txt  --output answer/task2/swedish.txt

python3 prettify_submission.py
zip -r answer.zip answer
