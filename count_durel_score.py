import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from substs_stat import load_substs
from scipy.spatial.distance import cosine


def make_vocab(word_dict, low_threshold, high_threshold):
    words = list([i[0] for i in sorted(word_dict.items(), key=lambda t: t[1])])
    words = words[
            int(low_threshold * len(words)): int(high_threshold * len(words))]
    return set(words)


def make_word_vectors(word_dict, global_set, df_dict=None):
    word_vectors = dict()
    word_list = list()
    for word in word_dict:
        word_list.append(word)
        word_vector = list()
        for word_axis in global_set:
            if word_axis in word_dict[word]:
                if df_dict:
                    word_vector.append(word_dict[word][word_axis] / df_dict[word_axis])
                else:
                    word_vector.append(word_dict[word][word_axis])
            else:
                word_vector.append(0)
        word_vectors[word] = word_vector
    return word_list, word_vectors


def make_pred(first_subst, second_subst, threshold, low_bound,
             high_bound, model, output):
    TOPK_THRESHOLD = threshold

    # MAKE VOCAB
    dta_1_df = dict()

    substs = load_substs(first_subst)

    bz2 = substs['substs_probs']
    input_list_word = substs['word'].tolist()
    global_word_dict = dict()

    for pods_list, word in zip(bz2, input_list_word):
        one_word_df = set()
        for pod in pods_list:
            num, pod_word = pod
            if pod_word not in one_word_df:
                one_word_df.update([pod_word])
            if pod_word not in global_word_dict:
                global_word_dict[pod_word] = 1
            else:
                global_word_dict[pod_word] += 1
        for pod_word in one_word_df:
            if pod_word not in dta_1_df:
                dta_1_df[pod_word] = 1
            else:
                dta_1_df[pod_word] += 1

    dta_2_df = dict()
    substs = load_substs(second_subst)

    bz2 = substs['substs_probs']
    input_list_word = substs['word'].tolist()

    for pods_list, word in zip(bz2, input_list_word):
        one_word_df = set()
        for pod in pods_list:
            num, pod_word = pod
            if pod_word not in one_word_df:
                one_word_df.update([pod_word])
            if pod_word not in global_word_dict:
                global_word_dict[pod_word] = 1
            else:
                global_word_dict[pod_word] += 1
        for pod_word in one_word_df:
            if pod_word not in dta_2_df:
                dta_2_df[pod_word] = 1
            else:
                dta_2_df[pod_word] += 1
    vocab = make_vocab(global_word_dict, low_bound, high_bound)
    # END MAKE VOCAB

    dta_1_counter = dict()

    substs = load_substs(first_subst)

    bz2 = substs['substs_probs']
    input_list_word = substs['word'].tolist()

    for pods_list, word in zip(bz2, input_list_word):
        if word not in dta_1_counter:
            dta_1_counter[word] = dict()

        count = 0
        for pod in pods_list:
            if count >= TOPK_THRESHOLD:
                break
            num, pod_word = pod
            if pod_word not in vocab:
                continue
            count += 1
            if pod_word not in dta_1_counter[word]:
                dta_1_counter[word][pod_word] = 1
            else:
                dta_1_counter[word][pod_word] += 1

    dta_2_counter = dict()

    substs = load_substs(second_subst)

    bz2 = substs['substs_probs']
    input_list_word = substs['word'].tolist()

    for pods_list, word in zip(bz2, input_list_word):
        if word not in dta_2_counter:
            dta_2_counter[word] = dict()

        count = 0
        for pod in pods_list:
            if count >= TOPK_THRESHOLD:
                break
            num, pod_word = pod
            if pod_word not in vocab:
                continue
            count += 1
            if pod_word not in dta_2_counter[word]:
                dta_2_counter[word][pod_word] = 1
            else:
                dta_2_counter[word][pod_word] += 1

    if model == 'tfidf':
        dta1_words, dta1_vectors = make_word_vectors(dta_1_counter, vocab,
                                                     dta_1_df)
        dta2_words, dta2_vectors = make_word_vectors(dta_2_counter, vocab,
                                                     dta_2_df)
    elif model == 'count':
        dta1_words, dta1_vectors = make_word_vectors(dta_1_counter, vocab)
        dta2_words, dta2_vectors = make_word_vectors(dta_2_counter, vocab)
    else:
        raise

    targets = ['Abend', 'Anstalt', 'Anstellung', 'Bilanz', 'billig',
               'Donnerwetter', 'englisch', 'Feder', 'Feine', 'geharnischt',
               'locker', 'Motiv', 'Museum', 'packen', 'Presse', 'Reichstag',
               'technisch', 'Vorwort', 'Zufall']

    targets_scores = np.array(
        [-3.79, -2.0725, -2.6789473684, -3.2, -2.4316666667,
         -1.8375, -3.3375, -2.1403508772, -1.93, -3, -2.84,
         -2.66, -3.7325, -2.7350877193, -1.8825, -3.4525,
         -2.89, -1.5825, -3.1125])
    # with open(target_words) as f:
    #     targets = [line[:-1] for line in f]

    dta1_vectors = list([dta1_vectors[i] for i in targets])
    dta2_vectors = list([dta2_vectors[i] for i in targets])

    np_dta1 = np.asarray(dta1_vectors)
    np_dta2 = np.asarray(dta2_vectors)

    result_cosines = list()
    for vec1, vec2 in zip(np_dta1, np_dta2):
        result_cosines.append(cosine(vec1, vec2))

    result_cosines = np.asarray(result_cosines)

    rho, p = spearmanr(result_cosines, targets_scores, nan_policy='omit')

    slash_ind = first_subst.find('/')
    if slash_ind == -1:
        name = first_subst
    else:
        name = first_subst[slash_ind + 1:]

    with open(output, 'a') as f:
        f.write('{},{},{},{},{},{},{}\n'.format(name,
                                                model,
                                                low_bound,
                                                high_bound,
                                                threshold,
                                                rho,
                                                p))


parser = argparse.ArgumentParser()

parser.add_argument('--low-bound', nargs='+',required=True)
parser.add_argument('--high-bound', nargs='+',required=True)
parser.add_argument('--threshold', nargs='+',required=True,
                    help='top N substitution')
parser.add_argument('--model', nargs='+',required=True,
                    help='model type')
parser.add_argument('--first-subst', nargs='+',required=True,
                    help='path to first substitution archive')
parser.add_argument('--second-subst', nargs='+',required=True,
                    help='path to first substitution archive')
# parser.add_argument('--target-words', required=True,
#                     help='path to target words file')
parser.add_argument('--output', required=True)

args = parser.parse_args()

low_bounds = args.low_bound
high_bounds = args.high_bound
thresholds = args.threshold
first_substs = args.first_subst
second_substs = args.second_subst
substs = zip(first_substs, second_substs)
models = args.model
# target_words = args.target_words
output = args.output

for subst in substs:
    for low_bound in low_bounds:
        for high_bound in high_bounds:
            for threshold in thresholds:
                for model in models:
                    make_pred(subst[0],
                              subst[1],
                              int(threshold),
                              float(low_bound),
                              float(high_bound),
                              model,
                              output)
