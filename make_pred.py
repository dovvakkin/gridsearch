import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine


def make_vocab(word_dict, low_threshold, high_threshold):
    words = list([i[0] for i in sorted(word_dict.items(), key=lambda t: t[1])])
    words = words[
            int(low_threshold * len(words)): int(high_threshold * len(words))]
    return set(words)


def string_to_tuple(shit):
    shit = shit.replace('(', '').replace(' ', '')
    num, word = shit[:shit.find(',')], shit[shit.find(',') + 1:]

    return float(num), word[1:-1]


def make_word_vectors(word_dict, global_set):
    word_vectors = dict()
    word_list = list()
    for word in word_dict:
        word_list.append(word)
        word_vector = list()
        for word_axis in global_set:
            if word_axis in word_dict[word]:
                word_vector.append(word_dict[word][word_axis])
            else:
                word_vector.append(0)
        word_vectors[word] = word_vector
    return word_list, word_vectors


def make_pred(first_subst, second_subst, threshold, low_bound,
             high_bound, target_words):
    TOPK_THRESHOLD = threshold

    input_list_word = list()
    dta_1_counter = dict()

    bz2 = pd.read_csv(first_subst)
    inp = pd.read_csv(first_subst + '.input')

    for word in inp['word']:
        input_list_word.append(word)

    # MAKE VOCAB
    # TODO
    global_word_dict = dict()
    for pods, word in zip(bz2['0'], input_list_word):
        pods_list = pods.strip('][').split('), ')
        for pod in pods_list:
            num, pod_word = string_to_tuple(pod)
            if pod_word not in global_word_dict:
                global_word_dict[pod_word] = 1
            else:
                global_word_dict[pod_word] += 1
    vocab = make_vocab(global_word_dict, low_bound, high_bound)
    # END MAKE VOCAB

    for pods, word in zip(bz2['0'], input_list_word):
        if word not in dta_1_counter:
            dta_1_counter[word] = dict()

        pods_list = pods.strip('][').split('), ')
        count = 0
        for pod in pods_list:
            if count >= TOPK_THRESHOLD:
                break
            num, pod_word = string_to_tuple(pod)
            if pod_word not in vocab:
                continue
            count += 1
            if pod_word not in dta_1_counter[word]:
                dta_1_counter[word][pod_word] = 1
            else:
                dta_1_counter[word][pod_word] += 1

    input_list_word = list()
    dta_2_counter = dict()

    bz2 = pd.read_csv(second_subst)
    inp = pd.read_csv(second_subst + '.input')

    for word in inp['word']:
        input_list_word.append(word)

    for pods, word in zip(bz2['0'], input_list_word):
        if word not in dta_2_counter:
            dta_2_counter[word] = dict()

        pods_list = pods.strip('][').split('), ')
        count = 0
        for pod in pods_list:
            if count >= TOPK_THRESHOLD:
                break
            num, pod_word = string_to_tuple(pod)
            if pod_word not in vocab:
                continue
            count += 1
            if pod_word not in dta_2_counter[word]:
                dta_2_counter[word][pod_word] = 1
            else:
                dta_2_counter[word][pod_word] += 1

    dta1_words, dta1_vectors = make_word_vectors(dta_1_counter, vocab)
    dta2_words, dta2_vectors = make_word_vectors(dta_2_counter, vocab)

    # targets = ['Abend', 'Anstalt', 'Anstellung', 'Bilanz', 'billig',
    #            'Donnerwetter', 'englisch', 'Feder', 'Feine', 'geharnischt',
    #            'locker', 'Motiv', 'Museum', 'packen', 'Presse', 'Reichstag',
    #            'technisch', 'Vorwort', 'Zufall']
    #
    # targets_scores = np.array(
    #     [-3.79, -2.0725, -2.6789473684, -3.2, -2.4316666667,
    #      -1.8375, -3.3375, -2.1403508772, -1.93, -3, -2.84,
    #      -2.66, -3.7325, -2.7350877193, -1.8825, -3.4525,
    #      -2.89, -1.5825, -3.1125])
    with open(target_words) as f:
        targets = [line for line in f]

    dta1_vectors = list([dta1_vectors[i] for i in targets])
    dta2_vectors = list([dta2_vectors[i] for i in targets])

    np_dta1 = np.asarray(dta1_vectors)
    np_dta2 = np.asarray(dta2_vectors)

    # result_cosines = list()
    for word, vec1, vec2 in zip(targets, np_dta1, np_dta2):
        print('{}\t{}'.format(word, cosine(vec1, vec2)))

    # result_cosines = np.asarray(result_cosines)
    #
    # rho, p = spearmanr(result_cosines, targets_scores, nan_policy='omit')
    #
    # return rho, p


parser = argparse.ArgumentParser()

parser.add_argument('--low-bound', required=True)
parser.add_argument('--high-bound', required=True)
parser.add_argument('--threshold', required=True,
                    help='top N substitution')
parser.add_argument('--first-subst', required=True,
                    help='path to first substitution archive')
parser.add_argument('--second-subst', required=True,
                    help='path to first substitution archive')
parser.add_argument('--target-words', required=True,
                    help='path to target words file')

args = parser.parse_args()

low_bound = args.low_bound
high_bound = args.high_bound
threshold = args.threshold
first_subst = args.first_subst
second_subst = args.second_subst
target_words = args.target_words

make_pred(first_subst, second_subst, threshold, low_bound, high_bound, target_words)
