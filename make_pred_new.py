import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def make_vocab(word_dict, low_threshold, high_threshold):
    words = list([i[0] for i in sorted(word_dict.items(), key=lambda t: t[1])])
    words = words[
            int(low_threshold * len(words)): int(high_threshold * len(words))]
    return set(words)


def string_to_tuple(shit):
    shit = shit.replace('(', '').replace(' ', '')
    num, word = shit[:shit.find(',')], shit[shit.find(',') + 1:]

    return float(num), word[1:-1]


def make_word_vectors(word_dict, global_set, df):
    word_vectors = dict()
    word_list = list()
    for word in word_dict:
        word_list.append(word)
        word_vector = list()
        for word_axis in global_set:
            if word_axis in word_dict[word]:
                word_vector.append(word_dict[word][word_axis] / df[word_axis])
            else:
                word_vector.append(0)
        word_vectors[word] = word_vector
    return word_list, word_vectors


def preprocess_substitutes(x, threshold = None, exclude_lemmas=[]):
    words = [word.strip() for prob, word in x]
    if exclude_lemmas:
        words = [s for s in words if not s in exclude_lemmas]
    if not threshold is None:
        words = words[:threshold]
    return ' '.join(words)


def make_pred(first_subst, second_subst, threshold, min_df,
             max_df, target_words, output, golden_data_path=None):

    subst1 = pd.read_csv(first_subst)['0'].apply(eval)
    subst1 = subst1.apply(lambda x : preprocess_substitutes(x, threshold))
    inp1 = pd.read_csv(first_subst + '.input')

    subst2 = pd.read_csv(second_subst)['0'].apply(eval)
    subst2 = subst2.apply(lambda x : preprocess_substitutes(x, threshold))
    inp2 = pd.read_csv(second_subst + '.input')

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1['substs'] = subst1
    df2['substs'] = subst2
    df1['word'] = inp1['word']
    df2['word'] = inp2['word']

    vecs1 = dict()
    vecs2 = dict()

    words = pd.concat([df1['substs'], df2['substs']]).to_numpy()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=min_df, max_df=max_df)
    vectorizer = vectorizer.fit(words)

    for word in df1['word'].unique():
        subs = df1[df1['word']==word]['substs'].tolist()
        subs_str = ' '.join(subs)
        vecs1[word] = vectorizer.transform([subs_str]).todense()
        # print(vecs1[word].shape)

    for word in df2['word'].unique():
        subs = df2[df2['word']==word]['substs'].tolist()
        subs_str = ' '.join(subs)
        vecs2[word] = vectorizer.transform([subs_str]).todense()
        # print(vecs2[word].shape)

    with open(target_words) as f:
        targets = [line[:-1] for line in f]

    dta1_vectors = list([vecs1[i] for i in targets])
    dta2_vectors = list([vecs2[i] for i in targets])

    np_dta1 = np.asarray(dta1_vectors)
    np_dta2 = np.asarray(dta2_vectors)

    scores = []
    with open(output, 'w') as f:
        for word, vec1, vec2 in zip(targets, np_dta1, np_dta2):
            score = cosine(vec1, vec2)
            f.write('{}\t{}\n'.format(word, score))
            scores.append((word,score))

    if not golden_data_path is None:
        with open(golden_data_path) as input:
            lines = [l.strip().split('\t') for l in input.readlines()]
        golden_scores = [(s[0], float(s[1])) for s in lines]
        predicted_ranking = []
        golden_ranking = []
        for (w, s), (wg, sg) in zip(sorted(scores), sorted(golden_scores)):
            assert w == wg, "words don't match %s %s" % (w, wg)
            predicted_ranking.append(s)
            golden_ranking.append(sg)
        print('spearman score = %f' % spearmanr(predicted_ranking, golden_ranking)[0])


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
parser.add_argument('--output', required=True)
parser.add_argument('--golden', required=False, default=None)

args = parser.parse_args()

low_bound = args.low_bound
high_bound = args.high_bound
threshold = args.threshold
first_subst = args.first_subst
second_subst = args.second_subst
target_words = args.target_words
output = args.output
golden_path = args.golden

low_bound = float(low_bound) if float(low_bound) <= 1.0 else int(low_bound)
high_bound = float(high_bound) if float(high_bound) <= 1.0 else int(high_bound)

make_pred(first_subst, second_subst, int(threshold), low_bound, high_bound, target_words, output, golden_path)
