english_words = dict()

with open('target_words/english.txt') as f:
    for line in f:
        word, _ = line.split('\t')
        english_words[word[:-3]] = word

scores = dict()
for name in ['english', 'german', 'latin', 'swedish']:
    with open('answer/task2/{}.txt'.format(name)) as f:
        scores[name] = list()
        for line in f:
            word, score = line.split('\t')
            score = float(score)
            scores[name].append((word, score))
for name in ['english', 'german', 'latin', 'swedish']:
    with open('answer/task1/{}.txt'.format(name), 'w') as f:
        for word, score in scores[name]:
            if name == 'english':
                word = english_words[word]
            f.write('{}\t{}\n'.format(word, int(score > 0.12)))

task2 = list()
with open('answer/task2/english.txt') as f:
    for line in f:
        word, score = line.split('\t')
        score = float(score)
        task2.append((word, score))

with open('answer/task2/english.txt', 'w') as f:
    for tup in task2:
        f.write('{}\t{}\n'.format(english_words[tup[0]], tup[1]))

