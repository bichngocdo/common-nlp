from collections import Counter

from core.data.encoders.helpers import flatten_list


def build_vocab(sequences, cutoff_threshold=0):
    counts = Counter()
    vocab = list()

    for token in flatten_list(sequences):
        counts[token] += 1
    for token, count in counts.items():
        if count >= cutoff_threshold:
            vocab.append(token)

    return vocab
