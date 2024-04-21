from pathlib import Path
import numpy as np
from collections import defaultdict
import math


class Vectorizer:

    def __init__(self, test=False) -> None:
        self.X_raw = None
        self.y = None
        self.X = None
        self._vocabulary = None
        self.ordered_vocabulary = None
        self.term_freqs = None
        self.doc_freqs = None
        self._test = test
        
    def train_test_split(self, X, y, step=5):
        """Returns a train-test split of the dataset."""

        X_copy, y_copy = np.copy(X), np.copy(y)
        X_train, X_test = np.delete(X_copy, np.s_[::step], axis=0), X[::step]
        y_train, y_test = np.delete(y_copy, np.s_[::step]), y[::step]
        return X_train, X_test, y_train, y_test

    def fit(self, path):
        """Learns the vocabulary of the given texts and initializes the classes for y."""

        y = []
        X_raw = []
        term_frequencies = defaultdict(lambda: defaultdict(int))
        doc_frequencies = defaultdict(set)
        ordered_vocabulary = defaultdict()
        rootdir = Path(path)
        
        i = 0
        for f in rootdir.glob('**/*'):
            if f.is_file():
                with open(f, 'r') as fhand:
                    text = []
                    for line in fhand:
                        for word in line.split():
                            word = word.rstrip('\n')
                            text.append(word)
                            ordered_vocabulary[word] = None
                            term_frequencies[i][word] += 1
                            doc_frequencies[word].add(i)
                    i+=1
                    X_raw.append(' '.join(text))

                y.append(-1) if f._parts[1] == 'neg' else y.append(1)

        self._vocabulary = dict([(word, i) for i, word in enumerate(ordered_vocabulary)])  # Used in transform()
        self.ordered_vocabulary = dict([(i, word) for i, word in enumerate(ordered_vocabulary)])
        self.term_freqs = term_frequencies
        self.doc_freqs = doc_frequencies

        self.X_raw, self.y = X_raw, np.asarray(y)

        if self._test:
            self._unittests(raw_data=True)

        return X_raw, np.asarray(y)

    def transform(self, X_raw, mode='tf-idf'):
        """Transforms the raw texts into a term-document matrix.
        
        Can create tf-idf vectors or one-hot binary bag-of-words vectors."""

        X = np.zeros((len(X_raw), len(self._vocabulary)))
        
        for i, text in enumerate(X_raw):
            for word in text.split():
                word = word.lower()
                j = self._vocabulary[word]
                
                if mode == 'tf-idf':
                    count = self.term_freqs[i][word]
                    tf = math.log10(count+1) if count > 0 else 0 
                    idf = math.log10(len(X_raw) / len(self.doc_freqs[word]))
                    value = tf * idf
                elif mode == 'bag-of-words':
                    value = 1

                X[i, j] = value

        self.X = X
        
        if self._test:
            self._unittests(processed_data=True)

        return X
    
    def _unittests(self, raw_data=False, processed_data=False):
        if raw_data:
            assert np.all([isinstance(x, str) for x in self.X_raw])
            assert len(self.X_raw) == self.y.shape[0]
            assert len(np.unique(self.y))==2
            assert self.y.min() == -1
            assert self.y.max() == 1
        
        elif processed_data:
            assert np.min(self.X) == 0
        
    
if __name__ == '__main__':
    process = Vectorizer(test=True)
    X_raw, y = process.fit('txt_sentoken')
    X = process.transform(X_raw)
    breakpoint
