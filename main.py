from model_selection import ModelSelection
from vectorizer import Vectorizer
from model import Model
import numpy as np


if __name__ == '__main__':
    vectorizer = Vectorizer()
    X_raw, y = vectorizer.fit('txt_sentoken')
    X = vectorizer.transform(X_raw, mode='tf-idf')
    X_train, X_test, y_train, y_test = vectorizer.train_test_split(X, y)

    learning_rate, reg_dampening = 0.1, 0.0001

    ms = ModelSelection()
    ms.optimize_hyperparameters(X_train, y_train)

    fold = ms.split(X, y)
    ms.cross_validation(fold, learning_rate, reg_dampening)
    model = Model(eta0=learning_rate, alpha=reg_dampening)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    # Sort by absolute value
    idx = np.argsort(np.abs(model.theta[1:]))

    print("Word   Weight  Occurences")
    for i in idx[-20:]:   # Pick those with highest 'voting' values
        print("%20s   %.3f\t%i " % (vectorizer.ordered_vocabulary[i], model.theta[i+1], 
                                    np.sum([vectorizer.ordered_vocabulary[i] in d for d in X_raw])))