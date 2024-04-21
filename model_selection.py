from sklearn.model_selection import ParameterSampler
from collections import defaultdict
import numpy as np
from model import Model


class ModelSelection:

    def split(self, X, y, k=10):
        """Returns a k-fold split of the X and y arrays.
        
        The fold is returned as a list with tuples. The first element of the 
        tuple is the training data and the second is the test data."""

        rng = np.random.default_rng()
        y = y.reshape((y.shape[0], 1))
        xy = np.hstack((X, y))
        rng.shuffle(xy)
        folds = np.split(xy, indices_or_sections=k)
        return [(folds[:i]+folds[i+1:], folds[i]) for i in range(len(folds))]

    def cross_validation(self, fold, learning_rate, reg_dampening):
        """Performs cross validation of the given folds of data."""

        test_scores = []
        for train, test in fold:
            train = np.concatenate(train)
            X_train, y_train = train[:, :-1], train[:, -1:].flatten()
            X_test, y_test = test[:, :-1], test[:, -1:].flatten()
            
            model = Model(eta0=learning_rate, alpha=reg_dampening)
            model.fit(X_train, y_train)
            test_scores.append(np.sum(model.predict(X_test)==y_test)/len(y_test))
            print('SCORE: ', np.sum(model.predict(X_test)==y_test)/len(y_test))

        print(f'Average accuracy for ten-fold cross validation: {sum(test_scores)/len(test_scores)}')

    def optimize_hyperparameters(self, X, y, min=0.001, max=3, size=10, n_iter=10):
        """Searches for the optimal hyperparameters with the use of SK-learn's 
        ParameterSampler.
        
        Prints a table with the hyperparameters tested and their accuracy"""

        param_distribution = {'learning_rate': np.exp(np.linspace(np.log(min), np.log(max), size)),
                                    'reguliser_dampening': np.exp(np.linspace(np.log(min/10), np.log(max), size))}
        
        param_sampler = ParameterSampler(param_distribution, n_iter=n_iter)
        sampled_params = defaultdict(int)  # For storing hyperparameters
        for hyperparameters in param_sampler:
            reguliser_dampening = hyperparameters['reguliser_dampening']
            learning_rate = hyperparameters['learning_rate']
            
            model = Model(eta0=learning_rate, alpha=reguliser_dampening)

            model.fit(X, y)
            accuracy = np.sum(model.predict(X)==y)/len(y)
            sampled_params[(learning_rate, reguliser_dampening)] = accuracy

        print("Learning rate:\tReg.dampening:\tTraining set accuracy:")
        for (lr, reg_damp), accuracy in sorted(sampled_params.items(), key=lambda x: x[1], reverse=True):
            print("%.5f\t\t%.5f\t\t%.1f%%" % (lr, reg_damp, 100*accuracy))
        
