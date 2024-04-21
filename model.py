import numpy as np
import math


class Model:

    def __init__(self, eta0=0.1, alpha=0.001, theta=None, max_epochs=40, n_batches=10, decay_rate=0.9) -> None:
        """Inits Model with class fields:
        self.theta: The parameters of the model. If None will be randomly initialized in method fit.
        self.alpha: The regulariser dampening. 
        self.learning_rate: The initial learning rate of the model.
        self.beta: The decay rate used in RMSprop.
        self.loss: A list of all recorded loss values recorded during the fitting of the model.
        self.min_epochs: The minimum number of epochs the fitting has to go on for.
        self.n_batches: The number of batches the arrays are divided in.
        self.best_loss: The best recorded loss during the fitting of the model.
        self._v: The running average of the gradient.
        self._exit: The condition for the fit method to exit. Will exit when _exit == 5."""

        self.theta = theta
        self.alpha = alpha
        self.learning_rate = eta0
        self.beta = decay_rate
        self.loss = list()
        self.max_epochs = max_epochs
        self.n_batches = n_batches
        self.best_loss = float('inf')
        self._v = None
        self._exit = 0
        
    def fit(self, X, y):
        """Fits model with the Stochastic gradient descent algorithm + RMSprop.
        
        For each mini batch of X and y, calculates the gradient of the loss function,
        through Root Mean Square propagation, calculate the update of the parameters theta.
        The function will exit either when the current loss > best_loss - 0.0001 for 5 
        consecutive epochs, or when more than 300 epochs have passed and the best loss
        is higher than 1000.
        
        Args: 
            X: A 2d numpy array of input vectors.
            y: A flat numpy array of output class for each vector x."""

        X = np.insert(X, 0, 1, axis=1)  # Insert dummy value to handle theta_0 during dot product calculation.
        if self.theta == None: 
            self.theta = np.random.normal(loc=0, scale=1, size=X.shape[1])
        
        self.loss.append(self._loss(X, y))

        for n_epoch in range(self.max_epochs):
            for X_batch, y_batch in self._get_batches(X, y):
                gradient = self._gradient(X_batch, y_batch)
                self.theta = self.theta - self._rmsprop(gradient)
                loss = self._loss(X, y)
                self.loss.append(loss)

            if self._exit_criterion():
                break

            if n_epoch % 10 == 0 and n_epoch != 0:
                print(n_epoch, self.best_loss)

        print(f'Finished with Model fit after {n_epoch} epochs. Best loss: {self.best_loss}')

        """
        n_epoch = 0
        while not self._exit_criterion():
            
            for X_batch, y_batch in self._get_batches(X, y):
                gradient = self._gradient(X_batch, y_batch)
                self.theta = self.theta - self._rmsprop(gradient)
                loss = self._loss(X, y)
                self.loss.append(loss)

            if n_epoch > 300 and self.best_loss > 1000:
                break

            n_epoch += 1
            if n_epoch % 10 == 0:
                print(n_epoch, self.loss[-1])

        print(f'Finished with Model fit after {n_epoch} epochs. Best loss: {self.best_loss}')
        """

    def predict(self, X):
        """Predicts the output class y for each input x.
        
        Calculates the output based on the sign of the dot product of the 
        parameter and vector x.
        
        Args:
            X: A 2d numpy array of input vectors.
        
        Returns:
            y_pred: A flat numpy array of classes for each value x."""
        
        X = np.insert(X, 0, 1, axis=1) 
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            product = np.dot(self.theta, X[i])
            if product < 0: 
                y_pred[i] = -1
            elif product == 0: 
                y_pred[i] = 0
            elif product > 0: 
                y_pred[i] = 1
        return y_pred

    def score(self, X, y):
        """Calculates the score of the model based on the accuracy of the predictions."""

        return np.sum(self.predict(X)==y)/len(y)

    def _get_batches(self, X, y):
        """Gets a random sample of X and y to be used as mini batches by the model.
        
        Stacks y to the end of the array X, then samples the combined array xy,
        and returns the unpacked X and y sample."""

        y = y.reshape((y.shape[0], 1))
        xy = np.hstack((X, y))
        rng = np.random.default_rng()
        rng.shuffle(xy)
        samples = np.array_split(xy, indices_or_sections=self.n_batches)
        for sample in samples:
            yield sample[:, :-1], sample[:, -1:].flatten()
    
    def _exit_criterion(self):
        """Returns True if the exit criterion is fulfilled:
        
        5 consecutive epochs where current loss > best_loss - 0.0001,
        and the minimum number of epochs has passed."""

        if self.loss[-1] > (self.best_loss - 0.0001):
            self._exit += 1
        elif self.loss[-1] < self.best_loss:
            self.best_loss = self.loss[-1]
            self._exit = 0
        else:
            self._exit = 0
        return self._exit == 5
            
    def _rmsprop(self, gradient):
        """Returns the update of the parameters based on the learning rate divided by the averaged gradient. 
        
        First calculates the running average of the gradient based on the value beta (default: 0.9),
        then divides the learning rate by that value and returns it as the update for the parameters."""

        if self._v is None:
            self._v = np.ones((50921,))
        
        self._v =  self.beta * self._v + (1 - self.beta) * np.square(gradient)
        update = self.learning_rate / np.sqrt(self._v) * gradient
        return update

    def _gradient(self, X, y):
        """Calculates the gradient of the loss function."""

        summed = 0
        for i in range(X.shape[0]):
            if y[i] * np.dot(self.theta, X[i]) >= 1:
                summed += 0
            else: 
                summed += -y[i]*X[i]

        return self.alpha*self.theta + summed

    def _loss(self, X, y):
        """Calculates the loss of the model. 
        
        Hinge loss in combination with l2 regularisation is used."""

        return self._regularization() + np.sum(np.maximum(0, (1 - X.dot(self.theta)) * y))        

    def _regularization(self):
        """Performs L2-regularization on the parameter vector."""
        
        return (self.alpha / 2) * np.sqrt(np.sum(np.square(self.theta)))