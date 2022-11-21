import warnings

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.layers import Activation


def cleanfit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ret = model.fit(X, y)

    return ret

class ModelComparator:
    def __init__(self, score_func, error_func, X_train, y_train, X_val, y_val, large_number=1000):
        super().__init__()
        self.score_func = score_func
        self.error_func = error_func
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.score = -large_number
        self.error = large_number
        self.best_score = self.score
        self.best_error = self.error
        self.best_score_model = None
        self.best_error_model = None

    def set_data(self, X_train=None, y_train=None, X_val=None, y_val=None):
        if X_train is not None:
            self.X_train = X_train

        if y_train is not None:
            self.y_train = y_train

        if X_val is not None:
            self.X_val = X_val

        if y_val is not None:
            self.y_val = y_val

    def iterate_score(self, score_new, model=None):
        improved = (score_new > self.score)
        is_best = (score_new >= self.best_score)
        self.score = score_new

        if improved:
            if is_best:
                self.best_score = score_new

                if model is not None:
                    self.best_score_model = model

                return "Best!"
            else:
                return "Better!"
        else:
            return "Worse!"

    def iterate_error(self, error_new, model=None):
        improved = (error_new < self.error)
        is_best = (error_new <= self.best_error)
        self.error = error_new

        if improved:
            if is_best:
                self.best_error = error_new

                if model is not None:
                    self.best_error_model = model

                return "Best!"
            else:
                return "Better!"
        else:
            return "Worse!"

    def print_performance(self, model):
        train_score = self.score_func(model, self.X_train, self.y_train)
        val_score = self.score_func(model, self.X_val, self.y_val)

        train_error = self.error_func(model, self.X_train, self.y_train)
        val_error = self.error_func(model, self.X_val, self.y_val)

        print(f"train score: {train_score}")
        print(f"validation score: {val_score}, {self.iterate_score(val_score, model=model)}")

        print(f"train error: {train_error}")
        print(f"validation error: {val_error}, {self.iterate_error(val_error, model=model)}")

        return (train_score, val_score, train_error, val_error)
