import pickle
import os


class NN_Model:
    def __init__(self, model):
        self.model = model

    def predict(self, age, job, marital, education, default, balance, housing, loan, day, month, duration, campaign, pdays, previous, poutcome=0):
        # 15 total features mapped into a 2D array
        X = [[age, job, marital, education, default, balance, housing, loan, day, month, duration, campaign, pdays, previous, poutcome]]
        return self.model.predict(X)