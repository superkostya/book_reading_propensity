import pandas as pd
import numpy as np
from scipy import sparse
import os

from surprise import Reader, Dataset, SVD, SVDpp, BaselineOnly, NMF, accuracy
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import train_test_split, GridSearchCV, PredefinedKFold

# ------------------------------------------------------------------------------------------------
# REF: https://rubikscode.net/2020/04/27/collaborative-filtering-with-machine-learning-and-python/
class DatasetBuilder():
    def __init__(self, data_location):
        reader = Reader(rating_scale=(1, 5))
        self.ratings = pd.read_csv(data_location)
        
        self.dataset = Dataset.load_from_df(self.ratings[['user_id', 'book_id', 'rating']], reader)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.2)
        
        
class TrainEvalAlgos():
    def __init__(self, dataset):
        self.algos = []
        self.dataset = dataset
        
    def addAlgorithm(self, algo):
        self.algos.append(algo)
        
    def train_and_evaluate(self):
        for algo in self.algos:
            algo.fit(self.dataset.train_dataset)
            predictions = algo.test(self.dataset.test_dataset)
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)
            fcp = accuracy.fcp(predictions)
            print('-----------')
            print(f'{algo.__class__.__name__}') 
            print('-----------')
            print(f'      Metrics - RMSE: {rmse}, MAE: {mae}, FCP: {fcp}')
            print('-----------')
# ------------------------------------------------------------------------------------------------