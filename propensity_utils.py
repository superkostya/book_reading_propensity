import pandas as pd
import numpy as np
from scipy import sparse
from statistics import mean

import os
import time

from surprise import Reader, Dataset, SVD, BaselineOnly, NMF, accuracy
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import train_test_split, GridSearchCV, PredefinedKFold, cross_validate

from sklearn.model_selection import train_test_split as skl_train_test_split

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
class Propensity():
    
    def __init__(self, train_data_path, sep=",", skip_lines=1, 
                 algo_class=SVD, algo_params={}, rating_scale=(1, 5), conf_cv_n=10):
        self.train_data_path = train_data_path
        self.skip_lines = skip_lines
        self.algo_class = algo_class #(**algo_params)
        self.algo_params = algo_params
        self.model = algo_class(**self.algo_params)
        self.rating_scale = rating_scale
        self.reader = Reader(line_format="user item rating", 
                             sep=sep, skip_lines=skip_lines, 
                             rating_scale=rating_scale)
        # self.data = Dataset.load_from_file(self.train_data_path, reader=self.reader)
        self.confidence = None
        self.conf_cv_n = conf_cv_n
        self.trained = False
        
    # ------------------------------------------------------------------------------------------------------    
    # Cross-validate on the Train Set
    def cross_validate(self, cv=5, measures=["RMSE", "MAE", "FCP"], verbose=True):
        algo = self.algo_class(**self.algo_params)
        data = Dataset.load_from_file(self.train_data_path, reader=self.reader)
        cross_validate(algo, data, measures=measures, cv=cv, verbose=verbose)
    # ------------------------------------------------------------------------------------------------------    
    # Train & Evaluate on Test Set
    def train_and_test(self, testset_path, verbose=True):
        
        # folds_files is a list of tuples containing file paths:
        train_file = self.train_data_path
        test_file = testset_path
        folds_files = [(train_file, test_file) ]

        data = Dataset.load_from_folds(folds_files, reader=self.reader)
        pkf = PredefinedKFold()

        for trainset, testset in pkf.split(data):

            # train and test algorithm
            self.model.fit(trainset)
            predictions = self.model.test(testset)

            # Compute and print Root Mean Squared Error
            print('Evaluating the model performance on the test set')
            accuracy.rmse(predictions, verbose=verbose)
            accuracy.mae(predictions,  verbose=verbose)
            accuracy.fcp(predictions,  verbose=verbose)
        # -------------------------------------------    
        # Estimate the accuracy on the Test Set
        # Read the Data
        df_test = pd.read_csv(testset_path)
        # Predict the Ratings 
        df_test['rating_predicted'] = df_test.apply(lambda x: self.model.predict(x.user_id, 
                                                                  x.book_id, 
                                                                  verbose=False).est, axis=1)
        
        # Get the predicted binary recommendations ('yes/no' = 1/0)
        df_test['would_recommend'] = df_test['rating'].apply(lambda x: 1 if x >= 4 else 0)
        df_test['would_recommend_pred'] = df_test['rating_predicted'].apply(lambda x: 1 if x >= 4 else 0)
        

        # Estimate the Accuracy 
        acc = df_test[df_test['would_recommend_pred'] == df_test['would_recommend']]['rating'].\
                                count()/df_test.shape[0]

        print(f'Accuracy (Test Set): {acc*100:.2f}%')
    # ------------------------------------------------------------------------------------------------------
    # Estimate the global Confidence for the train data
    def estimate_confidence(self, n_cv_folds=10, verbose=False):
        if verbose:
            print(f'Estimating the Confidence on the Train Set...')
            
        seed = self.algo_params.get('random_state', 42)
        confidence_estimates = []
        df_train_val = pd.read_csv(self.train_data_path)

        for k in range(1,n_cv_folds+1):
            start_time = time.time()
            # Shuffle & Split the Data
            df_train_k, df_val_k = skl_train_test_split(df_train_val, 
                                                        test_size=0.2, 
                                                        random_state=seed+k)
            # Create a Surprise Trainset object
            train = Dataset.load_from_df(df_train_k, self.reader).build_full_trainset()
            # Train the MF model
            algo = self.algo_class(**self.algo_params)
            algo.fit(train)
            # Predict the ratings for the validation set
            df_val_k['rating_predicted'] = df_val_k.apply(lambda x: algo.predict(x['user_id'], 
                                                                          x.book_id, 
                                                                          r_ui=x.rating, 
                                                                          verbose=False).est, axis=1)
            # Convert the rating column into a binary classification (1/0 = "yes/no") column
            df_val_k['would_recommend'] = df_val_k['rating'].apply(lambda x: 1 if x >= 4 else 0)
            df_val_k['would_recommend_pred'] = df_val_k['rating_predicted'].\
                                                    apply(lambda x: 1 if x >= 4 else 0)
            # Make a point esimate for the confidence as a probability of a correct prediction
            conf = df_val_k[df_val_k['would_recommend_pred'] == df_val_k['would_recommend']]['rating'].\
                        count()/df_val_k.shape[0]
            confidence_estimates.append(conf)
            time_elapsed = time.time() - start_time
            if verbose:
                print(f'Fold {k:>2d}: C = {conf:.5f}. Time elapsed: {time_elapsed/60:>5.2f} minutes')
            del df_train_k, df_val_k

        confidence_on_train = mean(confidence_estimates)
        if verbose:
            print('*'*60)
            print(f'Estimated Confidence (on a {k}-fold CV): {confidence_on_train:.2f}')
            print('*'*60)
        self.confidence = confidence_on_train
        return confidence_on_train

    # ------------------------------------------------------------------------------------------------------
    # Train the Model
    def train_model(self, verbose=False): 
        data = Dataset.load_from_file(self.train_data_path, reader=self.reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        del trainset
        # ------------------------------------------------------------
        # Estimate the Confidence on the Train Set
        if self.confidence is None:
            conf = self.estimate_confidence(n_cv_folds=self.conf_cv_n, verbose=verbose)
        else:
            conf = self.confidence
        # ------------------------------------------------------------
        # Estimate the user confidence (a probability of a correct propensity prediction for a given user)
        # Read the Data
        df_train = pd.read_csv(self.train_data_path)
        # Predict the Ratings 
        df_train['rating_predicted'] = df_train.apply(lambda x: self.model.predict(x.user_id, 
                                                                  x.book_id, 
                                                                  verbose=False).est, axis=1)
        
        # Get the predicted binary recommendations ('yes/no' = 1/0)
        df_train['would_recommend'] = df_train['rating'].apply(lambda x: 1 if x >= 4 else 0)
        df_train['would_recommend_pred'] = df_train['rating_predicted'].apply(lambda x: 1 if x >= 4 else 0)
        df_train['prediction_is_correct'] = (df_train['would_recommend_pred'] == \
                                             df_train['would_recommend']).astype(int)
        # User Confidence
        # df_train['confidence'] = self.confidence
        df_train['user_correct_pred_cnt'] = df_train.groupby(['user_id'])['prediction_is_correct'].\
                                            transform("sum")
        df_train['user_rating_cnt'] = df_train.groupby(['user_id'])['rating'].\
                                            transform("count")
        df_train['user_correct_pred_fraction'] = df_train['user_correct_pred_cnt'] / df_train['user_rating_cnt']
        df_train['confidence'] = (10 * self.confidence + df_train['user_correct_pred_cnt']) / \
                                (10 + df_train['user_rating_cnt'])
        self.train_data_fitted = df_train #[['user_id', 'book_id', 'rating', 'rating_predicted', 'would_recommend_pred', 'confidence']]
        self.user_confidence = df_train[['user_id', 'confidence']].drop_duplicates()
    # ------------------------------------------------------------------------------------------------------
    # Get the user-based confidence
    def get_user_confidence(self, user_id):
        if self.user_confidence.user_id.apply(lambda uid_list: user_id in uid_list).any():
            confidence = float(self.user_confidence.loc[self.user_confidence.user_id == user_id, 'confidence'])
        else:
            confidence = self.confidence
        return confidence
    # ------------------------------------------------------------------------------------------------------
    # Infer the propensity for a user-item pair
    def infer_propensity_for_pair(self, user_id, item_id, verbose=True):
        rating = self.model.predict(user_id, item_id, verbose=verbose).est
        propensity = 1 if rating >= 4 else 0
        return propensity, self.get_user_confidence(user_id)
    # ------------------------------------------------------------------------------------------------------
    # Batch Prediction: Infer the propensity for the entire dataset
    def infer_propensity_from_df(self, data_path, verbose=True):
        # Read the Data
        df_test = pd.read_csv(data_path)
        # Predict the Ratings 
        df_test['rating_predicted'] = df_test.apply(lambda x: self.model.predict(x.user_id, 
                                                                  x.book_id, 
                                                                  verbose=False).est, axis=1)
        
        # Get the predicted binary recommendations ('yes/no' = 1/0)
        df_test['would_recommend_pred'] = df_test['rating_predicted'].apply(lambda x: 1 if x >= 4 else 0)
        
        df_test['confidence'] = self.confidence
        df_test['confidence'] = df_test[['user_id']].merge(self.user_confidence, how='left')['confidence']
        df_test['confidence'].fillna(self.confidence, inplace=True)
        # df_test['user_id'].apply(lambda uid: self.get_user_confidence(uid))
        
        return df_test[['user_id', 'book_id', 'would_recommend_pred', 'confidence']]