
import gc
import os
from os.path import isfile
import time
import random
import pickle

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import KFold, train_test_split
os.system('pip3 install matplotlib')



class Model:
    lgb_isinstalled = False
    dataset = 0
    offline = False

    def __init__(self, datainfo, timeinfo):
        
        self.early_stopping_rounds = 45
        self.num_boost_round = 1000
        self.predict_y = []
        self.params = {
            'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
            'learning_rate': 0.05, 'num_leaves': 50, 'verbose': -1, 'colsample_bytree': 0.6,
            'subsample': 0.8, 'max_depth': 6, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
            'min_split_gain': 0.05, 'min_child_weight': 25, 'num_threads': 4}
        
        
        if Model.offline:
            Model.lgb_isinstalled = True
            self.params['num_threads'] = 30

        if Model.lgb_isinstalled is False:
            start_time = time.time()
            os.system('apt-get update')
            os.system('apt-get install cmake -y')
            os.system('apt-get install build-essential -y')
            os.system("pip3 install lightgbm")
            os.system("pip3 install matplotlib")
            os.system('pip3 install category_encoders')
            # os.system("pip3 install psutil")
            Model.lgb_isinstalled = True
            print("[CheckPoint] install LightGBM spend %.3f" % (time.time()-start_time))
        os.system("pip3 install seaborn")
        from AutoML3_sample_code_submission.preprocessing import FeatureProcessing
        
        print("Loaded %d time features, %d numerical Features, %d categorical features and %d multi valued categorical variables" \
        %(datainfo['loaded_feat_types'][0], datainfo['loaded_feat_types'][1],datainfo['loaded_feat_types'][2],datainfo['loaded_feat_types'][3]))
        # self.X = None
        # self.y = None
        self.datainfo = datainfo
        self.timeinfo = timeinfo
        self.clf = None
        self.feature_clf = None
        self.feature = []
        self.is_trained = False
        self.fp = FeatureProcessing()
        
        self.processed_data = 0
        # import psutil
        # self.proc = psutil.Process(os.getpid())
        Model.dataset = Model.dataset + 1
        self.batchnumber = 0
        # self.num_train_samples = 0
        # self.num_feat = 1
        # self.num_labels = 1

    def fit(self, F, y, datainfo, timeinfo):
        
        print("this is the 2018-10-17")
        import lightgbm as lgb
        print("[CheckPoint] Start fit...")
        print("[CheckPoint] resample: F shape", F['numerical'].shape, "y shape", y.shape)
        
        if self.is_trained == False:
            X = self.fp.fit(F)
        else:
            X = self.processed_data
        del F
        num_sample = 200000
        
        if X.shape[0] > num_sample:
            X, y = self.resample(X, y, rownum=num_sample)
        
        if self.is_trained == False:
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2018)
            train_data = lgb.Dataset(data=X_train, label=y_train.reshape(-1, ), free_raw_data=False)
            val_data = lgb.Dataset(data=X_val, label=y_val.reshape(-1, ), free_raw_data=False)
        
            self.feature_clf = lgb.train(
                self.params, train_data, init_model=self.feature_clf, keep_training_booster=True,
                valid_sets=[train_data, val_data], verbose_eval=100,
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds
            )
        # self.best_iteration = self.clf.best_iteration
        
        
        self.feature = self.__identify_zero_importance()
        print(self.__identify_zero_importance())
        
        X = X[:,self.feature]
        
        self.params['learning_rate'] = self.params['learning_rate'] + 0.01 * self.batchnumber

        if not self.is_trained:
            self.X = X
            self.y = y
        else:
            self.X = np.concatenate([self.X, X], axis=0)
            self.y = np.concatenate([self.y, y], axis=0)
            
            if(self.X.shape[0]>400000):
                self.X = self.X[self.X.shape[0]-400000:self.X.shape[0]]
                self.y = self.y[self.y.shape[0]-400000:self.y.shape[0]]
                
        print("[CheckPoint]: ", self.X.shape)

        print("[CheckPoint] resample: X shape", X.shape, "y shape", y.shape)
        
        print("[CheckPoint] resample: X shape", self.X.shape, "y shape", y.shape)
        
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.03, random_state=2018)
        train_data = lgb.Dataset(data=X_train, label=y_train.reshape(-1, ), free_raw_data=False)
        val_data = lgb.Dataset(data=X_val, label=y_val.reshape(-1, ), free_raw_data=False)
    
        self.clf = lgb.train(
            self.params, train_data, init_model=self.clf, keep_training_booster=True,
            valid_sets=[train_data, val_data], verbose_eval=100,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds
        )
        
        importance = self.clf.feature_importance(importance_type='split')
        feature_name = self.clf.feature_name()
        # for (feature_name,importance) in zip(feature_name,importance):
        #     print (feature_name,importance) 
        feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
        feature_importance.to_csv('feature_importance.csv',index=False)
        
        self.is_trained = True
        del train_data, val_data, X_train, X_val, y_train, y_val, X, y
        gc.collect()
        # self.__show_memory()
        print("[CheckPoint] Fit over!")
        print("train_time___________________________________________",time.time() - timeinfo[1])
        
    def predict(self, F, datainfo, timeinfo):
        
        print("[CheckPoint] Start predict...")
        # self.__show_memory()
        X = self.fp.fit(F)
        self.processed_data = X
        
        X = X[:,self.feature] 
        y_combine = self.clf.predict(X)
        
        
        self.predict_y = y_combine
        print("[CheckPoint] predict over!")
        # self.__show_memory()
        print("fit_time___________________________________________",time.time() - timeinfo[1])
        return y_combine

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

    def resample(self, X, y, rownum=None):
        dfx = pd.DataFrame(X)
        dfy = pd.DataFrame(y)
        dfxs = dfx.sample(n=rownum)
        idx = dfxs.index.values
        dfys = dfy.ix[idx]
        return dfxs.values, dfys.values
    def isrepeat(self,data,y):
        data = pd.DataFrame(data)
        y = pd.DataFrame(y)
        data = data.drop_duplicates()
        y_index = data.index.values
        y = y.ix[y_index].reset_index(drop=True)
        data = data.reset_index(drop=True)
        return data.values,y.values
    def __identify_zero_importance(self):
        score = self.feature_clf.feature_importance()/self.feature_clf.feature_importance().sum()
        if(len(self.feature_clf.feature_importance()))>170:
            return list(np.where(score > np.percentile(score, 70))[0])
        else:
            return list(np.where(score > np.percentile(score, 50))[0])