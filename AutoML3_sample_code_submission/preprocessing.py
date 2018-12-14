# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, BayesianRidge, SGDRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA
import math
import time
from multiprocessing import Pool
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

def counts(df,columns):
    df_cnt = pd.DataFrame()
    for col in columns:
        gb = df.groupby(col).size().reset_index()
        gb.rename(columns={col: col, 0: col+'_cnt'}, inplace=True)
        cnt = pd.merge(df[[col]], gb, how='left', left_on=col, right_on=col)
        df_cnt = pd.concat([df_cnt, cnt[col+'_cnt']], axis=1)
    return df_cnt



class FeatureProcessing():
    def __init__(self):
        self.target_type = 1
        self.mvEncoder = []
        self.catEncoder = []
        self.is_trained = False
        self.number = 0
        self.num_threads = 4
        
    def fit(self, F):
        
        start_time = time.time()
        NUM = self.__num_feat(F['numerical'])
        num_time = time.time() - start_time
        print("num_time______________",num_time)
        
        CAT = self.__cat_feat(F['CAT'])
        cat_time = time.time() - start_time
        print("cat_time______________",cat_time)
        
        MV = self.__mv_feat(F['MV'])
        
        mv_time = time.time() - start_time
        print("mv_time______________",mv_time)
        
        
        
        if len(F['CAT'])!=0:
            data = np.concatenate((F['numerical'],F['CAT']),axis=1)
        if len(F['MV'])!=0:
            data = np.concatenate((data,F['MV']),axis=1)
        
        data = pd.DataFrame(data)
        # new_y = pd.DataFrame(y)
        null_total = np.array(data.isnull().sum(axis=1)).reshape(-1,1)
        
        
        print("[CheckPoint]", 'NUM shape', NUM.shape, ' CAT shape', CAT.shape, ' MV shape', MV.shape)
        if MV.shape[0] > 0:
            X = np.concatenate((NUM, CAT, MV, null_total), axis=1)
        else:
            X = np.concatenate((NUM, CAT, null_total), axis=1)
        print(X.shape)
        return X

    def __num_feat(self, data):
        # start_time = time.time()
        print("without denosing.....")
        return data

    def __cat_feat(self, df):
        start_time = time.time()
        if len(df) != 0:
            
            cat = self._cat_encoder(df.copy())
            print("cat_______________", time.time()-start_time)
            # log = self.cat_gen_feats(df.copy())    
            print("log_______________", time.time()-start_time)
        df.columns = df.columns.map(lambda x: 'CAT_'+str(x))
        df = self.__cat_value_counts(df)
        print("counts_______________", time.time()-start_time)
        
        df = np.concatenate([df, cat], axis=1)

        return df
    
    def __mv_feat(self, df):
        if len(df) != 0:
            mv = self._mv_encoder(df)
        if df.__len__() == 0: return pd.DataFrame().values
        def mv_len(x):
            if isinstance(x, str):
                return x.split(',').__len__()
            else:
                return 0
        # df = pd.concat(tmp, axis=1)
        df = self.__cat_cnt_feat(df)
        for cl in df.columns:
            df.rename(columns={cl: 'multi' + str(cl)}, inplace=True)
        # print('multi cate df *****', df.head())
        
        df = pd.concat([df,mv],axis=1)
        # print("mv_feat______________",time.time()-start_time)
        # df = np.concatenate([df,mv_vector],axis=1)
        return df.values

    # 种类特征：统计出现次数
    def __cat_cnt_feat(self, df, inplace=True):
        df = df.fillna(0)
        for clf in df.columns:
            t = df[[clf]].groupby([clf]).size().reset_index().rename(columns={0: clf + '_cnt'})
            df = df.merge(t[[clf, '%s_cnt' % clf]], on=clf, how='left')
            del df[clf]
        return df
        
    def _cat_encoder(self,df):
        # start_time = time.time()
        df = df.fillna(0)
        from category_encoders import OrdinalEncoder
        if self.is_trained==False:
            enca = OrdinalEncoder().fit(df)        
            self.catEncoder.append(enca)
        cat = self.catEncoder[0].transform(df)
        # print("cat_encoder______________",time.time()-start_time)
        return cat
    
    def _mv_encoder(self,df):
        # start_time = time.time()
        df = df.fillna(0)
        from category_encoders import OrdinalEncoder
        if self.is_trained==False:
            enca = OrdinalEncoder().fit(df)        
            self.mvEncoder.append(enca)
        mv = self.mvEncoder[0].transform(df)
        self.is_trained = True
        # print("mv_encoder______________",time.time()-start_time)
        return mv
        
    def __cat_value_counts(self, df):
    
        pool = Pool(processes = self.num_threads)
        col_num = int(np.ceil(df.columns.shape[0] / self.num_threads))

        res1 = pool.apply_async(counts, args=(df, df.columns[:col_num]))
        res2 = pool.apply_async(counts, args=(df, df.columns[col_num:2*col_num]))
        res3 = pool.apply_async(counts, args=(df, df.columns[2*col_num:3*col_num]))
        res4 = pool.apply_async(counts, args=(df, df.columns[3*col_num:]))
        pool.close()
        pool.join()

        df_counts = pd.concat([res1.get(), res2.get(), res3.get(), res4.get()], axis=1)
        return df_counts
