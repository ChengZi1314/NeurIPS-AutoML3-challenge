import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from sklearn.feature_selection import RFE, SelectFromModel, RFECV
# from sklearn.linear_model import LogisticRegression, Lasso, Ridge, BayesianRidge, SGDRegressor
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, MinMaxScaler
# from sklearn.decomposition import PCA

# def counts(df, columns, id_feat):
#     df_cnt = pd.DataFrame()
#     for col in columns:
#         if col == id_feat:
#             continue
#         cat_cnt_dict = df.groupby(by=col).count()[id_feat].to_dict()
#         df_cnt[col+'_cnt'] = df[col].apply(lambda x: cat_cnt_dict[x])
#     return df_cnt

def counts(df, columns, id_feat):
    df_cnt = pd.DataFrame()
    for col in columns:
        gb = df.groupby(col).size().reset_index()
        gb.rename(columns={col: col, 0: col+'_cnt'}, inplace=True)
        cnt = pd.merge(df[[col]], gb, how='left', left_on=col, right_on=col)
        df_cnt = pd.concat([df_cnt, cnt[col+'_cnt']], axis=1)
    return df_cnt

# def label_encoder_fit(df, columns):
#     label_encoder = {}
#     for col in columns:
#         enc = LabelEncoder().fit(df[col])
#         label_encoder[col] = enc
#     return label_encoder

# def label_encoder_transform(df, label_encoder, columns):
#     df_trans = pd.DataFrame()
#     for col in columns:
#         df_trans[col] = label_encoder[col].transform(df[col])
#     return df_trans

def tune_type(df, columns):
    df_type = pd.DataFrame()
    for col in columns:
        df_type[col] = df[col].astype('uint64').apply(lambda x: int(100000*np.log(1+x)))
    return df_type

class FeatEngine():
    def __init__(self, datainfo, num_threads=4):
        self.is_trained = False
        self.num_threads = num_threads
        self.datainfo = datainfo
        self.time_feat = datainfo['loaded_feat_types'][0]
        self.num_feat = datainfo['loaded_feat_types'][1]
        self.cat_feat = datainfo['loaded_feat_types'][2]
        self.mv_feat = datainfo['loaded_feat_types'][3]

    def fit(self, F):
        NUM = self.__num_feat(F['numerical'])
        CAT = self.__cat_feat(F['CAT'])
        MV = self.__mv_feat(F['MV'])

        print("[CheckPoint] Feature processing...", 'NUM shape', NUM.shape, ' CAT shape', CAT.shape, ' MV shape', MV.shape)
        if MV.shape[0] > 0:
            X = pd.concat([NUM, CAT, MV], axis=1)
        else:
            X = pd.concat([NUM, CAT], axis=1)

        self.is_trained = True

        return X.values

    def __num_feat(self, data):
        df = pd.DataFrame(data)
        if self.time_feat > 0:
            df.ix[:, :self.time_feat] = df.ix[:, :self.time_feat] - df.ix[:, :self.time_feat].min()

        df.columns = df.columns.map(lambda x:'NUM_'+str(x))
        df = df.fillna(0)
        return df
    
    def __cat_feat(self, df):
        df.columns = df.columns.map(lambda x:'CAT_'+str(x))
        df = df.fillna('0')

        pool = Pool(processes = self.num_threads)
        col_num = int(np.ceil(df.columns.shape[0] / self.num_threads))
        res1 = pool.apply_async(tune_type, args=(df, df.columns[:col_num], ))
        res2 = pool.apply_async(tune_type, args=(df, df.columns[col_num:2*col_num], ))
        res3 = pool.apply_async(tune_type, args=(df, df.columns[2*col_num:3*col_num], ))
        res4 = pool.apply_async(tune_type, args=(df, df.columns[3*col_num:], ))
        pool.close()
        pool.join()
        df = pd.concat([res1.get(), res2.get(), res3.get(), res4.get()], axis=1)
        
        df_counts = self.__cat_value_counts(df)
        df = pd.concat([df, df_counts], axis=1)
        
        return

    def __mv_feat(self, df):
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
        print('multi cate df *****', df.head())
        df = df.fillna(0)
        return df
    
        # def mv_len(x):
        #     if isinstance(x, str):
        #         return x.split(',').__len__()
        #     else:
        #         return 0
        # for col in df.columns:
        #     df[col] = df[col].map(lambda x: mv_len(x))
        # df.columns = df.columns.map(lambda x:'NUM_'+str(x))
        # return pd.concat([df, df_counts], axis=1)

    def __cat_cnt_feat(self, df, inplace=True):
        for col in df.columns:
            gb = df.groupby(col).size().reset_index()
            gb.rename(columns={col: col, 0: col+'_cnt'}, inplace=True)
            df = pd.merge(df, gb, how='left', left_on=col, right_on=col)
            del df[col]
        return df

    def __cat_value_counts(self, df):
        id_feat = None
        max_unique = 0
        for col in df.columns:
            if df[col].unique().shape[0] > max_unique:
                id_feat = col
                max_unique = df[col].unique().shape[0]

        pool = Pool(processes = self.num_threads)
        col_num = int(np.ceil(df.columns.shape[0] / self.num_threads))

        res1 = pool.apply_async(counts, args=(df, df.columns[:col_num], id_feat, ))
        res2 = pool.apply_async(counts, args=(df, df.columns[col_num:2*col_num], id_feat, ))
        res3 = pool.apply_async(counts, args=(df, df.columns[2*col_num:3*col_num], id_feat, ))
        res4 = pool.apply_async(counts, args=(df, df.columns[3*col_num:], id_feat, ))
        pool.close()
        pool.join()

        df_counts = pd.concat([res1.get(), res2.get(), res3.get(), res4.get()], axis=1)
        return df_counts


