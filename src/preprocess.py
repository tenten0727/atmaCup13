import datetime as dt
import numpy as np
import pandas as pd
import os

DATA_PATH = '../input/'

def get_sessiontarget(df, cartlog, master_cheese):
    cartlog['target'] = 0
    cartlog.loc[cartlog['JAN'].isin(master_cheese['JAN'].unique()), 'target'] = 1
    id2target = dict(cartlog[cartlog['duration']>=180].groupby('session_id')['target'].max())
    return df['session_id'].map(id2target).fillna(0).astype(np.uint8)

def preprocess_datetime(data):
    data['start_at__date'] = pd.to_datetime(data['start_at__date'])
    # data['year'] = data['start_at__date'].dt.year.astype(np.float16)
    data['month'] = data['start_at__date'].dt.month.astype('category')
    data['day'] = data['start_at__date'].dt.day.astype('category')
    data['week'] = data['start_at__date'].dt.dayofweek.astype('category')
    data = data.drop('start_at__date', axis=1)

    return data

def get_train_test(args):
    data = pd.read_csv(os.path.join(DATA_PATH, "session.csv"))
    test_session = pd.read_csv(os.path.join(DATA_PATH, "test_session.csv"))
    cart_log = pd.read_csv(os.path.join(DATA_PATH, "cart_log.csv"))
    master = pd.read_csv(os.path.join(DATA_PATH, "product_master.csv"))
    master_cheese = master[master['category']=='チーズ']

    data['test'] = data['session_id'].isin(test_session['session_id'])
    data['target'] = get_sessiontarget(data, cart_log, master_cheese)
    data.loc[data['test']==True, 'target'] = -1

    #targetのlag特徴量
    data = data.sort_values('start_at__date')
    data['lag_target'] = data.groupby('user_id').target.shift(1)

    data = preprocess_datetime(data)

    data.loc[data['distance_to_the_store']=='不明', 'distance_to_the_store'] = np.nan
    data['distance_to_the_store'] = data['distance_to_the_store'].fillna(-1)

    sex_dict = {'女性': 0, '男性': 1}    
    data.loc[data['sex']=='不明', 'sex'] = np.nan
    data['sex'] = data['sex'].replace(sex_dict)
    data['sex'] = data['sex'].fillna(-1)
    data['sex'] = data['sex'].astype(int).astype('category')

    age_dict = {'0~4': 2.0, '5~9': 7.0, '10~14': 12.0, '15~19': 17.0, '20~24': 22.0, '25~29': 27.0, '30~34': 32.0,
            '35~39': 37.0, '40~44': 42.0, '45~49': 47.0, '50~54': 52.0, '55~59': 57.0, '60~64': 62.0, '65~69': 67.0,
            '70~74': 72.0, '75~79': 77.0, '80~84': 82.0, '85~89': 87.0, '90~94': 92.0, '95~99': 97.0, '100~': 100.0,
            }
    data.loc[data['age']=='不明', 'age'] = np.nan
    data['age'] = data['age'].replace(age_dict)
    data['age'] = data['age'].fillna(-1).astype(np.float16)

    data['registor_number'], _ = pd.factorize(data['registor_number'])
    data['registor_number'] = data['registor_number'].astype('category')
    data['user_id'], _ = pd.factorize(data['user_id'])
    data['user_id'] = data['user_id'].astype('category')

    data['distance_to_the_store'] = data['distance_to_the_store'].astype(np.float16)

    train = data[data['test']==False].drop(['session_id', 'test'], axis=1).reset_index(drop=True)
    test = data[data['test']==True].drop(['session_id', 'test'], axis=1).reset_index(drop=True)
    return train, test

