import datetime as dt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
import sweetviz as sv

DATA_PATH = '../input/'

def get_sessiontarget(df, cartlog, master_cheese):
    cartlog['target'] = 0
    cartlog.loc[cartlog['JAN'].isin(master_cheese['JAN'].unique()), 'target'] = 1
    id2target = dict(cartlog[cartlog['duration']>=180].groupby('session_id')['target'].max())
    return df['session_id'].map(id2target).fillna(0).astype(np.uint8)

def preprocess_datetime(data):
    data['start_at__date'] = pd.to_datetime(data['start_at__date'])
    # data['year'] = data['start_at__date'].dt.year.astype(np.float16)
    # data['month'] = data['start_at__date'].dt.month.astype(np.int16)
    # data['day'] = data['start_at__date'].dt.day.astype(np.int16)
    data['week'] = data['start_at__date'].dt.dayofweek.astype('category')
    # data = data.drop('start_at__date', axis=1)

    return data

def cart_log_feature(data, cart_log, master, master_cheese):
    idx_before_180 = cart_log['duration'] < 180
    before_180_cart_log = cart_log[idx_before_180]
    price = pd.read_csv(os.path.join(DATA_PATH, "price.csv"))
    JAN2price = dict(zip(price['JAN'], price['avg_price']))
    before_180_cart_log['price'] = before_180_cart_log['JAN'].map(JAN2price) * before_180_cart_log['n_items']

    agg_dict = {
        'n_items': 'sum',
        # 'coupon_is_activated': 'sum',
        # 'duration': ['min', 'max'],
        'JAN': ['count', pd.Series.nunique],
        # 'price': ['min', 'max', np.nanmean, np.nanstd]
    }
    before_180_cart_feature = before_180_cart_log.groupby('session_id').agg(agg_dict)
    before_180_cart_feature.columns = ['cart_'+'_'.join(col) for col in before_180_cart_feature.columns]
    data = pd.merge(data, before_180_cart_feature, how='left', on='session_id')
    return data

def user_cart_feature(data, cart_log, master, master_cheese):
    cart_log = pd.merge(cart_log, data[['session_id', 'user_id']], how='left', on='session_id')
    idx_train = cart_log['session_id'].isin(data[data['test']==False].session_id)
    idx_not_target = ~cart_log['JAN'].isin(master_cheese['JAN'].unique())
    cart_log = cart_log[idx_not_target & idx_train]
    cart_log = department_feature(data, cart_log, master, master_cheese)
    price = pd.read_csv(os.path.join(DATA_PATH, "price.csv"))
    JAN2price = dict(zip(price['JAN'], price['avg_price']))
    cart_log['price'] = cart_log['JAN'].map(JAN2price) * cart_log['n_items']

    agg_dict = {
        'n_items': 'sum',
        'coupon_is_activated': 'sum',
        'duration': ['min', 'max'],
        # 'JAN': pd.Series.nunique,
        'created_at__hour': ['min', 'max', 'median'],
        'created_at__date': pd.Series.nunique,
        'lift_target': ['min', 'max', np.nanmean, np.nanstd],
        # 'lift_top10': 'sum',
        'price': ['min', 'max', np.nanstd, np.nanmean],
    }

    cart_user_feature = cart_log.groupby('user_id').agg(agg_dict)
    cart_user_feature.columns = ['user_'+'_'.join(col) for col in cart_user_feature.columns]
    data = pd.merge(data, cart_user_feature, how='left', on='user_id')
    # for col in data.filter(like='user_', axis=0):
    #     data[col] = data[col].fillna(data[col].mean())
    
    return data

def department_feature(data, cart_log, master, master_cheese):
    jan2department = dict(zip(master['JAN'], master['department']))
    # not_target_log = cart_log[~cart_log['JAN'].isin(master_cheese['JAN'].unique())]
    log_category = cart_log['JAN'].map(jan2department)
    
    use_department = log_category.value_counts()
    idx = log_category.isin(use_department.index)
    _piv = pd.pivot_table(data=cart_log[idx],
                        index="session_id",
                        columns=log_category[idx], 
                        aggfunc="sum",
                        values="n_items")
    _piv = _piv.fillna(0).astype(int)
    target_columns = _piv.columns.tolist()

    same_session_department_df = pd.merge(
        data, 
        _piv, on='session_id', 
        how="left"
    ).groupby("target")[target_columns]\
    .mean()\
    .T
    same_session_department_df["difference"] = same_session_department_df[1] - same_session_department_df[0]
    same_session_department_df["difference_ratio"] = same_session_department_df["difference"] / same_session_department_df[0]
    same_session_department_df = same_session_department_df.sort_values(1, ascending=False)
    same_session_department_df['diff_top10'] = same_session_department_df.index.isin(same_session_department_df.tail(10).index)

    cart_log['department'] = cart_log['JAN'].map(jan2department)
    dapa2lift = dict(zip(same_session_department_df.index, same_session_department_df['difference']))
    cart_log['lift_target'] = cart_log['department'].map(dapa2lift)
    # dapa2lift = dict(zip(same_session_department_df.index, same_session_department_df['diff_top10']))
    # cart_log['lift_top10'] = cart_log['department'].map(dapa2lift)
    
    return cart_log

def dijkstra(data, cart_log, master, master_cheese):
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
    # data = data.sort_values('start_at__date').reset_index(drop=True)
    # data['lag_target'] = data.groupby('user_id').target.shift(1)
    
    data = preprocess_datetime(data)
    data = cart_log_feature(data, cart_log, master, master_cheese)
    data = user_cart_feature(data, cart_log, master, master_cheese)

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

    # target_enc 効かなかった。。。
    # TE_col = ['week']
    # for col in TE_col:
    #     tmp = np.repeat(np.nan, data.shape[0])

    #     target_enc = data[data['test']==False].groupby(col)['target'].mean()
    #     tmp[data['test']==True] = data.loc[data['test']==True, col].map(target_enc)

    #     kf = KFold(n_splits=5, shuffle=True)
    #     for trn, val in kf.split(data[data['test']==False]):
    #         target_enc = data.iloc[trn].groupby(col)['target'].mean()
    #         tmp[val] = data.loc[val, col].map(target_enc)
    #     data['target_enc_'+col] = tmp
    
    data['registor_number'], _ = pd.factorize(data['registor_number'])
    data['registor_number'] = data['registor_number'].astype('category')

    data['user_id'], _ = pd.factorize(data['user_id'])
    data['user_id'] = data['user_id'].astype('category')

    data['distance_to_the_store'] = data['distance_to_the_store'].astype(np.float16)

    return data

if __name__ == '__main__':
    data = get_train_test(-1)
    print(data.head())
    print(data.columns)
    print(data.shape)
    data['target'] = data['target'].astype(bool)
    report = sv.compare([data[data['test']==False].drop(['session_id', 'test', 'start_at__date'], axis=1).reset_index(drop=True), "train"], [data[data['test']==True].drop(['session_id', 'test', 'start_at__date'], axis=1).reset_index(drop=True), "test"])
    report.show_html(os.path.join('../save', "train_vs_test.html"))