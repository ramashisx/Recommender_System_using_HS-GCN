import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sparse
import gc

######Loading Data........

data = pd.read_csv('./raw_data/ratings.csv', encoding='utf-8').iloc[:, :3]
# print(data)
data.columns = ['user_id', 'item_id', 'ratings']
print('data loaded...')


def get_count(tp, key):
    count_groupBy_key = tp[[key, 'ratings']].groupby(key, as_index=True)
    count = count_groupBy_key.size()
    return count


min_user = 1
min_item = 1


def filter_triplets(tp, min_uc=min_user, min_ic=min_item):
    
    # Items which are atleast rated once.
    _item_count = get_count(tp, 'item_id')
    tp = tp[tp['item_id'].isin(_item_count.index[_item_count >= min_ic])]

    # users who listened to at least min_uc songs.
    _user_count = get_count(tp, 'user_id')
    tp = tp[tp['user_id'].isin(_user_count.index[_user_count >= min_uc])]
    _item_count = get_count(tp, 'item_id')
    _user_count = get_count(tp, 'user_id')

    return tp, _item_count, _user_count


data, item_count, user_count = filter_triplets(data)

unique_uid = user_count.index
unique_iid = item_count.index

user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
item2id = dict((iid, i) for (i, iid) in enumerate(unique_iid))

user_num = len(user2id)  # total number of users
item_num = len(item2id)  # total number of items
interaction_num = data.shape[0]  # total number of interactions

print('user_num: %d, item_num: %d, interaction_num: %d' % (user_num, item_num, interaction_num))


def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    iid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(iid)
    return tp


csv_data = numerize(data)

df_gp = csv_data.groupby(['user_id'])
uid_name = df_gp.size().index


def split_train_test(tp_rating):
    n_ratings = tp_rating.shape[0]

    #70-30 train-test split 
    test = np.random.choice(n_ratings, size=int(0.30 * n_ratings), replace=False)

    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    _tp_test = tp_rating[test_idx]
    _tp_train = tp_rating[~test_idx]
    return _tp_train, _tp_test


edge_index = np.empty(shape=[0, 2], dtype=int)
train_ui = np.empty(shape=[0, 2], dtype=int)
test_ui = np.empty(shape=[0, 2], dtype=int)

for name in uid_name:
    tp_train, tp_test = split_train_test(df_gp.get_group(name))
    tr_ui = np.array(tp_train[['user_id', 'item_id']])
    te_ui = np.array(tp_test[['user_id', 'item_id']])
    edge = tr_ui + np.array([0, user_num])
    re_edge = edge[:, [1, 0]]
    edge_index = np.append(edge_index, edge, axis=0)
    edge_index = np.append(edge_index, re_edge, axis=0)
    train_ui = np.append(train_ui, tr_ui, axis=0)
    test_ui = np.append(test_ui, te_ui, axis=0)

row = train_ui[:, 0]
col = train_ui[:, 1]
data = np.ones_like(row)
train_matrix = sparse.coo_matrix((data, (row, col)), shape=(user_num, item_num), dtype=np.int8)
row = test_ui[:, 0]
col = test_ui[:, 1]
data = np.ones_like(row)
test_matrix = sparse.coo_matrix((data, (row, col)), shape=(user_num, item_num), dtype=np.int8)

print("Saving data...")
para = {'user_num': user_num, 'item_num': item_num, 'edge_index': edge_index, 'train_matrix': train_matrix,
        'test_matrix': test_matrix, 'train_ui': train_ui}
pickle.dump(para, open('./para/movie_load.para', 'wb')) #saved to movie_load.para
print('data loading finished...\n\n\n')



######Forming Triplets......
f_para = open('./para/movie_load.para', 'rb')
para = pickle.load(f_para)
user_num = para['user_num']  # total number of users
item_num = para['item_num']  # total number of items
train_matrix = para['train_matrix']
train_ui = para['train_ui']

print('train triple started...')
ratio = 5  # the ratio of positive and negative
item_ids = np.array(list(range(item_num)))
train_triple = np.empty(shape=[0, 3], dtype=int)
mtx = np.array(train_matrix.todense())
para_index = 0
for lh, inter in enumerate(train_ui):
    user_id = inter[0]  # user_id is an 1-D numpy array
    bool_index = ~np.array(mtx[user_id, :], dtype=bool)
    can_item_ids = item_ids[bool_index]  # the id list of 0-value items
    a1 = np.random.choice(can_item_ids, size=ratio, replace=False)  # a1 is an 1-D numpy array
    inter = np.expand_dims(inter, axis=0)
    inter = np.repeat(inter, repeats=ratio, axis=0)
    a1 = np.expand_dims(a1, axis=1)
    triple = np.append(inter, a1, axis=1)
    train_triple = np.append(train_triple, triple, axis=0)
    if lh % 10000 == 9999:
        print('%d completed.....' % (lh + 1))
    if lh % 3e4 == (3e4 - 1) and lh < len(train_ui)-1:
        train_i = train_triple[:, 0]
        train_j = train_triple[:, 1] + user_num
        train_m = train_triple[:, 2] + user_num
        para = {'train_i': train_i, 'train_j': train_j, 'train_m': train_m}

        #saving data to movie_triple
        pickle.dump(para, open('./para/movie_triple_' + str(para_index) + '.para', 'wb'))

        train_triple = np.empty(shape=[0, 3], dtype=int)
        para_index += 1
        del para
        gc.collect()

train_i = train_triple[:, 0]
train_j = train_triple[:, 1] + user_num
train_m = train_triple[:, 2] + user_num
para = {'train_i': train_i, 'train_j': train_j, 'train_m': train_m}
pickle.dump(para, open('./para/movie_triple_'+str(para_index)+'.para', 'wb'))
print('data triplets finished...')

