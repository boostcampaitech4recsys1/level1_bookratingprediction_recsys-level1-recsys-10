import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.utils import shuffle
import pickle
import tqdm

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2):
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.replace('na', np.nan) 
    users = users.replace('', np.nan)

    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
    location_list = []
    
    for location in modify_location:
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass
        
    for location in location_list:
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]
        
    users = users.drop(['location'], axis=1)
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)
    users.loc[users['age'].isnull(),"age"] = users[users['age'].notnull()]["age"].mean()
    users.loc[users['age'].notnull(),"age"] = users[users['age'].notnull()]["age"].apply(age_map)
    
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)        
    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    
    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass  
    
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
    'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
    'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
    'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in categories:
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category'] = category
            
    
    # for col in ['location_city','location_state','location_country',"age"]:
    #     lst = sorted(users[users[col].notnull()][col].unique())
    #     p = users[users[col].notnull()][col].value_counts().sort_index().values
    #     sample_col = random.choices(list(lst),list(p), k=users[col].isnull().sum())
    #     users.loc[users[col].isnull(),col] = sample_col
        
    # for col in ['category', 'language']:
    #     lst = sorted(books[books[col].notnull()][col].unique())
    #     p = books[books[col].notnull()][col].value_counts().sort_index().values
    #     sample_col = random.choices(list(lst),list(p), k=books[col].isnull().sum())
    #     books.loc[books[col].isnull(),col] = sample_col
            
    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df


def context_data_load(args):
    
    # with open('/opt/ml/data.pickle', 'rb') as fr:
    #     data = pickle.load(fr)

    # return data

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    # idx, context_train, context_test = process_context_data(users, books, train, test)
    context_train = pd.read_csv('/opt/ml/input/code/data/context_train_non.csv')
    context_test = pd.read_csv('/opt/ml/input/code/data/context_test_non.csv')
    # xgb = pd.read_csv('/opt/ml/input/code/data/xgb_train.csv')
    # xgb_y = pd.read_csv('/opt/ml/input/code/data/xgb_test.csv')
    # user_mean = context_train.groupby(['user_id']).apply(func)
    # book_mean = context_train.groupby(['isbn']).apply(func)
    
    # for i in tqdm.tqdm(range(context_test.shape[0]), smoothing=0, mininterval=1.0):
    #     try:
    #         context_test.loc[i,"user_mean"] = user_mean.loc[context_test.loc[i,"user_id"],"f_mean"]
    #     except:
    #         context_test.loc[i,"user_mean"] = np.nan
    # for i in tqdm.tqdm(range(context_test.shape[0]), smoothing=0, mininterval=1.0):
    #     try:
    #         context_test.loc[i,"book_mean"] = book_mean.loc[context_test.loc[i,"isbn"],"f_mean"]
    #     except:
    #         context_test.loc[i,"book_mean"] = np.nan
    # for i in tqdm.tqdm(range(context_train.shape[0]), smoothing=0, mininterval=1.0):
    #     context_train.loc[i,"user_mean"] = user_mean.loc[context_train.loc[i,"user_id"],"f_mean"]
    # for i in tqdm.tqdm(range(context_train.shape[0]), smoothing=0, mininterval=1.0):
    #     context_train.loc[i,"book_mean"] = book_mean.loc[context_train.loc[i,"isbn"],"f_mean"]
        
    context_df = pd.concat([context_train,context_test]).reset_index(drop = True)
    # context_df.to_csv('/opt/ml/input/code/data/context_df.csv', index=False)
    
    # xgb2idx = {v:k for k,v in enumerate(xgb['pre_rating'].unique())}
    user_mean2idx = {v:k for k,v in enumerate(context_train['user_mean'].unique())}
    book_mean2idx = {v:k for k,v in enumerate(context_train['book_mean'].unique())}
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    
  
    # context_train.to_csv('/opt/ml/input/code/data/context_train_non.csv', index=False)
    # context_test.to_csv('/opt/ml/input/code/data/context_test_non.csv', index=False)
    
    # field_dims = np.array([len(user2idx), len(isbn2idx),
    #                         6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
    #                         len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    
    # xgb['pre_rating'] = xgb['pre_rating'].map(xgb2idx)
    # context_train['pre_rating'] = xgb['pre_rating']
    # context_test['pre_rating'] = xgb_y['pre_rating']
    idx = {
    "loc_city2idx":loc_city2idx,
    "loc_state2idx":loc_state2idx,
    "loc_country2idx":loc_country2idx,
    "category2idx":category2idx,
    "publisher2idx":publisher2idx,
    "language2idx":language2idx,
    "author2idx":author2idx,
    "user_mean2idx":user_mean2idx,
    "book_mean2idx":book_mean2idx
    # "xgb2idx":xgb2idx
    }
    
    field_dims = np.array([len(user2idx), len(isbn2idx),6,len(idx['loc_country2idx']), len(idx['category2idx']),
                           len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx']),
                           len(idx['user_mean2idx']), len(idx['book_mean2idx'])] ,dtype=np.uint32)
    
    # field_dims = np.array([len(user2idx), len(isbn2idx),6, len(idx['category2idx']),
    #                        len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])] ,dtype=np.uint32)
    
    # field_dims = np.array([len(user2idx), len(isbn2idx),
    #                         6,  len(idx['loc_country2idx']),
    #                         len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    
    
    data = {
            'train':context_train.drop(['location_city','location_state'], axis=1),
            'test':context_test.drop(['rating','location_city','location_state'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):    
    # data["train"]["rating"] = round(data["train"]["rating"])
    # useridx = (data['train']['user_id'].value_counts()>10).to_dict()
    # bookidx = (data['train']['isbn'].value_counts()>10).to_dict()
    # data['train'] = data['train'][data['train']['user_id'].map(useridx) & data['train']['isbn'].map(bookidx)]
    # data['train']['rating'] = round(data['train']['rating']) 
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                    data['train'].drop(['rating'], axis=1),
                                                    data['train']['rating'],
                                                    test_size=args.TEST_SIZE,
                                                    random_state=args.SEED,
                                                    shuffle=True
                                                    )
    
    # idx = (data['train']["user_id"].value_counts()>10).to_dict()
    # data['train']["one_up"] = data['train']["user_id"].map(idx)
    # data['test']["one_up"] = data['test']["user_id"].map(idx)
    # data['test'].loc[data['test']["one_up"].isnull(),"one_up"] = False 
    # data['test']["one_up"] = data['test']["one_up"].apply(lambda x: bool(x))
    
    # train_norm = data['train'][data['train']["one_up"]].drop(['one_up'], axis=1).reset_index(drop=True)
    # test_norm = data['test'][data['test']["one_up"]].drop(['one_up'], axis=1).reset_index(drop=True)
    
    
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #                                                     train_norm.drop(['rating'], axis=1),
    #                                                     train_norm['rating'],
    #                                                     test_size=args.TEST_SIZE,
    #                                                     random_state=args.SEED,
    #                                                     shuffle=True
    #                                                     )

    # test_norm = data["test"][data["test"]["user_mean"].notnull() & data["test"]["book_mean"].notnull()]
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    # data['test'] = test_norm
    
    # data['X_train'], data['y_train']= data['train'].drop(['rating'], axis=1), data['train']['rating']
    return data

    with open('data.pickle', 'rb') as fr:
        data = pickle.load(fr)
    
    idx = (data['train']["user_id"].value_counts()>10).to_dict()
    data['train']["one_up"] = data['train']["user_id"].map(idx)
    data['test']["one_up"] = data['test']["user_id"].map(idx)
    data['test'].loc[data['test']["one_up"].isnull(),"one_up"] = False 
    data['test']["one_up"] = data['test']["one_up"].apply(lambda x: bool(x))
    
    train_norm = data['train'][data['train']["one_up"]].drop(['one_up'], axis=1).reset_index(drop=True)
    train_not_norm = data['train'][~data['train']["one_up"]].drop(['one_up'], axis=1).reset_index(drop=True)
    test_norm = data['test'][data['test']["one_up"]].drop(['one_up'], axis=1).reset_index(drop=True)
    test_not_norm = data['test'][~data['test']["one_up"]].drop(['one_up'], axis=1).reset_index(drop=True)
    train_min_max = train_norm.groupby(['user_id']).apply(func)

    for i in range(train_norm.shape[0]):

        min1 = train_min_max.loc[train_norm.loc[i,"user_id"],"uid_min"]
        max_min = train_min_max.loc[train_norm.loc[i,"user_id"],"max-min"]
        train_norm.loc[i,"rating"] = (train_norm.loc[i,"rating"]-min1)/max_min

    X_train_norm, X_valid_norm, y_train_norm, y_valid_norm = train_test_split(
                                                                            train_norm.drop(['rating'], axis=1),
                                                                            train_norm['rating'],
                                                                            test_size=args.TEST_SIZE,
                                                                            random_state=args.SEED,
                                                                            shuffle=True
                                                                            )

    X_train_not_norm, X_valid_not_norm, y_train_not_norm, y_valid_not_norm = train_test_split(
                                                                                            train_not_norm.drop(['rating'], axis=1),
                                                                                            train_not_norm['rating'],
                                                                                            test_size=args.TEST_SIZE,
                                                                                            random_state=args.SEED,
                                                                                            shuffle=True
                                                                                            )
        
    data['X_train_norm'], data['X_valid_norm'], data['y_train_norm'], data['y_valid_norm'] = X_train_norm, X_valid_norm, y_train_norm, y_valid_norm
    data['X_train_not_norm'], data['X_valid_not_norm'], data['y_train_not_norm'], data['y_valid_not_norm'] = X_train_not_norm, X_valid_not_norm, y_train_not_norm, y_valid_not_norm
    data['train_min_max'], data['test_norm'], data['test_not_norm'] = train_min_max, test_norm, test_not_norm
    
    # with open("data.pickle","wb") as fw:
    #     pickle.dump(data,fw)
        
    # with open('data.pickle', 'rb') as fr:
    #     data = pickle.load(fr)
    
    return data


def func(x):
    d = {}
    d["f_mean"] = np.mean(x['rating'])
    # d["uid_min"] = min(x['rating'])
    # d["uid_max"] = max(x["rating"])
    # d["max-min"] = d["uid_max"] - d["uid_min"]
    return pd.Series(d,index = ["f_mean"])
    

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    # data['train_dataloader'], data['test_dataloader'] = train_dataloader,  test_dataloader

    with open("data.pickle","wb") as fw:
        pickle.dump(data,fw)
    
    return data


    train_norm_dataset = TensorDataset(torch.LongTensor(data['X_train_norm'].values), torch.LongTensor(data['y_train_norm'].values))
    valid_norm_dataset = TensorDataset(torch.LongTensor(data['X_valid_norm'].values), torch.LongTensor(data['y_valid_norm'].values))
    train_not_norm_dataset = TensorDataset(torch.LongTensor(data['X_train_not_norm'].values), torch.LongTensor(data['y_train_not_norm'].values))
    valid_not_norm_dataset = TensorDataset(torch.LongTensor(data['X_valid_not_norm'].values), torch.LongTensor(data['y_valid_not_norm'].values))
    test_norm_dataset = TensorDataset(torch.LongTensor(data['test_norm'].values))
    test_not_norm_dataset = TensorDataset(torch.LongTensor(data['test_not_norm'].values))

    # train_norm_dataloader = DataLoader(train_norm_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    # valid_norm_dataloader = DataLoader(valid_norm_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    # train_not_norm_dataloader = DataLoader(train_not_norm_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    # valid_not_norm_dataloader = DataLoader(valid_not_norm_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    # test_norm_dataloader = DataLoader(test_norm_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    # test_not_norm_dataloader = DataLoader(test_not_norm_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    
    # data['train_norm_dataloader'], data['valid_norm_dataloader'],data['train_not_norm_dataloader'], data['valid_not_norm_dataloader'], data['test_norm_dataloader'], data['test_not_norm_dataloader'] = train_norm_dataloader, valid_norm_dataloader, train_not_norm_dataloader, valid_not_norm_dataloader, test_norm_dataloader, test_not_norm_dataloader

    # return data
