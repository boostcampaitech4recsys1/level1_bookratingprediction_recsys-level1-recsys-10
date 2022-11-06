import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re

def language_map(x) -> int:
    if x == 'en':
        return 1  # 영어면 1
    else:
        return 0  # 아니면 0
    
    
def country_map(x) -> int:
    if type(x) == str:
        if 'usa' in x or 'united kingdom' in x:
            return 1

    return 0


def book_author(x):
    try:
        temp = x.split()
        return temp[0] + temp[-1]
    except:
        return 'NN'


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

def average_map(x):
    if x < 2:
        return 0
    elif 2 <= x < 5:
        return 1
    elif 5 < x <= 8:
        return 2
    elif 8 < x <= 11:
        return 3
    else:
        return 4

def process_context_data(users, books, ratings, ratings2):
    users = users.copy()
    books = books.copy()
    ratings3 = pd.read_csv('/opt/ml/input/code/submit/test_IH2.csv')
    
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    # users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])


    # 인덱싱 맵핑
    user2idx = {v:k for k,v in enumerate(list(users['user_id'].unique()))}
    book2idx = {v:k for k,v in enumerate(list(books['isbn'].unique()))}

    ratings['user_id'] = ratings['user_id'].map(user2idx)
    ratings['isbn'] = ratings['isbn'].map(book2idx)
    ratings2['user_id'] = ratings2['user_id'].map(user2idx)  # prediction이 0인 빈 파일
    ratings2['isbn'] = ratings2['isbn'].map(book2idx)
    ratings3['user_id'] = ratings3['user_id'].map(user2idx)  # 이전까지 가장 좋았던 파일
    ratings3['isbn'] = ratings3['isbn'].map(book2idx)

    users['user_id'] = users['user_id'].map(user2idx)
    books['isbn'] = books['isbn'].map(book2idx)

    n_train = len(ratings)  # 트레인 셋 개수
    n_test = len(ratings2)  # 테스트 셋 개수
    n_user = len(users['user_id'].unique())  # 등록된 유저 수
    n_books = len(books['isbn'].unique())  # 등록된 책 수
    
    # for i in range(n_books):
    #     if str(books.at[i, 'isbn']).startswith('0') or str(books.at[i, 'isbn']).startswith('1'):
    #         books.at[i, 'language'] = 'en'
    books = books.drop(['language'], axis=1)

    user_average_dict = dict(ratings.groupby('user_id')['rating'].mean())
    user_averages = [0 for _ in range(n_user)]
    for i in list(user_average_dict.keys()):
        user_averages[int(i)] = user_average_dict[int(i)]

    book_average_dict = dict(ratings.groupby('isbn')['rating'].mean())
    book_averages = [0 for _ in range(n_books)]
    for i in list(book_average_dict.keys()):
        book_averages[int(i)] = book_average_dict[int(i)]
    
    user_average_dict2 = dict(ratings3.groupby('user_id')['rating'].mean())
    user_averages2 = [0 for _ in range(n_user)]
    for i in list(user_average_dict2.keys()):
        user_averages2[int(i)] = user_average_dict2[int(i)]
    book_average_dict2 = dict(ratings3.groupby('isbn')['rating'].mean())
    book_averages2 = [0 for _ in range(n_books)]
    for i in list(book_average_dict2.keys()):
        book_averages2[int(i)] = book_average_dict2[int(i)]

    u_rate_count1 = np.array([0.0001 for _ in range(n_user)])
    u_rate_count2 = np.array([0.0001 for _ in range(n_user)])
    
    b_rate_count1 = np.array([0.0001 for _ in range(n_books)])
    b_rate_count2 = np.array([0.0001 for _ in range(n_books)])
    
    u_ratings1_dict = ratings.groupby('user_id')['rating'].count()
    u_ratings2_dict = ratings3.groupby('user_id')['rating'].count()
    b_ratings1_dict = ratings.groupby('isbn')['rating'].count()
    b_ratings2_dict = ratings3.groupby('isbn')['rating'].count()

    for i in list(u_ratings1_dict.keys()):
        u_rate_count1[i] += u_ratings1_dict[i]
    for i in list(u_ratings2_dict.keys()):
        u_rate_count2[i] += u_ratings2_dict[i]
    for i in list(b_ratings1_dict.keys()):
        b_rate_count1[i] += b_ratings1_dict[i]
    for i in list(b_ratings2_dict.keys()):
        b_rate_count2[i] += b_ratings2_dict[i]

    del u_ratings1_dict, u_ratings2_dict, b_ratings1_dict, b_ratings2_dict
    
    weighted_user_averages = (np.array(user_averages)*u_rate_count1*0.2 + np.array(user_averages2)*u_rate_count2*0.8)/ np.array(u_rate_count1*0.2+u_rate_count2*0.8)
    users['user_average'] = pd.Series(weighted_user_averages)

    weighted_book_averages = (np.array(book_averages)*b_rate_count1*0.2 + np.array(book_averages2)*b_rate_count2*0.8) / np.array(b_rate_count1*0.2+b_rate_count2*0.8)
    books['book_average'] = pd.Series(weighted_book_averages)

    books['book_author'] = books['book_author'].apply(book_author)
    
    del ratings3, user_average_dict, user_average_dict2, book_average_dict, book_average_dict2
    
    # user location
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
    location_list = []
    for location in modify_location:
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass
    
    for location in location_list:
        # users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]  # 주와 국가를 다시 제대로 채워넣어줌 
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]
    
    ############################################################################################################################################
    # 책 전처리
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values  # 2번 이상 책이 등록되어 있는 출판사의 목록을 뽑아냄 
    
    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]  # 해당 이름을 가진 출판사의 고유번호 중에서 가장 많이 사용된 번호를 number에 저장
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]  # number와 같은 번호를 가지는 출판사 이름 중에서 가장 많이 사용된 이름을 right_publisher로 저장
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher  # 책의 고유번호가 위의 고유 번호와 같은 항목들에 대해서 출판사를 가장 많이 사용된 이름인 right_publisher로 
        except:
            pass

    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_0-9]+',' ',str(x)).strip())
    books['category'] = books['category'].str.lower()
    category_df = pd.DataFrame(books['category'].value_counts()).reset_index()
    category_df.columns = ['category','count']

    books['category_high'] = books['category'].copy()

    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction', 'SF', 'social','homicide',
    'sociology','disease','religion','christian','philosophy','psycholog','math','agricul','environment',
    'business','poet','drama','literary','travel','motion picture','children','cook','literature','elec',
    'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in categories:
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category
    
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'
    
    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(books['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(books['publisher'].unique())}
    # language2idx = {v:k for k,v in enumerate(books['language'].unique())}
    author2idx = {v:k for k,v in enumerate(books['book_author'].unique())}
    category_high2idx = {v:k for k,v in enumerate(books['category_high'].unique())}
    pubyear2idx = {v:k for k,v in enumerate(books['year_of_publication'].unique())}

    books['category'] = books['category'].map(category2idx)
    books['publisher'] = books['publisher'].map(publisher2idx)
    # books['language'] = books['language'].map(language2idx)
    books['book_author'] = books['book_author'].map(author2idx)
    books['category_high'] = books['category_high'].map(category_high2idx)
    books['book_average'] = books['book_average'].apply(average_map)
    books['year_of_publication'] = books['year_of_publication'].map(pubyear2idx)

    ############################################################################################################################################
    # users 전처리

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location', 'location_state'], axis=1)
    users['location_country'] = users['location_country'].apply(country_map)


    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(users['location_city'].unique())}
    # loc_state2idx = {v:k for k,v in enumerate(users['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(users['location_country'].unique())}


    users['location_city'] = users['location_city'].map(loc_city2idx)
    # train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    users['location_country'] = users['location_country'].map(loc_country2idx)
    users['location_city'] = users['location_city'].map(loc_city2idx)
    # test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    users['location_country'] = users['location_country'].map(loc_country2idx)
       
    users['age'] = users['age'].fillna(int(users['age'].mean()))
    users['age'] = users['age'].apply(age_map)
    users['user_average'] = users['user_average'].apply(average_map)

    books = books.drop(['img_url', 'summary', 'img_path', 'book_title'], axis=1)
    users = users.drop(['location_city'], axis=1)
    
    books_short = books.iloc[b_rate_count1>10, :]
    users_short = users.iloc[u_rate_count1>10, :]

    train_df = ratings.merge(books, on='isbn', how='left').merge(users, on='user_id', how='left').drop(['user_average', 'book_average'], axis=1)
    # train_df = ratings.merge(books, on='isbn', how='left').merge(users, on='user_id', how='left')
    train_short_df = ratings.merge(books_short, on='isbn', how='left').merge(users_short, on='user_id', how='left')
    test_df = ratings2.merge(books, on='isbn', how='left').merge(users, on='user_id', how='left').drop(['user_average', 'book_average'], axis=1)
    test_short_df = ratings2.merge(books_short, on='isbn', how='left').merge(users_short, on='user_id', how='left')
    
    rate_list = [False for _ in range(len(ratings2))]

    for i in range(len(ratings2)):
        if u_rate_count1[ratings.iloc[i, 0]] > 10 and b_rate_count1[ratings.iloc[i, 1]] >10:
            rate_list[i] = True
            
    short_list = list(ratings2[rate_list].index)

    idx = {
        # "loc_city2idx":loc_city2idx,
        # "loc_state2idx":loc_state2idx,
        "user2idx":user2idx,
        "book2idx":book2idx,
        "category_high2idx":category_high2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        # "language2idx":language2idx,
        "author2idx":author2idx,
        "pubyear2idx":pubyear2idx
    }

    return idx, train_df, train_short_df, test_df, test_short_df, short_list


def context_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    # idx2user = {idx:id for idx, id in enumerate(ids)}
    # idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    # user2idx = {id:idx for idx, id in idx2user.items()}
    # isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    # train['user_id'] = train['user_id'].map(user2idx)
    # sub['user_id'] = sub['user_id'].map(user2idx)
    # test['user_id'] = test['user_id'].map(user2idx)
    # users['user_id'] = users['user_id'].map(user2idx)

    # train['isbn'] = train['isbn'].map(isbn2idx)
    # sub['isbn'] = sub['isbn'].map(isbn2idx)
    # test['isbn'] = test['isbn'].map(isbn2idx)
    # books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_train_short, context_test, context_test_short, short_list = process_context_data(users, books, train, test)
    # field_dims = np.array([len(user2idx), len(isbn2idx),
    #                         6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
    #                         len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
    len_book_average = len(pd.concat([context_train_short, context_test_short])['book_average'].unique())
    len_user_average = len(pd.concat([context_train_short, context_test_short])['user_average'].unique())

    field_dims = np.array([len(idx['user2idx']), len(idx['book2idx']), len(idx['author2idx']), len(idx['pubyear2idx']), len(idx['publisher2idx']), 
                            len(idx['category2idx']), len(idx['category_high2idx']), 6, len(idx['loc_country2idx'])], dtype=np.uint32)
    field_dims_short = np.array([len(idx['user2idx']), len(idx['book2idx']), len(idx['author2idx']), len(idx['pubyear2idx']), len(idx['publisher2idx']),
                            len(idx['category2idx']), len_book_average, len(idx['category_high2idx']), 6, len(idx['loc_country2idx']), len_user_average], dtype=np.uint32)
    
    n_test = len(context_test_short)

    data = {
            'train':context_train,
            'train_short': context_train_short,
            'test':context_test.drop(['rating'], axis=1),
            'test_short': context_test_short.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'field_dims_short':field_dims_short,
            'users':users,
            'books':books,
            'sub':sub,
            'short_list': short_list,
            'n_test' : n_test
            # 'idx2user':idx2user,
            # 'idx2isbn':idx2isbn,
            # 'user2idx':user2idx,
            # 'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

    X_train_short, X_valid_short, y_train_short, y_valid_short = train_test_split(
                                                        data['train_short'].drop(['rating'], axis=1),
                                                        data['train_short']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )    
    data['X_train_short'], data['X_valid_short'], data['y_train_short'], data['y_valid_short'] = X_train_short, X_valid_short, y_train_short, y_valid_short

    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    train_short_dataset = TensorDataset(torch.LongTensor(data['X_train_short'].values), torch.LongTensor(data['y_train_short'].values))
    valid_short_dataset = TensorDataset(torch.LongTensor(data['X_valid_short'].values), torch.LongTensor(data['y_valid_short'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    test_short_dataset = TensorDataset(torch.LongTensor(data['test_short'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    train_short_dataloader = DataLoader(train_short_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_short_dataloader = DataLoader(valid_short_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    test_short_dataloader = DataLoader(test_short_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['train_short_dataloader'], data['valid_short_dataloader'], data['test_dataloader'], data['test_short_dataloader'] = train_dataloader, valid_dataloader, train_short_dataloader, valid_short_dataloader, test_dataloader, test_short_dataloader

    return data
