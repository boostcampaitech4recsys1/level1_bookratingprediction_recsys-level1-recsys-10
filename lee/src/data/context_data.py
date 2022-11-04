import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

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
    users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

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

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

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

    idx, context_train, context_test = process_context_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
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
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data


books = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\books.csv')
users = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\users.csv')
train = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\train_ratings.csv')
test = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\test_ratings.csv')
sub = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\sample_submission.csv')

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
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.
    
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
    location = users[(users['location'].str.contains('seattle'))&(users['location_country'].notnull())]['location'].value_counts().index[0]

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
        
    users = users.drop(['location','location_state','location_city'], axis=1)

    country_list = users['location_country'].unique()

    for i in country_list:
        a = users['age'][users['location_country']==str(i)].mean()
        users[(users['location_country']==str(i))&(users['age'].isnull())].fillna(a)

    users = users.dropna(axis=0)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    books['language'] = books['language'].fillna('Unknown')
    books['summary'] = books['summary'].fillna('Unknown')
    books['category'] = books['category'].fillna('Unknown')

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

    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())

    books['category'] = books['category'].str.lower()

    category_df = pd.DataFrame(books['category'].value_counts()).reset_index()
    category_df.columns = ['category','count']
    category_df[category_df['count']>=10]

    books['category_high'] = books['category'].copy()
    books.loc[books[books['category']=='biography'].index, 'category_high'] = 'biography autobiography'
    books.loc[books[books['category']=='autobiography'].index,'category_high'] = 'biography autobiography'


    books[books['category'].str.contains('history', na=False)]['category'].unique()

    books.loc[books[books['category'].str.contains('history',na=False)].index,'category_high'] = 'history'

    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
    'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
    'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
    'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    for category in categories:
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']

    # 5개 이하인 항목은 others로 묶어주도록 하겠습니다.
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'

    books = books.drop(['img_url', 'img_path'],axis=1)
    
    
    
    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'category_high']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'category_high']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'category_high']], on='isbn', how='left')

    # 인덱싱 처리
#     loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
#     loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

#     train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
#     train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
#     test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
#     test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)


   



    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    category_high2idx = {v:k for k,v in enumerate(context_df['category_high'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df['category_high'] = train_df['category_high'].map(category_high2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df['category_high'] = test_df['category_high'].map(category_high2idx)

    idx = {
#         "loc_city2idx":loc_city2idx,
#         "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "category_high2idx":category_high2idx
    }

    return idx, train_df, test_df


#     users = pd.read_csv(args.DATA_PATH + 'users.csv')
#     books = pd.read_csv(args.DATA_PATH + 'books.csv')
#     train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
#     test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
#     sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

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

idx, context_train, context_test = process_context_data(users, books, train, test)
# field_dims = np.array([len(user2idx), len(isbn2idx),
#                         6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
#                         len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

field_dims = np.array([len(user2idx), len(isbn2idx),
                        6,len(idx['loc_country2idx']),len(idx['category_high2idx']),
                        len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)


data = {
        'train':context_train,
        'test':context_test.drop(['rating'], axis=1),
        'field_dims':field_dims,
        'users':users,
        'books':books,
        'sub':sub,
        'idx2user':idx2user,
        'idx2isbn':idx2isbn,
        'user2idx':user2idx,
        'isbn2idx':isbn2idx,
        }