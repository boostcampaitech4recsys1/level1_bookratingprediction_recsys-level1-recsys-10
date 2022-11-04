#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

books = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\books.csv')
users = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\users.csv')
train = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\train_ratings.csv')
test = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\test_ratings.csv')
sub = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\sample_submission.csv')


# In[63]:


##############################################################################################원본 EDA

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
#     users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
#     users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)


    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

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

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
#         "loc_city2idx":loc_city2idx,
#         "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df


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
                        6,len(idx['loc_country2idx']),
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


# In[64]:


data


# In[65]:


books = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\books.csv')
users = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\users.csv')
train = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\train_ratings.csv')
test = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\test_ratings.csv')
sub = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\sample_submission.csv')


# In[66]:


users = users.dropna(axis=0)


# In[67]:


users.isna().sum()


# In[68]:


users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거

users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.


# In[69]:


users.isna().sum()


# In[70]:


modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
location = users[(users['location'].str.contains('seattle'))&(users['location_country'].notnull())]['location'].value_counts().index[0]

location_list = []
for location in modify_location:
    try:
        right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
        location_list.append(right_location)
    except:
        pass


# In[71]:


for location in location_list:
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]


# In[72]:


users.isna().sum()


# In[73]:


users = users.drop(['location','location_state','location_city'], axis=1)


# In[74]:


users.isna().sum()


# In[75]:


# 나이 분포
fig, ax = plt.subplots(1, 2, figsize=(20, 7))

users['age'].hist(bins=40, color='teal', ax=ax[0])
sns.boxenplot(data=users, x='age', color='teal',ax=ax[1])

plt.show()


# In[76]:


users['age'].value_counts()


# In[77]:


plt.figure(figsize=(30,5))
users['age'].value_counts().sort_index().plot(kind='bar')
plt.xticks(rotation=30)
plt.show()


# In[78]:


my_dict=(users['location_country'].value_counts()).to_dict()
count= pd.DataFrame(list(my_dict.items()),columns = ['location_country','count'])
f = count.sort_values(by=['count'], ascending = False)
f = f.head(15)
# f.drop(7,inplace=True)
fig=plt.figure(figsize=(11,6))
ax = sns.barplot(y = 'count',x= 'location_country' , data = f)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60,horizontalalignment='center')
for bar in ax.patches: 
    ax.annotate(format(bar.get_height(), '.0f'),  
                   (bar.get_x() + bar.get_width() / 2,  
                    bar.get_height()), ha='center', va='center', 
                   size=8, xytext=(0,8), 
                   textcoords='offset points') 

plt.xlabel("Country", size=10)
plt.ylabel("# of Users", size=10)
plt.show()


# In[79]:


fig, ax = plt.subplots(3,5,figsize=(20,8))
for country, ax_ in zip(f['location_country'], ax.flatten()):
    users[(users['location_country']==country)]['age'].value_counts().sort_index().plot(ax=ax_, title=country)
plt.xlim(0,100)
plt.tight_layout()
plt.show()


# In[80]:


users[users['age'].isna()]['location_country'].value_counts()


# In[81]:


my_dict=(users[users['age'].isna()]['location_country'].value_counts()).to_dict()
count= pd.DataFrame(list(my_dict.items()),columns = ['location_country','count'])
f = count.sort_values(by=['count'], ascending = False)
f = f.head(15)
fig=plt.figure(figsize=(11,6))
ax = sns.barplot(y = 'count',x= 'location_country' , data = f)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60,horizontalalignment='center')
for bar in ax.patches: 
    ax.annotate(format(bar.get_height(), '.0f'),  
                   (bar.get_x() + bar.get_width() / 2,  
                    bar.get_height()), ha='center', va='center', 
                   size=8, xytext=(0,8), 
                   textcoords='offset points') 

plt.xlabel("Country", size=10)
plt.ylabel("# of Users", size=10)
plt.show()


# In[82]:


users['age'].mean()


# In[83]:


users.columns


# In[84]:


users


# In[85]:


a = users['age'][users['location_country']=='usa'].mean()
a


# In[86]:


users.isna().sum()


# In[87]:


country_list = users['location_country'].unique()


# In[88]:


for i in country_list:
    a = users['age'][users['location_country']==str(i)].mean()
    users[(users['location_country']==str(i))&(users['age'].isnull())].fillna(a)


# In[89]:


users


# In[90]:


users.isna().sum()


# In[91]:


users = users.dropna(axis=0)


# In[92]:


users.isna()


# In[93]:


users.isna().sum()


# In[94]:


users.info()


# In[97]:


users.isna().sum()


# In[98]:


users.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\users1.csv")


# In[99]:


books.info()


# In[100]:


books.describe()


# In[101]:


books.shape


# In[102]:


books['isbn'].nunique()


# In[103]:


books['book_title'].nunique()


# In[104]:


books.head()


# In[105]:


books.isna().sum()


# In[106]:


books['language'].unique()


# In[107]:


books['category'].unique()


# In[108]:


a = books.copy()


# In[109]:


a


# In[110]:


a['language'] = a['language'].fillna('Unknown')


# In[111]:


a


# In[112]:


books.isna().sum()


# In[113]:


books['language'] = books['language'].fillna('Unknown')


# In[114]:


books['summary'] = books['summary'].fillna('Unknown')


# In[115]:


books['category'] = books['category'].fillna('Unknown')


# In[116]:


books.isna().sum()


# In[117]:


books['publisher'].nunique()


# In[118]:


publisher_dict=(books['publisher'].value_counts()).to_dict()
publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])

publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)


# In[119]:


modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values


# In[120]:


for publisher in modify_list:
    try:
        number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
        right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
        books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
    except: 
        pass


# In[121]:


books['publisher'].nunique()


# In[122]:


books.isna().sum()


# In[123]:


books


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


books = books.drop(['year_of_publication'],axis=1)


# In[ ]:


books


# In[ ]:


train


# In[ ]:


train['rating'].value_counts(True)


# In[ ]:


train.groupby('user_id')['rating'].count().sort_values(ascending=False)


# In[ ]:


heavy_users_list = train.groupby('user_id')['rating'].count().sort_values(ascending=False).head(20).index


# In[ ]:


train[train['user_id'].isin(heavy_users_list)].groupby('user_id')['rating'].mean()


# In[36]:


train.groupby('isbn')['rating'].count().sort_values(ascending=False)


# In[37]:


users


# In[38]:


books = books.drop(['img_url', 'img_path'],axis=1)


# In[39]:


books


# In[124]:


import re
books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())


# In[125]:


books['category'].value_counts()


# In[126]:


books['category'] = books['category'].str.lower()


# In[127]:


books['category'].value_counts()


# In[128]:


category_df = pd.DataFrame(books['category'].value_counts()).reset_index()
category_df.columns = ['category','count']
category_df.head()


# In[129]:


category_df[category_df['count']>=10]


# In[130]:


books['category_high'] = books['category'].copy()
books.loc[books[books['category']=='biography'].index, 'category_high'] = 'biography autobiography'
books.loc[books[books['category']=='autobiography'].index,'category_high'] = 'biography autobiography'


# In[131]:


books[books['category'].str.contains('history', na=False)]['category'].unique()


# In[132]:


books.loc[books[books['category'].str.contains('history',na=False)].index,'category_high'] = 'history'


# In[133]:


categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
 'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
 'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
 'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

for category in categories:
    books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category


# In[134]:


category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
category_high_df.columns = ['category','count']
category_high_df.head(10)


# In[135]:


# 5개 이하인 항목은 others로 묶어주도록 하겠습니다.
others_list = category_high_df[category_high_df['count']<5]['category'].values


# In[136]:


books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'


# In[137]:


books['category'].nunique()


# In[138]:


books['category_high'].nunique()


# In[139]:


books.isna().sum()


# In[140]:


books


# In[144]:


books = books.drop(['category','img_url','img_path'],axis=1)


# In[146]:


books = books.drop(['year_of_publication'], axis=1)
books


# In[147]:


books.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\books1.csv")


# In[57]:


merge1 = train.merge(books, how='left', on='isbn')
data = merge1.merge(users, how='inner', on='user_id')
print('merge 결과 shape: ', data.shape)


# In[58]:


data


# In[12]:


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


# In[13]:


data


# In[14]:


import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

books = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\books.csv')
users = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\users.csv')
train = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\train_ratings.csv')
test = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\test_ratings.csv')
sub = pd.read_csv(r'C:\Users\wjdgh\workspace\프로젝트1 데이터\sample_submission.csv')


# In[15]:


train


# In[59]:


data


# In[60]:


data['langeage'] = data['language'].fillna('Unknown')


# In[61]:


data


# In[148]:


train


# In[150]:


books


# In[151]:


users


# In[152]:


train


# In[154]:


f_data = train.merge(books, how='left', on='isbn')
f_data


# In[156]:


f_data = f_data.merge(users, how='left', on='user_id')
f_data


# In[157]:


f_data = f_data.drop(['age_y','location_country_y'],axis = 1)
f_data


# In[160]:


f_data = f_data.fillna('Unknown')


# In[161]:


f_data


# In[212]:


f_data.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\train_xgb.csv")


# In[211]:


f_data


# In[163]:


test


# In[164]:


test = test.merge(books, how='left',on='isbn')
test


# In[165]:


test = test.merge(users, how='left',on='user_id')
test


# In[166]:


test.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\test_xgb.csv")


# In[168]:


f_data


# In[174]:


f_data = f_data.rename(columns={'location_country_x':'location_country'})


# In[209]:


f_data


# In[180]:


test = test[['user_id','isbn','book_title','book_author','publisher','language','summary','category_high','age','location_country','rating']]


# In[178]:


f_data


# In[183]:


test.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\test_xgb.csv")


# In[184]:


df = pd.read_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\train_xgb.csv")


# In[185]:


df


# In[187]:


df = df.rename(columns={'age_x':'age'})


# In[210]:


df


# In[189]:


df = df[['user_id','isbn','book_title','book_author','publisher','language','summary','category_high','age','location_country','rating']]


# In[208]:


df


# In[206]:


df.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\train_xgb.csv")


# In[193]:


len(df['book_title'].unique())


# In[ ]:


ids = pd.concat([df['user_id']])


# In[ ]:


user2idx = {id:idx for idx, id in idx2user.items()}


# In[194]:


from sklearn.preprocessing import LabelEncoder


# In[196]:


le = LabelEncoder()


# In[207]:


df


# In[198]:


le.fit(df['book_title'])


# In[200]:


df['book_title'] = le.transform(df['book_title'])


# In[201]:


df


# In[202]:


df.columns


# In[203]:


arr = ['book_title','book_author', 'publisher', 'language','summary', 'category_high', 'location_country']


# In[204]:


arr = ['book_title','book_author', 'publisher', 'language','summary', 'category_high', 'location_country']
for i in arr:
    le.fit(df[i])
    df[i] = le.transform(df[i])


# In[205]:


df


# In[226]:


a = train.merge(books, how='left',on='isbn')


# In[227]:


a


# In[228]:


a = a.merge(users, how='left',on='user_id')
a


# In[264]:


a


# In[229]:


a['age'] = a['age'].fillna(100)


# In[231]:


a['location_country'] = a['location_country'].fillna('Unknown')


# In[232]:


a


# In[233]:


a = a[['user_id','isbn','book_title','book_author','publisher','language','summary','category_high','age','location_country','rating']]
a


# In[234]:


a.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\train_xgb.csv")


# In[236]:


test['age'] = test['age'].fillna(100)


# In[238]:


test['location_country'] = test['location_country'].fillna('Unknown')


# In[240]:


test.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\test_xgb.csv")


# In[241]:


test


# In[247]:


arr = ['book_title','book_author', 'publisher', 'language','summary', 'category_high', 'location_country']
for i in arr:
    le.fit(test[i])
    test[i] = le.transform(test[i])


# In[246]:


a


# In[248]:


test


# In[262]:


a.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\train_xgb.csv")


# In[263]:


test.to_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\test_xgb.csv")


# In[258]:


le.fit(test['isbn'])
test['isbn'] = le.transform(test['isbn'])


# In[260]:


test


# In[261]:


a


# In[8]:


train = pd.read_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\train_xgb.csv")
test = pd.read_csv(r"C:\Users\wjdgh\OneDrive\LEEJUNGHO\test_xgb.csv")


# In[3]:


train


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


fig, ax = plt.subplots(1,2, figsize=(20,7))

train['age'].hist(bins=40, color='teal', ax=ax[0])
sns.boxenplot(data=train, x='age', color='teal', ax=ax[1])
plt.show()


# In[11]:


users


# In[2]:


import seaborn as sb


# In[ ]:


sb.heatmap(data = tra)


# In[5]:


a = train.merge(users, how='left',on='user_id')


# In[6]:


a = a.merge(books, how='left',on='isbn')
a


# In[ ]:


sb.heatmap(train, annot=True, fmt='.2f')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




