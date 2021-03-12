#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/1/13 13:49                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pickle

articles_df = pd.read_csv('data/shared_articles.csv')
interactions_df = pd.read_csv('data/users_interactions.csv')
articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)

articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.drop('eventType', axis=1, inplace=True)
df = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on = 'contentId')
df.head()

event_type_strength = {
    'VIEW': 1.0,
    'LIKE': 2.0,
    'BOOKMARK': 3.0,
    'FOLLOW': 4.0,
    'COMMENT CREATED': 5.0,
}

df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])
df.sample(10)

df = df.drop_duplicates()
grouped_df = df.groupby(['personId', 'contentId', 'title']).sum().reset_index()
grouped_df.sample(10)

grouped_df['title'] = grouped_df['title'].astype("category")
grouped_df['personId'] = grouped_df['personId'].astype("category")
grouped_df['contentId'] = grouped_df['contentId'].astype("category")
grouped_df['person_id'] = grouped_df['personId'].cat.codes
grouped_df['content_id'] = grouped_df['contentId'].cat.codes
grouped_df.sample(10)

sparse_content_person = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
sparse_person_content = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))

print(sparse_content_person.shape)
print(sparse_person_content.shape)

# np.save('data/sparse_content_person',sparse_content_person)# 应该是这个类型，np.save方法出错
# np.save('data/sparse_person_content',sparse_person_content)
# d = np.load('data/sparse_content_person.npy')
# print(type(d),d.shape)

with open('data/sparse_content_person.pickle','wb') as f:
    pickle.dump(sparse_content_person,f)
with open('data/sparse_person_content.pickle','wb') as f:
    pickle.dump(sparse_person_content,f)

data = pickle.load(open('data/sparse_content_person.pickle','rb'))
print(type(data),data.shape)

alpha = 15
data = (sparse_content_person * alpha).astype('double')

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(data)


content_id=235
n_similar=10
print('AAAAAAAAA')
print(grouped_df.title.loc[grouped_df.content_id==content_id])
#获取用户矩阵
person_vecs = model.user_factors
#获取内容矩阵
content_vecs = model.item_factors
#计算内容的向量的范数
content_norms = np.sqrt((content_vecs * content_vecs).sum(axis=1))
#计算指定的content_id 与其他所有文章的相似度
scores = content_vecs.dot(content_vecs[content_id]) / content_norms
#获取相似度最大的10篇文章
top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
#组成content_id和title的元组
similar = sorted(zip(top_idx, scores[top_idx] / content_norms[content_id]), key=lambda x: -x[1])

print(person_vecs.shape)
print(content_vecs.shape)

print('--------------------------------------------------------')
def recommend(person_id, sparse_person_content, person_vecs, content_vecs, num_contents=10):
    #*****************得到指定用户对所有文章的评分向量******************************
    # 将该用户向量乘以内容矩阵(做点积),得到该用户对所有文章的评价分数向量
    rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray()

    #**********过滤掉用户已经评分过的文章(将其评分值置为0),因为用户已经发生过交互行为的文章不应该被推荐*******
    # 从稀疏矩阵sparse_person_content中获取指定用户对所有文章的评价分数
    person_interactions = sparse_person_content[person_id,:].toarray()
    # 为该用户的对所有文章的评价分数+1，那些没有被该用户看过(view)的文章的分数就会等于1(原来是0)
    person_interactions = person_interactions.reshape(-1) + 1
    # 将那些已经被该用户看过的文章的分数置为0
    person_interactions[person_interactions > 1] = 0
    # 将该用户的评分向量做标准化处理,将其值缩放到0到1之间。
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    # 过滤掉和该用户已经交互过的文章，这些文章的评分会和0相乘。
    recommend_vector = person_interactions * rec_vector_scaled

    #*************将余下的评分分数排序，并输出分数最大的10篇文章************************
    # 根据评分值进行排序,并获取指定数量的评分值最高的文章
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]

    # 定义两个list用于存储文章的title和推荐分数。
    titles = []
    scores = []

    for idx in content_idx:
        # 将title和分数添加到list中
        titles.append(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'title': titles, 'score': scores})

    return recommendations

# 从model中获取经过训练的用户和内容矩阵,并将它们存储为稀疏矩阵
person_vecs = sparse.csr_matrix(model.user_factors)
content_vecs = sparse.csr_matrix(model.item_factors)

# 为指定用户推荐文章。
person_id = 2
recommendations = recommend(person_id, sparse_person_content, person_vecs, content_vecs)
print(recommendations)

person_id = 1
print('----------person_id:{}----------'.format(person_id))
recommendations = recommend(person_id, sparse_person_content, person_vecs, content_vecs)
print(recommendations)

import random
def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy() # 拷贝一份评分数据当作测试集
    test_set[test_set != 0] = 1 # 将有评分数据置为1，我们要模拟成二分类数据集

    training_set = ratings.copy() # 拷贝一份评分数据当作训练集
    nonzero_inds = training_set.nonzero() # 找到有过评分(有交互行为，评分数不为0)的数据的索引。
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # 将它们组成元组并存放在list中

    random.seed(0) #设置随机数种子

    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # 获取20%的非0评价的数量
    samples = random.sample(nonzero_pairs, num_samples) # 随机从非零评价的索引对中抽样20%

    content_inds = [index[0] for index in samples] # 从样本中得到文章列(第一列)索引值
    person_inds = [index[1] for index in samples] # 从样本中得到文章列(第二列)索引值

    training_set[content_inds, person_inds] = 0 # 在训练集中将这20%的随机样本的评分值置为0
    training_set.eliminate_zeros() # 在测试集中删除这0元素

    return training_set, test_set, list(set(person_inds))


content_train, content_test, content_persons_altered = make_train(sparse_content_person, pct_test = 0.2)
print('BBBB',content_train.shape,content_test.shape)