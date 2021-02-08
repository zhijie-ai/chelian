#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/28 16:30
 =================知行合一=============
 观测值为离散的隐马模型
'''
import  numpy as np
from hmmlearn import hmm

def viterbi():
    states = ['box 1', 'box 2', 'box 3']
    n_states = len(states)

    observations = ['red', 'white']
    n_observations = len(observations)

    start_probability = np.array([0.2, 0.4, 0.4])
    transition_probability = np.array([[0.5, 0.2, 0.3],
                                       [0.3, 0.5, 0.2],
                                       [0.2, 0.3, 0.5]])
    emission_probability = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    model = hmm.MultinomialHMM(n_components=n_states)
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    print('A' * 20)
    X, Z = model.sample(5, random_state=4)
    print(X)
    print(Z)

    seen = np.array([[0, 1, 0]]).T
    logprob, box = model.decode(seen, algorithm='viterbi')
    print('the ball picked:', ', '.join(map(lambda x: observations[x[0]], seen)))
    print('the hidden box', ', '.join(map(lambda x: states[x], box)))

    # 使用predict函数，结果是一样的
    box2 = model.predict(seen)
    print('the ball picked:', ', '.join(map(lambda x: observations[x[0]], seen)))
    print('the hidden box', ', '.join(map(lambda x: states[x], box)))
    print('A' * 50)

    # 概率计算问题
    print(model.score(seen))
    print(np.log(0.13022))


def learn():
    states = ["box 1", "box 2", "box3"]
    n_states = len(states)

    observations = ["red", "white"]
    n_observations = len(observations)

    model = hmm.MultinomialHMM(n_components=n_states,n_iter=20,tol=0.01)
    X = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1]])
    # model.fit(X)
    # print('model.startprob_',model.startprob_)
    # print('model.transmat_',model.transmat_)
    # print('model.emissionprob_',model.emissionprob_)
    # print('概率计算问题：',model.score(X))
    # model.fit(X)
    # print('model.startprob_', model.startprob_)
    # print('model.transmat_', model.transmat_)
    # print('model.emissionprob_', model.emissionprob_)
    # print('概率计算问题：', model.score(X))
    # model.fit(X)
    # print('model.startprob_', model.startprob_)
    # print('model.transmat_', model.transmat_)
    # print('model.emissionprob_', model.emissionprob_)
    # print('概率计算问题：', model.score(X))
    for i in range(10):
        model.fit(X)
        print('model.startprob_', model.startprob_)
        print('model.transmat_', model.transmat_)
        print('model.emissionprob_', model.emissionprob_)
        print('概率计算问题：', model.score(X))
        print('*'*20)

    X,Z = model.sample(5,random_state=4)
    print(X)
    print(Z)

if __name__ == '__main__':
    # learn()
    viterbi()
