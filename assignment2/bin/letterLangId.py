# -*- coding: latin-1 -*-
'''
Author: Yifeng Chu (ychu26)

These are functions for letter bigram language identification
some details: 
    space and punctuations are preserved when training the model.
'''
import os
import numpy as np
import codecs
import collections
import pandas as pd
import itertools

def get_data():
    '''get the preprocessed training data and testing data and store 
    them in the dictionary'''
    workdir = get_workdir()
    join_path = os.path.join
    data_path = join_path(workdir, 'database')
    testin_path = join_path(workdir, 'input') 
    data = {}
    data['test'] = readtxt(join_path(testin_path,'LangId.test'))
    data['train'] = {}
    data['train']['English'] = preprocess(readtxt(join_path(data_path,'LangId.train.English')))
    data['train']['French'] = preprocess(readtxt(join_path(data_path,'LangId.train.French')))
    data['train']['Italian'] = preprocess(readtxt(join_path(data_path,'LangId.train.Italian')))
    return data

def get_workdir():
    bin_path = os.getcwd()
    return  os.path.dirname(bin_path)

def readtxt(path):
    f = codecs.open(path, encoding='latin-1', mode='r')
    #'≤≥' are usd as start and end symbol for every sentence
    data = [(u'≤'+line).replace(u'\n', u'≥') for line in f.readlines()]
    f.close()
    return data

def preprocess(sentences):
    #sentences is a list containing multiple string
    s = np.array(list(''.join(sentences)))
    ss_ = bicharconc(s)
    return {'s': s, 'ss_': ss_} 

def bicharconc(s):
    #s_ is the shifted array of s by one unit
    s_ = np.append(s[1:], s[0])
    #element wise addition of two array, get bigram letter array now
    ss_ = np.core.defchararray.add(s, s_)
    #since the combintation '≥≤' is useless
    return np.delete(ss_, np.where(ss_==u'≥≤'))

def train(train_data):
    """get the model.
    since every element cc_ in ss_, c appears first, then c_,
    P(c_|c) = P(cc_) / P(c) = #(cc_) / #(c)
    """
    model = pd.DataFrame()
    #fetch all bichars and remove the redundancy
    sskeys = pd.DataFrame(train_data).loc['ss_',:].values
    sskeys =  list(set(itertools.chain(*map(list, sskeys))))
    #fill the table with probs
    for dkey in train_data.keys():
        s, ss_ = train_data[dkey]['s'], train_data[dkey]['ss_']
        s_cont, ss_cont = map(collections.Counter, [s, ss_])
        for sskey in sskeys: 
            model.loc[sskey, dkey] = smooth(ss_cont[sskey], s_cont[sskey[0]], len(s_cont.keys()))
    return model

def smooth(joint, marg, n_letter):
    'smooth function'
    #joint represents joint prob, marg represents marginal prob
    return laplace(joint, marg, n_letter)

def laplace(joint, marg, n_letter):
    return float(joint+1) / (marg+n_letter)

def predict(test_data, model):
    'predict the language class using trained model'
    predicted_values = []
    for i in xrange(len(test_data)):
        #same technique for preprocessing the training data, concat the neighbor chars
        ss_ = bicharconc(list(test_data[i]))   
        #to avoid the underflow, use log of prob
        logp_sum = np.zeros(3,)
        for bichar in ss_:
            if bichar in model.index.values:
                logp_sum += np.log(model.loc[bichar, :].values)
        predicted_values.append(model.columns.values[logp_sum.argmax()])

    return np.array(predicted_values)

def write_solution(predicted_values):
    idx = np.arange(1, predicted_values.size+1)
    out = np.array(zip(idx, predicted_values))
    workdir = get_workdir()
    out_path = os.path.join(workdir, 'output', 'letterLangId.out')
    np.savetxt(out_path, out, fmt='%s', delimiter=' ')
    print "Refer to output file under work directory for solution\n"

def evaluate(predicted_values):
    'calculate the accuracy'
    workdir = get_workdir()
    #read the test solution
    sol_path = os.path.join(workdir, 'solution')
    groundtruth_values = np.loadtxt(os.path.join(sol_path, 'LangId.sol'), dtype='S', delimiter=' ', usecols=[-1])
    return (groundtruth_values == predicted_values).sum() / float(groundtruth_values.size)



