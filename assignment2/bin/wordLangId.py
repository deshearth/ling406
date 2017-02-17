# -*- coding: latin-1 -*-
'''
Author: Yifeng Chu (ychu26)

These are functions for word bigram language identification
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
    data['test'] = readtxt(join_path(testin_path,'LangId.test'), 'test')
    data['train'] = {}
    data['train']['English'] = preprocess(readtxt(join_path(data_path,'LangId.train.English')))
    data['train']['French'] = preprocess(readtxt(join_path(data_path,'LangId.train.French')))
    data['train']['Italian'] = preprocess(readtxt(join_path(data_path,'LangId.train.Italian')))
    return data

def get_workdir():
    bin_path = os.getcwd()
    return  os.path.dirname(bin_path)

def readtxt(path, *args):
    f = codecs.open(path, encoding='latin-1', mode='r')
    #'≤≥' are usd as start and end symbol for every sentence
    data = [[u'≤'] + line.strip().split(' ') + [u'≥'] for line in f.readlines()]
    f.close()
    return data if args else data[:100]

def preprocess(sentences):
    w = np.array(list(itertools.chain(*sentences)))
    ww_ = biwordconc(w)
    return {'w': w, 'ww_': ww_}

def biwordconc(w):
    w_ = np.append(w[1:], w[0])
    # in order to separate two words, add a space between them
    ww_ = reduce(np.core.defchararray.add, [w, u' ', w_])
    return np.delete(ww_, np.where(ww_==u'≥ ≤'))
    
def train(train_data):
    """get the model.
    since every element vv_ in ww_, v appears first, then v_,
    P(v_|v) = P(vv_) / P(v) = #(vv_) / #(v)
    """
    model = pd.DataFrame()
    #fetch all bichars and remove the redundancy
    wwkeys = pd.DataFrame(train_data).loc['ww_',:].values
    wwkeys =  list(set(itertools.chain(*map(list, wwkeys))))
    #fill the table with probs
    for dkey in train_data.keys():
        w, ww_ = train_data[dkey]['w'], train_data[dkey]['ww_']
        w_cont, ww_cont = map(collections.Counter, [w, ww_])
        for wwkey in wwkeys: 
            model.loc[wwkey, dkey] = smooth(ww_cont[wwkey], w_cont[wwkey[0]], len(w_cont.keys()))
    return model

def smooth(joint, marg, n_word):
    'smooth function'
    #joint represents joint prob, marg represents marginal prob
    return laplace(joint, marg, n_word)

def laplace(joint, marg, n_word):
    return float(joint+1) / (marg+n_word)

def predict(test_data, model):
    'predict the language class using trained model'
    predicted_values = []
    for i in xrange(len(test_data)):
        #same technique for preprocessing the training data, concat the neighbor chars
        ww_ = biwordconc(test_data[i])   
        #to avoid the underflow, use log of prob
        logp_sum = np.zeros(3,)
        for biword in ww_:
            if biword in model.index.values:
                logp_sum += np.log(model.loc[biword, :].values)
        predicted_values.append(model.columns.values[logp_sum.argmax()])

    return np.array(predicted_values)

def write_solution(predicted_values):
    idx = np.arange(1, predicted_values.size+1)
    out = np.array(zip(idx, predicted_values))
    workdir = get_workdir()
    out_path = os.path.join(workdir, 'output', 'wordLangId.out')
    np.savetxt(out_path, out, fmt='%s', delimiter=' ')
    print "Refer to output file under work directory for solution\n"
