#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:52:44 2018

@author: aram
"""

import seq2seq_wrapper
import importlib
importlib.reload(seq2seq_wrapper)
import data_preprocessing
import data_utils_1
import data_utils_2
import os

class TheBot():
    
    def __init__(self):
        self.metadata, self.idx_q, self.idx_a = data_preprocessing.load_data(PATH = os.getcwd() + '/plugins/hipchap/')

        (self.trainX, self.trainY), (self.testX, self.testY), (self.validX, self.validY) = data_utils_1.split_dataset(self.idx_q, self.idx_a)

        self.xseq_len = self.trainX.shape[-1]
        self.yseq_len = self.trainY.shape[-1]
        self.batch_size = 16
        self.vocab_twit = self.metadata['idx2w']
        self.xvocab_size = len(self.metadata['idx2w'])  
        self.yvocab_size = self.xvocab_size
        self.emb_dim = 1024
        self.idx2w, self.w2idx, self.limit = data_utils_2.get_metadata()
        
        self.model = seq2seq_wrapper.Seq2Seq(xseq_len = self.xseq_len,
                                        yseq_len = self.yseq_len,
                                        xvocab_size = self.xvocab_size,
                                        yvocab_size = self.yvocab_size,
                                        ckpt_path = './plugins/hipchap/weights',
                                        emb_dim = self.emb_dim,
                                        num_layers = 3)
        self.session = self.model.restore_last_session()
    
    def get_response(self, question):
        encoded_question = data_utils_2.encode(question, self.w2idx, self.limit['maxq'])
        answer = self.model.predict(self.session, encoded_question)[0]
        return data_utils_2.decode(answer, self.idx2w) 
    
    @classmethod
    def get_meta(self):
        import os
        cwd = os.getcwd()
        return cwd
    
    