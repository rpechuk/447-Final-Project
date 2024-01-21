# Python Imports
import io
import os
import re
import pickle
import csv
import logging
import abc

from collections import defaultdict, namedtuple
from datetime import datetime

# Scientific Libraries
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy.stats import norm
from scipy.optimize import minimize

# NLP toolkits
from nltk.corpus import stopwords as sw
from gensim.utils import simple_preprocess

from CatGO.categorize import Categorizer
from tqdm import trange

# Deep Learning
import torch

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from torch.utils.data import DataLoader
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator

# Dataset Specific
from GSD.GSD_util import GSD_Definition, GSD_Word

# Helper functions

punctuations = '!"#$%&()\*\+,-\./:;<=>?@[\\]^_`{|}~'

re_punc = re.compile(r"["+punctuations+r"]+")
re_space = re.compile(r" +")

stopwords = set(sw.words('english'))

Definition = namedtuple('Definition', ['word', 'type', 'def_sent', 'meta_data'])
# For slang data entries
SlangEntry = namedtuple('SlangEntry', ['word', 'def_sent', 'meta_data'])
DataIndex = namedtuple('DataIndex', ['train', 'dev', 'test'])
Triplet = namedtuple('Triplet', ['anchor', 'positive', 'negative'])

def tokenize(sentence):
    return re.compile(r"(?:^|(?<=\s))\S+(?=\s|$)").findall(sentence)

def processTokens(fun, sentence):
    return re.compile(r"(?:^|(?<=\s))\S+(?=\s|$)").sub(fun, sentence)

def normalize(array, axis=1):
    denoms = np.sum(array, axis=axis)
    if axis == 1:
        return array / denoms[:,np.newaxis]
    if axis == 0:
        return array / denoms[np.newaxis, :]
    
def normalize_L2(array, axis=1):
    if axis == 1:
        return array / np.linalg.norm(array, axis=1)[:, np.newaxis]
    if axis == 0:
        return array / np.linalg.norm(array, axis=0)[np.newaxis, :]
    
def acronym_check(entry):
    if 'acronym' in entry.def_sent:
        return True
    for c in str(entry.word):
        if ord(c) >= 65 and ord(c) <= 90:
            continue
        return False
    return True

def is_close_def(query_sent, target_sent, threshold=0.5):
    query_s = [w for w in simple_preprocess(query_sent) if w not in stopwords]
    target_s = set([w for w in simple_preprocess(target_sent) if w not in stopwords])
    overlap_c = 0
    for word in query_s:
        if word in target_s:
            overlap_c += 1
    return overlap_c >= len(query_s) * threshold

def has_close_conv_def(word, slang_def_sent, conv_data, threshold=0.5):
    conv_sents = [d['def'] for d in conv_data[word].definitions]
    for conv_sent in conv_sents:
        if is_close_def(slang_def_sent, conv_sent, threshold):
            return True
    return False

def create_directory(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)


# For conventional data entries
class Word:
    
    def __init__(self, word):
        self.word = word
        self.pos_tags = set()
        self.definitions = []

    def attach_def(self, word_def, pos, sentences):
        new_def = {'def':word_def, 'pos':pos, 'sents':sentences}
        self.pos_tags.add(pos)
        self.definitions.append(new_def)
        
# Evaluation helpers

def get_rankings(l_model, inds, labels):
    N = l_model.shape[0]
    ranks = np.zeros(l_model.shape, dtype=np.int32)
    rankings = np.zeros(N, dtype=np.int32)
        
    for i in range(N):
        ranks[i] = np.argsort(l_model[i])[::-1]
        rankings[i] = ranks[i].tolist().index(labels[inds[i]])+1
            
    return rankings
    
def get_roc(rankings, N_cat):
    roc = np.zeros(N_cat+1)
    for rank in rankings:
        roc[rank]+=1
    for i in range(1,N_cat+1):
        roc[i] = roc[i] + roc[i-1]
    return roc / rankings.shape[0]

class ConvDataset:
    
    def __init__(self, data_path):
        self.entries, self.vocab = self.load_data(data_path)
        
        self.N_total = 0
        for e in self.entries:
            self.N_total += len(e.definitions)
        self.V = len(self.vocab)
        
        self.data = {d.word:d for d in self.entries}
    
    @abc.abstractmethod
    def load_data(self, data_path):
        raise NotImplementedError()
        
    def __str__(self):
        out = ""
        out += "Dataset Name: " + "\n"
        out += "Total Definition Entries: %d" % self.N_total + "\n"
        out += "Vocab Size: %d" % self.V + "\n"
        return out
    
class OED_Dataset(ConvDataset):
    
    def load_data(self, data_path):
        data_OED = np.load(data_path, allow_pickle=True)
        vocab_OED = set([w.word for w in data_OED])
        return data_OED, vocab_OED
    
class WN_Dataset(ConvDataset):
    
    def load_data(self, data_path):
        data_WN = np.load(data_path, allow_pickle=True)
        vocab_WN = set([w.word for w in data_WN])
        return data_WN, vocab_WN

class SlangDataset:
    
    def __init__(self, slang_path, conv_dataset):
        
        self.meta_set = set()
        
        self.slang_data, self.conv_data = self.process_entries(slang_path, conv_dataset)
        
        self.N_total = len(self.slang_data)
        
        vocab = []
        vocab_set = set()
        for d in self.slang_data:
            word = str(d.word)
            if word not in vocab_set:
                vocab.append(word)
                vocab_set.add(word)
        self.vocab = np.asarray(vocab)
        self.word2id = {self.vocab[i]:i for i in range(len(self.vocab))}
        self.vocab_ids = np.asarray([self.word2id[self.slang_data[i].word] for i in range(self.N_total)], dtype=np.int64)
        
        self.V = len(self.vocab)
        
    def has_meta(self, attribute):
        return attribute in self.meta_set
        
    @abc.abstractmethod
    def load_data(self, slang_path, conv_dataset):
        raise NotImplementedError()
        
    def process_entries(self, slang_path, conv_dataset):
        slang_entries = self.load_data(slang_path, conv_dataset)
        
        conv_data = conv_dataset.data
        
        slang_data = [d for d in slang_entries if not acronym_check(d)]
        slang_data = [d for d in slang_data if not has_close_conv_def(str(d.word), d.def_sent, conv_data)]
        
        return slang_data, conv_data
        
    def __str__(self):
        out = ""
        out += "Dataset Name: " + "\n"
        out += "Total Definition Entries: %d" % self.N_total + "\n"
        out += "Vocab Size: %d" % self.V + "\n"
        return out
    
class OSD_Dataset(SlangDataset):
    
    def __init__(self, slang_path, conv_dataset):
        super().__init__(slang_path, conv_dataset)
        self.meta_set.add('pos')
        self.meta_set.add('context')
        
    def load_data(self, slang_path, conv_dataset):
        
        data_OSD_raw = np.load(slang_path, allow_pickle=True)
        
        def process_def(d):
            return SlangEntry(d.word, d.def_sent, {'pos':d.type, 'context':d.meta_data})
        
        data_OSD = [process_def(d) for d in data_OSD_raw if str(d.word) in conv_dataset.vocab]
        
        return data_OSD
    
class GDS_Dataset(SlangDataset):
    
    def __init__(self, slang_path, conv_dataset):
        super().__init__(slang_path, conv_dataset)
        
        self.dates = [entry.meta_data['dates'] for entry in self.slang_data]
        
        self.meta_set.add('pos')
        self.meta_set.add('dates')
        
    def load_data(self, slang_path, conv_dataset):
        
        data_GDS_raw = np.load(slang_path, allow_pickle=True)
        
        re_hex = re.compile(r"\\x[a-f0-9][a-f0-9]")
        re_spacechar = re.compile(r"\\(n|t)")
        
        def get_date_tag(definition):
            return np.min([stamp[0] for stamp in definition.stamps])

        def proc_def(sent):
            return re_spacechar.sub('', re_hex.sub('', sent))
        
        data_GDS_raw = [d for d in data_GDS_raw if str(d.word) in conv_dataset.vocab]
        
        data_GDS = []
        for entry in data_GDS_raw:
            for d in entry.definitions:
                data_GDS.append(SlangEntry(entry.word, proc_def(d.def_sent), {'pos':entry.pos, 'dates': get_date_tag(d)}))
        
        return data_GDS
    
class Urban_Dataset(SlangDataset):
    
    def __init__(self, slang_path, conv_dataset):
        super().__init__(slang_path, conv_dataset)
        
    def load_data(self, slang_path, conv_dataset):
        
        data_Urban_raw = np.load(slang_path, allow_pickle=True)
        
        re_hex = re.compile(r"\\x[a-f0-9][a-f0-9]")
        re_spacechar = re.compile(r"\\(n|t)")
        
        def process_def(d):
            return SlangEntry(d[0], re_spacechar.sub('', re_hex.sub('', d[1])), {})
        
        data_Urban = [process_def(d) for d in data_Urban_raw if d[0] in conv_dataset.vocab]
        
        return data_Urban
        
class WordEncoder:
    
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def embed_word(self, word):
        raise NotImplementedError()
        
    def norm_embed(self, word):
        vec = self.embed_word(word)
        return vec / np.linalg.norm(vec)
    
class FTEncoder(WordEncoder):
    
    def __init__(self, embed_file_name):
        fin = io.open(embed_file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        self.embeddings = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            self.embeddings[tokens[0]] = np.asarray(tokens[1:], dtype=np.float)
        
        self.vocab = set(self.embeddings.keys())
        self.E = self.embeddings[list(self.embeddings.keys())[0]].shape[0]
        
        self.cache = set()
    
    def embed_word(self, word):
        self.cache.add(word)
        return self.embeddings[word]
    
    def cache_embed(self, path):
        output = {}
        for word in self.cache:
            output[word] = self.embeddings[word]
        pickle.dump(output, open(path, 'wb'))
        
    def clear_cache(self):
        self.cache = set()
        
class FTCachedEncoder(WordEncoder):
    
    def __init__(self, embed_file_name):
        self.embeddings = pickle.load(open(embed_file_name, 'rb'))
        
        self.vocab = set(self.embeddings.keys())
        self.E = self.embeddings[list(self.embeddings.keys())[0]].shape[0]
        
    def embed_word(self, word):
        return self.embeddings[word]

class SenseEncoder:
    
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()
    
    def encode_dataset(self, dataset, slang_ind):
        
        embeds = {}
        
        def collect_slang_sents(dataset, ind):
            sentences = []
            for i in ind:
                sentences.append(' '.join(simple_preprocess(dataset.slang_data[i].def_sent)))
            return sentences
        
        embeds['train'] = self.encode_sentences(collect_slang_sents(dataset, slang_ind.train))
        embeds['dev'] = self.encode_sentences(collect_slang_sents(dataset, slang_ind.dev))
        embeds['test'] = self.encode_sentences(collect_slang_sents(dataset, slang_ind.test))

        sentences = []
        for i in range(len(dataset.vocab)):
            word = dataset.vocab[i]
            for d in dataset.conv_data[word].definitions:
                sentences.append(' '.join(simple_preprocess(d['def'])))
          
        embeds['standard'] = self.encode_sentences(sentences)
        
        return embeds
    
    @abc.abstractmethod
    def encode_sentences(self, sentences):
        raise NotImplementedError()
        
class SBertEncoder(SenseEncoder):
    
    def __init__(self, sbert_model_name=None, name=None):
        
        if sbert_model_name is None:
            sbert_model_name = 'bert-base-nli-mean-tokens'
            self.name = 'sbert_base'
        elif name is not None:
            self.name = name
        else:
            self.name = sbert_model_name
            
        self.sbert_model = SentenceTransformer(sbert_model_name)
        
    def encode_sentences(self, sentences):
        
        sbert_embeddings = np.asarray(self.sbert_model.encode(sentences))
        return normalize_L2(sbert_embeddings, axis=1)   

class SlangGenTrainer:
    
    MAX_NEIGHBOR = 300
    
    def __init__(self, dataset, word_encoder, out_dir='', verbose=False):
        
        self.out_dir = out_dir
        create_directory(out_dir)
        
        self.dataset = dataset
        
        self.word_encoder = word_encoder
        
        self.verbose = verbose
            
        conv_lens = []
        for i in range(dataset.V):
            word = dataset.vocab[i]
            conv_lens.append(len(dataset.conv_data[word].definitions))
        self.conv_lens = np.asarray(conv_lens)

        self.conv_acc = np.zeros(dataset.V, dtype=np.int32)

        for i in range(1,dataset.V):
            self.conv_acc[i] = self.conv_acc[i-1] + self.conv_lens[i-1]
            
        self.word_dist = self.preprocess_word_dist()
        np.save(out_dir+'/word_dist.npy', self.word_dist)

    def preprocess_slang_data(self, slang_ind, fold_name='default', skip_steps=[]):

        out_dir = self.out_dir + '/' + fold_name
        create_directory(out_dir)
        out_dir += '/'
        
        # Generate contrastive pairs for training
        if 'contrastive' not in skip_steps:
            if self.verbose:
                print("Generating contrative pairs...")
            contrastive_pairs_train, contrastive_pairs_dev = self.preprocess_contrastive(slang_ind)
            np.save(out_dir+'contrastive_train.npy', contrastive_pairs_train)
            np.save(out_dir+'contrastive_dev.npy', contrastive_pairs_dev)
            if self.verbose:
                print("Complete!")
                
    def load_preprocessed_data(self, fold_name='default', skip_steps=[]):
        
        out_dir = self.out_dir + '/' + fold_name + '/'
        
        preproc_data = {}
        
        if 'contrastive' not in skip_steps:
            preproc_data['cp_train'] = np.load(out_dir+'contrastive_train.npy', allow_pickle=True)
            preproc_data['cp_dev'] = np.load(out_dir+'contrastive_dev.npy', allow_pickle=True)
            
        return preproc_data
    
    def get_trained_embeddings(self, slang_ind, fold_name='default', model_path='SBERT_contrastive'):
        
        model_name = self.out_dir + '/' + fold_name + '/SBERT_data/' + model_path
        sense_encoder = SBertEncoder(sbert_model_name=model_name, name=model_path)
        
        return self.get_sense_embeddings(slang_ind, sense_encoder, fold_name)
        
    def get_sense_embeddings(self, slang_ind, sense_encoder, fold_name='default'):
                    
        if self.verbose:
            print("Encoding sense definitions...")
            
        out_dir = self.out_dir + '/' + fold_name + '/'
            
        sense_embeds = sense_encoder.encode_dataset(self.dataset, slang_ind)
        np.savez(out_dir+"sum_embed_"+sense_encoder.name+".npz", train=sense_embeds['train'], dev=sense_embeds['dev'], test=sense_embeds['test'], standard=sense_embeds['standard'])
        
        if self.verbose:
            print("Complete!")
            
        return sense_embeds
        
    def train_contrastive_model(self, slang_ind, params=None, fold_name='default'):
        
        if params is None:
            params = {'train_batch_size':16, 'num_epochs':4, 'triplet_margin':1, 'outpath':'SBERT_contrastive'}
        
        self.prep_contrastive_training(slang_ind, fold_name=fold_name)
        
        out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'

        triplet_reader = TripletReader(out_dir, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',', has_header=True)
        output_path = out_dir+params['outpath']
        
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        
        train_data = SentencesDataset(examples=triplet_reader.get_examples('contrastive_train.csv'), model=sbert_model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=params['train_batch_size'])
        train_loss = losses.TripletLoss(model=sbert_model, triplet_margin=params['triplet_margin'])

        dev_data = SentencesDataset(examples=triplet_reader.get_examples('contrastive_dev.csv'), model=sbert_model)
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=params['train_batch_size'])
        evaluator = TripletEvaluator(dev_dataloader)
        
        warmup_steps = int(len(train_data)*params['num_epochs']/params['train_batch_size']*0.1) #10% of train data

        # Train the model
        sbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=params['num_epochs'],
                  evaluation_steps=len(dev_data),
                  warmup_steps=warmup_steps,
                  output_path=output_path)
    
    def prep_contrastive_training(self, slang_ind, fold_name='default'):
        
        if self.verbose:
            print("Generating triplet data for contrastive training...")
        
        out_dir = self.out_dir + '/' + fold_name + '/SBERT_data/'
        create_directory(out_dir)
        
        preproc_data = self.load_preprocessed_data(fold_name=fold_name)
        
        N_train, triplets = self.sample_triplets(preproc_data['cp_train'])
        N_dev, triplets_dev = self.sample_triplets(preproc_data['cp_dev'])
        
        np.save(out_dir+'triplets.npy', triplets)
        np.save(out_dir+'triplets_dev.npy', triplets_dev)
            
        slang_def_sents = []
        for i in range(self.dataset.N_total):
            slang_def_sents.append(' '.join(simple_preprocess(self.dataset.slang_data[i].def_sent)))

        conv_def_sents = []
        for i in range(self.dataset.V):
            word = self.dataset.vocab[i]
            for d in self.dataset.conv_data[word].definitions:
                conv_def_sents.append(' '.join(simple_preprocess(d['def'])))
        
        data_train = {'anchor':[slang_def_sents[slang_ind.train[triplets[i][0]]] for i in range(N_train)],\
                      'positive':[conv_def_sents[triplets[i][1]] for i in range(N_train)],\
                      'negative':[conv_def_sents[triplets[i][2]] for i in range(N_train)]}

        data_dev = {'anchor':[slang_def_sents[slang_ind.dev[triplets_dev[i][0]]] for i in range(N_dev)],\
                    'positive':[conv_def_sents[triplets_dev[i][1]] for i in range(N_dev)],\
                    'negative':[conv_def_sents[triplets_dev[i][2]] for i in range(N_dev)]}
        
        df_train = pd.DataFrame(data=data_train)
        df_dev = pd.DataFrame(data=data_dev)
        
        df_train.to_csv(out_dir+'contrastive_train.csv', index=False)
        df_dev.to_csv(out_dir+'contrastive_dev.csv', index=False)
        
        if self.verbose:
            print("Complete!")
        
    def sample_triplets(self, contrast_data):
    
        # Maximum number of positive pairs from the same positive definition
        MAX_PER_POSDEF = 1000
    
        triplets = []

        N_def = contrast_data.shape[0]

        for i in range(N_def):
            anchor = i
            if contrast_data[i]['negative'].shape[0] == 0:
                continue
            pre_pos = -100
            num_d = 0
            #for positive in contrast_data[i]['positive']:
            for positive in np.concatenate([contrast_data[i]['positive'], contrast_data[i]['neighbors']]):
                if positive != pre_pos+1:
                    num_d = MAX_PER_POSDEF
                pre_pos = positive
                if num_d > 0:
                    num_d -= 1

                    negative = np.random.choice(contrast_data[i]['negative'])
                    triplets.append(Triplet(anchor, positive, negative))

        N_triplets = len(triplets)

        if self.verbose:
            print("Sampled %d Triplets" % N_triplets)

        return N_triplets, np.asarray(triplets)
    
    def preprocess_word_dist(self):
    
        vocab_conv_embeds = np.zeros((self.dataset.V, self.word_encoder.E))

        for i in range(self.dataset.V):
            if self.dataset.vocab[i] in self.word_encoder.vocab:
                vocab_conv_embeds[i,:] = self.word_encoder.norm_embed(self.dataset.vocab[i])
            else:
                c_words = self.dataset.vocab[i].split(' ')
                count = 0
                if len(c_words) > 1:
                    embed = np.zeros(self.word_encoder.E)
                    for w in c_words:
                        if w in self.word_encoder.vocab:
                            embed = embed + self.word_encoder.norm_embed(w)
                            count += 1
                    if count > 0:
                        vocab_conv_embeds[i,:] = embed / float(count)

                if count == 0:
                    vocab_conv_embeds[i,:] = self.word_encoder.norm_embed('unk')

        return dist.squareform(dist.pdist(vocab_conv_embeds, metric='cosine'))

    def preprocess_contrastive(self, slang_ind):
        
        Neigh_pivot = int(np.ceil(self.dataset.V/5.0))
        N_neighbor = min(self.MAX_NEIGHBOR, self.dataset.V - Neigh_pivot)

        self.neighbors = np.zeros((self.dataset.V, N_neighbor), dtype=np.int32)
        self.neighbors_close = np.zeros((self.dataset.V, 5), dtype=np.int32)
        for i in range(self.dataset.V):
            self.neighbors[i,:] = np.argsort(self.word_dist[i,:])[max(Neigh_pivot, self.dataset.V-self.MAX_NEIGHBOR):]
            self.neighbors_close[i,:] = np.argsort(self.word_dist[i,:])[1:6]
            
        contrastive_pairs_train = self.compute_contrastive(slang_ind.train)
        contrastive_pairs_dev = self.compute_contrastive(slang_ind.dev)
        
        return contrastive_pairs_train, contrastive_pairs_dev
            
    def compute_contrastive(self, ind):
        
        def get_conv_definds(word_ind):
            return [self.conv_acc[word_ind]+j for j in range(self.conv_lens[word_ind])]
        
        contrast_data = np.empty(ind.shape[0], dtype=object)

        for i in trange(ind.shape[0]):
            word_ind = self.dataset.vocab_ids[ind[i]]
            contrast_data[i] = {}

            positives = [self.conv_acc[word_ind]+j for j in range(self.conv_lens[word_ind])]

            negatives = []
            conv_self = [d['def'] for d in self.dataset.conv_data[self.dataset.vocab[word_ind]].definitions]
            for far_word in self.neighbors[word_ind]:
                conv_defs = [d['def'] for d in self.dataset.conv_data[self.dataset.vocab[far_word]].definitions]
                for j in range(self.conv_lens[far_word]):
                    cand = self.conv_acc[far_word] + j
                    if not is_close_def(self.dataset.slang_data[ind[i]].def_sent, conv_defs[j], threshold=0.2):
                        has_close_cf_def = False
                        for self_def in conv_self:
                            if is_close_def(self_def, conv_defs[j], threshold=0.2):
                                has_close_cf_def = True
                                break
                        if not has_close_cf_def:
                            negatives.append(cand)            
            
            neigh_defs = []
            for close_word in self.neighbors_close[word_ind]:
                neigh_defs.extend([self.conv_acc[close_word]+j for j in range(self.conv_lens[close_word])])

            contrast_data[i]['positive'] = np.asarray(positives)
            contrast_data[i]['negative'] = np.asarray(negatives)
            contrast_data[i]['neighbors'] = np.asarray(neigh_defs)

        return contrast_data
    
class SlangGenModel:
    
    DEFAULT_PARAMS = {'embed_name':'SBERT_contrastive', 'out_name':'predictions', 'model':'cf_prototype_5', 'prior_name':'uniform', 'contr_params':None}
    
    def __init__(self, trainer, data_dir='', verbose=False):
        
        self.trainer = trainer
        self.data_dir = data_dir
        self.verbose=verbose
        
        self.vocab = trainer.dataset.vocab
        self.labels = trainer.dataset.vocab_ids
        
        self.cf_feats = np.stack([self.trainer.word_dist], axis=0)
        
    def train_contrastive(self, slang_ind, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
            
        self.trainer.preprocess_slang_data(slang_ind, fold_name=fold_name)
        self.trainer.train_contrastive_model(slang_ind, fold_name=fold_name, params=params['contr_params'])
        self.trainer.get_trained_embeddings(slang_ind, fold_name=fold_name)
        
    def run_categorization(self, slang_ind, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
        
        data_dir = self.data_dir + '/' + fold_name + '/'
        
        def_embeds = np.load(data_dir+'sum_embed_'+params['embed_name']+'.npz')
        E = def_embeds['train'].shape[1]
        
        conv_embed = def_embeds['standard']
        
        vocab_embeds = []
        c = 0
        for i in range(self.trainer.dataset.V):
            num_def = len(self.trainer.dataset.conv_data[self.vocab[i]].definitions)
            embed = np.zeros((num_def, E))
            for j in range(num_def):        
                embed[j,:] = conv_embed[c,:]
                c += 1
            vocab_embeds.append(embed)
            
        slang_def_embeds = np.concatenate([def_embeds['train'], def_embeds['dev'], def_embeds['test']])
        
        labels = self.labels[np.concatenate(slang_ind)]
        
        categorizer = Categorizer(self.vocab, vocab_embeds, slang_def_embeds, labels, self.cf_feats)
        
        N_train_dev = labels.shape[0] - slang_ind.test.shape[0]
        categorizer.set_inds(np.arange(N_train_dev), np.arange(N_train_dev, labels.shape[0]))
        
        model_dir = data_dir+params['out_name']+'/'
        create_directory(model_dir)
        categorizer.set_datadir(model_dir)
        
        if params['prior_name'] != 'uniform':
            categorizer.add_prior(params['prior_name'], params['prior'][np.concatenate(slang_ind)])
        
        categorizer.run_categorization(models=[params['model']], prior=params['prior_name'], verbose=self.verbose)
        
        return categorizer
    
    def get_results(self, slang_ind, fold_name='default', params=None):
            
        if params is None:
            params = self.DEFAULT_PARAMS
        
        return np.load(self.data_dir + '/' + fold_name + '/'+params['out_name']+'/'+'l_'+params['model']+'_'+params['prior_name']+'_test.npy')
    

class SlangGenTypeModel(SlangGenModel):
    
    def train_contrastive(self, slang_ind, types, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
            
        N_types = self.get_N_types(types)
        type_inds = self.compute_type_inds(slang_ind, types)
        
        if self.verbose:
            print([type_inds[i][0].shape for i in range(N_types)])
            print([type_inds[i][1].shape for i in range(N_types)])
            print([type_inds[i][2].shape for i in range(N_types)])
            
        for i in range(N_types):
            s_ind = type_inds[i]
            f_name = fold_name+'_'+str(i+1)

            self.trainer.preprocess_slang_data(s_ind, fold_name=f_name)
            self.trainer.train_contrastive_model(s_ind, fold_name=f_name, params=params['contr_params'])
            self.trainer.get_trained_embeddings(s_ind, fold_name=f_name)
        
    def run_categorization(self, slang_ind, types, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
            
        N_types = self.get_N_types(types)
        type_inds = self.compute_type_inds(slang_ind, types)
        
        model = SlangGenModel(trainer, data_dir=data_dir)  
    
        for i in range(N_types):
            s_ind = type_inds[i]
            f_name = fold_name+'_'+str(i+1)

            model.run_categorization(s_ind, fold_name=f_name, params=params)
        
    def get_results(self, slang_ind, types, fold_name='default', params=None):
        
        if params is None:
            params = self.DEFAULT_PARAMS
        
        N_types = self.get_N_types(types)
        type_inds = self.compute_type_inds(slang_ind, types)
        
        l_table = np.zeros((slang_ind.test.shape[0], dataset.V), dtype=np.float32)
        
        for i in range(N_types):
            table_i = np.load(self.data_dir + '/' + fold_name + '_' + str(i+1) + '/'+params['out_name']+'/'+'l_'+params['model']+'_'+params['prior_name']+'_test.npy')
            
            type_mask = types[slang_ind.test]==i
            l_table[type_mask] = table_i[type_mask]
        
        return l_table
    
    def get_N_types(self, types):
        return np.max(types)+1
    
    def compute_type_inds(self, slang_ind, types):
        
        N_types = self.get_N_types(types)
        
        type_ind = [[[],[],[]] for i in range(N_types)]
    
        for i in slang_ind.train:
            type_ind[types[i]][0].append(i)

        for i in slang_ind.dev:
            type_ind[types[i]][1].append(i)

        for i in slang_ind.test:
            for j in range(N_types):
                type_ind[j][2].append(i)

        return [DataIndex(*map(np.asarray, type_ind[i])) for i in range(N_types)]
    
