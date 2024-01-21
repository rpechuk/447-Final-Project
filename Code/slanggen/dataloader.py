import abc

import numpy as np
import pandas as pd

from .util import *

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
    
    def __init__(self, slang_path, conv_dataset, load_oov=False):
        
        self.meta_set = set()
        
        self.load_oov = load_oov
        
        if not self.load_oov:
            self.slang_data, self.conv_data = self.process_entries(slang_path, conv_dataset)
        else:
            self.slang_data, self.conv_data, self.slang_data_oov = self.process_entries(slang_path, conv_dataset)
        
        self.N_total = len(self.slang_data)
        if self.load_oov:
            self.N_total_oov = len(self.slang_data_oov)
        
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
        if self.load_oov:
            slang_entries, slang_entries_oov = self.load_data(slang_path, conv_dataset)
        else:
            slang_entries = self.load_data(slang_path, conv_dataset)
        
        conv_data = conv_dataset.data
        
        slang_data = [d for d in slang_entries if not acronym_check(d)]
        slang_data = [d for d in slang_data if not has_close_conv_def(str(d.word), d.def_sent, conv_data)]
        
        if not self.load_oov:
            return slang_data, conv_data
        
        slang_data_oov = [d for d in slang_entries_oov if (not acronym_check(d) and not ' ' in str(d.word))]
        
        return slang_data, conv_data, slang_data_oov
        
    def __str__(self):
        out = ""
        out += "Dataset Name: " + "\n"
        out += "Total Definition Entries: %d" % self.N_total + "\n"
        out += "Vocab Size: %d" % self.V + "\n"
        if self.load_oov:
            out += "Total OOV Definition Entries: %d" % self.N_total_oov + "\n"
        return out
    
class OSD_Dataset(SlangDataset):
    
    def __init__(self, slang_path, conv_dataset, load_oov=False):
        super().__init__(slang_path, conv_dataset, load_oov)
        self.meta_set.add('pos')
        self.meta_set.add('context')
        
    def load_data(self, slang_path, conv_dataset):
        
        data_OSD_raw = np.load(slang_path, allow_pickle=True)
        
        def process_def(d):
            return SlangEntry(d.word, d.def_sent, {'pos':d.type, 'context':d.meta_data})
        
        data_OSD = [process_def(d) for d in data_OSD_raw if str(d.word) in conv_dataset.vocab]
        
        if not self.load_oov:
            return data_OSD
        
        data_OSD_oov = [process_def(d) for d in data_OSD_raw if str(d.word) not in conv_dataset.vocab]

        return data_OSD, data_OSD_oov
    
class UD_Wil_Dataset(SlangDataset):
    
    def __init__(self, slang_path, conv_dataset, load_oov=False):
        super().__init__(slang_path, conv_dataset, load_oov)
        self.meta_set.add('context')
        
    def load_data(self, slang_path, conv_dataset):
        
        data_train = pd.read_csv(slang_path+'/train.tsv', sep='\t', error_bad_lines=False, header=None, usecols=[0,1,2]).values
        data_test = pd.read_csv(slang_path+'/test.tsv', sep='\t', error_bad_lines=False, header=None, usecols=[0,1,2]).values
        
        filter_mask_train = np.load(slang_path+'/filter_mask_train.npy')
        filter_mask_test = np.load(slang_path+'/filter_mask_test.npy')
        
        data_raw = np.concatenate((data_train[filter_mask_train], data_test[filter_mask_test]))
        
        def process_def(d):
            return SlangEntry(str(d[0]), str(d[1]), {'context':[str(d[2]).replace(str(d[0]), '[*SLANGAAAP*]')]})
        
        data_UD_Wil = [process_def(data_raw[i]) for i in range(data_raw.shape[0]) if str(data_raw[i][0]) in conv_dataset.vocab]
        
        if not self.load_oov:
            return data_UD_Wil
        
        data_UD_Wil_oov = [process_def(data_raw[i]) for i in range(data_raw.shape[0]) if str(data_raw[i][0]) not in conv_dataset.vocab]

        return data_UD_Wil, data_UD_Wil_oov
    
class GDS_Dataset(SlangDataset):
    
    def __init__(self, slang_path, conv_dataset, load_oov=False):
        super().__init__(slang_path, conv_dataset, load_oov)
        
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
    
    def __init__(self, slang_path, conv_dataset, load_oov=False):
        super().__init__(slang_path, conv_dataset, load_oov)
        
    def load_data(self, slang_path, conv_dataset):
        
        data_Urban_raw = np.load(slang_path, allow_pickle=True)
        
        re_hex = re.compile(r"\\x[a-f0-9][a-f0-9]")
        re_spacechar = re.compile(r"\\(n|t)")
        
        def process_def(d):
            return SlangEntry(d[0], re_spacechar.sub('', re_hex.sub('', d[1])), {})
        
        data_Urban = [process_def(d) for d in data_Urban_raw if d[0] in conv_dataset.vocab]
        
        return data_Urban