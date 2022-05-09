import os
import pandas as pd
import fasttext
import re
from collections import Counter
import numpy as np
import pickle


class SemanticEncode:
    def __init__(self, input_dir: str, filename: str):
        self.path = input_dir
        self.filename = filename

    def preprocess_template(self):
        df = pd.read_csv(os.path.join(self.path, self.filename + '_templates.csv'))
        lst_template = df['EventTemplate']
        lst_eventid = df['EventId']

        dict_template = {}
        for index, template in enumerate(lst_template):
            # split variables in the camel format
            template = self.transfer_camelname(template)
            # remove punctuation
            template = self.remove_nonword(template)
            # select non-stop words
            template = ' '.join(template.split())
            template = template.split()
            dict_template[lst_eventid[index]] = template

        return dict_template

    def transfer_camelname(self, template):
        template = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', template)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', template).lower()

    def remove_nonword(self, template):
        template = re.sub('\W+', ' ', template).replace("_", ' ')
        template = re.sub('\d', ' ', template)
        return template

    def build_semantic(self, dict_template):
        # training model for vector encoding
        # model = fasttext.train_unsupervised('dataset/fasttext/fil9', 'skipgram', minn=2, maxn=5, dim=300)
        # model.save_model('dataset/fasttext/fil9_skipgram_300.bin')

        # load trained fasttext model and generate encoding
        model = fasttext.load_model('dataset/fasttext/fil9_skipgram_300.bin')

        dict_counter = {}
        set_vocabulary = set()
        for eventid in dict_template:
            dict_counter[eventid] = Counter(dict_template[eventid])
            for vocabulary in dict_template[eventid]:
                set_vocabulary.add(vocabulary)

        dict_vocabulary = {}
        for vocabulary in set_vocabulary:
            dict_vocabulary[vocabulary] = model.get_word_vector(vocabulary)

        # building tf-idf-matrix
        df = pd.read_csv(os.path.join(self.path, self.filename + '_structured.csv'))
        lst_eventid = df['EventId']

        matrix = []
        for eventid in lst_eventid:
            matrix.append(dict_counter[eventid])

        df_matrix = pd.DataFrame(matrix)
        df_matrix = df_matrix.fillna(0)

        lst_columns = df_matrix.columns
        matrix_value = df_matrix.values
        num_events = matrix_value.shape[0]

        tmp_idf = np.sum(matrix_value > 0, axis=0)
        idf_vec = np.log(num_events / tmp_idf)
        dict_idf = {}
        for index, idf in enumerate(idf_vec):
            dict_idf[lst_columns[index]] = idf

        dict_embedding = {}
        for eventid in dict_template:
            num_vocabulary = len(dict_template[eventid])
            if num_vocabulary == 0:
                dict_embedding[eventid] = np.zeros(300)
            else:
                embedding = np.zeros(300)
                for vocabulary in dict_counter[eventid]:
                    tf = dict_counter[eventid][vocabulary] / num_vocabulary
                    weight = tf * dict_idf[vocabulary] * dict_counter[eventid][vocabulary]
                    embedding += (dict_vocabulary[vocabulary] * weight)
                embedding /= num_vocabulary
                dict_embedding[eventid] = embedding

        file_embedding = os.path.join(self.path, self.filename + '_embedding300.log')
        with open(file_embedding, 'wb') as fe:
            pickle.dump(dict_embedding, fe, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_domain_info(self):
        dict_template = self.preprocess_template()
        self.build_semantic(dict_template)