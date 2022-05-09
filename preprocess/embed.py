import pandas as pd
import os
import pickle
import csv


class Embedder:
    def __init__(self, path):
        self.path = path

    def hdfs_timeembed(self):
        lst_hour, lst_minute, lst_second = [], [], []
        dict_hour, dict_minute, dict_second = {}, {}, {}
        lst_time = self.df_structure['Time'].unique()
        for item in lst_time:
            hour, minute, second = item[0:2], item[2:4], item[4:]
            if hour not in lst_hour:
                lst_hour.append(hour)
            if minute not in lst_minute:
                lst_minute.append(minute)
            if second not in lst_second:
                lst_second.append(second)
        lst_hour = sorted(lst_hour)
        lst_minute = sorted(lst_minute)
        lst_second = sorted(lst_second)

        for index, item in enumerate(lst_hour):
            dict_hour[item] = index
        for index, item in enumerate(lst_minute):
            dict_minute[item] = index
        for index, item in enumerate(lst_second):
            dict_second[item] = index

        return dict_hour, dict_minute, dict_second

    def bgl_timeembed(self):
        lst_hour, lst_minute, lst_second, lst_msecond = [], [], [], []
        dict_hour, dict_minute, dict_second, dict_msecond = {}, {}, {}, {}
        lst_time = self.df_structure['Time'].unique()

        for item in lst_time:
            hour = item.split('.')[0].split('-')[3]
            [minute, second, msecond] = item.split('.')[1:]

            if hour not in lst_hour:
                lst_hour.append(hour)
            if minute not in lst_minute:
                lst_minute.append(minute)
            if second not in lst_second:
                lst_second.append(second)
            msecond = round(int(msecond) / 10000)
            if msecond not in lst_msecond:
                lst_msecond.append(msecond)
        lst_hour = sorted(lst_hour)
        lst_minute = sorted(lst_minute)
        lst_second = sorted(lst_second)
        lst_msecond = sorted(lst_msecond)

        for index, item in enumerate(lst_hour):
            dict_hour[item] = index
        for index, item in enumerate(lst_minute):
            dict_minute[item] = index
        for index, item in enumerate(lst_second):
            dict_second[item] = index
        for index, item in enumerate(lst_msecond):
            dict_msecond[item] = index

        return dict_hour, dict_minute, dict_second, dict_msecond

    def item_frequency(self, header):
        dict_item = dict(self.df_structure[header].value_counts(dropna=False))
        lst_item = sorted(dict_item.items(), key=lambda item: item[1], reverse=True)

        dict_index = {}
        fwriter = open(os.path.join(self.path, self.logName + '_' + header.lower() + '.log'), 'w')

        for index, item in enumerate(lst_item):
            dict_index[item[0]] = index
            fwriter.write(str(index) + '\t' + str(item[0]) + '\t' + str(item[1]) + '\n')
        fwriter.close()

        return dict_index

    def event_frequency(self):
        lst_eventid = list(self.df_structure['EventId'].unique())
        self.df_template = self.df_template[self.df_template['EventId'].isin(lst_eventid)]

        dict_item = {}

        for index, row in self.df_template.iterrows():
            eventid, occurrence = row['EventId'], row['Occurrences']
            dict_item[eventid] = int(occurrence)

        lst_item = sorted(dict_item.items(), key=lambda item: item[1], reverse=True)
        dict_index = {}
        fwriter = open(os.path.join(self.path, self.logName + '_event.log'), 'w')

        for index, item in enumerate(lst_item):
            dict_index[item[0]] = index
            fwriter.write(str(index) + '\t' + str(item[0]) + '\t' + str(item[1]) + '\n')
        fwriter.close()

        return dict_index

    def hdfs_labelembed(self):
        dict_label = {}
        num_normal = 0
        num_anomaly = 0

        # the label file of HDFS
        labelpath = 'datasets/HDFS_1/anomaly_label.csv'
        with open(labelpath) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                blkid = row[0]
                label = row[1]
                if label == 'Normal':
                    dict_label[blkid] = 0
                    num_normal += 1
                elif label == 'Anomaly':
                    dict_label[blkid] = 1
                    num_anomaly += 1
        csvfile.close()

        print(num_normal, num_anomaly)
        return dict_label

    def bgl_labelembed(self):
        lst_orilabel = self.df_structure['Label']
        lst_label = []

        for label in lst_orilabel:
            if label == '-':
                lst_label.append(0)
            else:
                lst_label.append(1)

        return lst_label

    def semanticvec(self, dict_event):
        floader300 = open(os.path.join(self.path, self.logName + '_embedding300.log'), 'rb')
        dict_semantic300 = pickle.load(floader300)
        dict_result300 = {}
        for eventid in dict_semantic300:
            if eventid in dict_event:
                eventindex = dict_event[eventid]
                dict_result300[eventindex] = dict_semantic300[eventid]
        floader300.close()

        return dict_result300

    def preprocess(self, dict_result, dict_label):

        lst_dat = []

        for index, blk in enumerate(dict_result):
            lst_dat.append([dict_result[blk], dict_label[blk]])

        return lst_dat

    def hdfs_embed(self):
        dict_hour, dict_minute, dict_second = self.hdfs_timeembed()
        dict_pid = self.item_frequency('Pid')
        dict_level = self.item_frequency('Level')
        dict_component = self.item_frequency('Component')
        dict_event = self.event_frequency()
        dict_label = self.hdfs_labelembed()
        dict_semantic = self.semanticvec(dict_event)

        dict_result = {}
        for index, row in self.df_structure.iterrows():
            blkid, timestr, pid, level, component, eventid = row['BlockId'], row['Time'], row['Pid'], row['Level'], \
                                                             row['Component'], row['EventId']
            hour, minute, second = timestr[0:2], timestr[2:4], timestr[4:]
            lst_log = [dict_hour[hour], dict_minute[minute], dict_second[second], dict_pid[pid],
                       dict_level[level], dict_component[component], dict_event[eventid]]
            if blkid not in dict_result:
                dict_result[blkid] = []
            dict_result[blkid].append(lst_log)

        lst_dat = self.preprocess(dict_result, dict_label)

        encode_file = os.path.join(self.path, self.logName + '_all.log')
        fencoder = open(encode_file, 'wb')
        pickle.dump(lst_dat, fencoder)
        pickle.dump(dict_semantic, fencoder)
        fencoder.close()

    def bgl_embed(self):
        dict_hour, dict_minute, dict_second, dict_msecond = self.bgl_timeembed()
        dict_node = self.item_frequency('Node')
        dict_type = self.item_frequency('Type')
        dict_component = self.item_frequency('Component')
        dict_level = self.item_frequency('Level')
        dict_event = self.event_frequency()
        lst_label = self.bgl_labelembed()
        dict_semantic = self.semanticvec(dict_event)

        lst_sequence = []
        for index, row in self.df_structure.iterrows():
            timestamp, node, timestr, typestr, component, level, eventid = row['Timestamp'], row['Node'], row['Time'],\
                                                                           row['Type'], row['Component'], row['Level'],\
                                                                           row['EventId']
            hour = timestr.split('.')[0].split('-')[3]
            [minute, second, msecond] = timestr.split('.')[1:]
            msecond = round(int(msecond) / 10000)

            lst_sequence.append([dict_hour[hour], dict_minute[minute], dict_second[second], dict_msecond[msecond],
                       dict_node[node], dict_type[typestr], dict_component[component], dict_level[level],
                       dict_event[eventid]])
        lst_result = [lst_sequence, lst_label]
        encode_file = os.path.join(self.path, self.logName + '_sequence.log')
        fencoder = open(encode_file, 'wb')
        pickle.dump(lst_result, fencoder)
        pickle.dump(dict_semantic, fencoder)
        fencoder.close()

    def embed(self, logName):
        self.logName = logName

        self.df_structure = pd.read_csv(os.path.join(self.path, self.logName+'_structured.csv'), dtype='string')
        self.df_template = pd.read_csv(os.path.join(self.path, self.logName+'_templates.csv'), dtype='string')
        self.df_structure.fillna('UNKNOWN', inplace=True)

        if 'HDFS' in self.logName:
            self.hdfs_embed()
        elif 'BGL' in self.logName:
            self.df_structure = self.df_structure[self.df_structure['Type'].isin(['RAS', 'KERNEL', 'UNKNOWN'])]
            self.bgl_embed()