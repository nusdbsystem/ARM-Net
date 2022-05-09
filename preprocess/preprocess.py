import argparse
import os
from logparser import Drain
from semantic_encode import SemanticEncode
import embed


# set the detail of each log file
dataset_settings = {
    'BGL': {
        'log_file': 'BGL.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'st': 0.5,
        'depth': 4
        },

    'HDFS': {
        'log_file': 'HDFS.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.5,
        'depth': 4
        }
}


def get_args():
    parser = argparse.ArgumentParser(description='preprocess each log message and generate an initial vector')
    parser.add_argument('--dataset', default='HDFS', type=str, help='dataset name')
    parser.add_argument('--input', type=str, help='dataset directory')
    parser.add_argument('--output', type=str, help='output directory')
    args = parser.parse_args()

    return args


args = get_args()
dataset = args.dataset
setting = dataset_settings[dataset]
input = args.input
output = args.output

indir = os.path.join(input, os.path.dirname(setting['log_file']))
log_file = os.path.basename(setting['log_file'])
print(indir, log_file)

# use the Drain parser for log parsing, and generate two files
# _strctured.csv: parse unstructured log messages into structured logs
# _templates.csv: extract constant log templates from raw log messages
parser = Drain.LogParser(log_format=setting['log_format'], indir=indir, outdir=output, rex=setting['regex'],
                         depth=setting['depth'], st=setting['st'])
parser.parse(log_file)

# use the fasttext for semantic encoding, and generate one file
# _embedding300.log: encode each word into a vector of 300 dimensions
semantic_encoder = SemanticEncode(input_dir=output, filename=setting['log_file'])
semantic_encoder.extract_domain_info()

# build an embedding vector for each log message via embedding lookup, and generate one file
#
embedder = embed.Embedder(path=output)
embedder.embed(log_file)
