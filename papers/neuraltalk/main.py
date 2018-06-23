#!/usr/bin/python3

from argparse import ArgumentParser
import json
from pprint import pprint

parser = ArgumentParser()
parser.add_argument("--json", required=True, type=str)
args = parser.parse_args()

def construct_vocab(data):
    word2idx = {}
    counter = 0 
    def add(word):
        if word not in word2idx:
            word2idx[word] = counter
            counter += 1

    base = ['<start>', '<end>', '<unknown>']
    for word in base:
        add(word)

# Construct vocabulary

with open(args.json) as fp:
    data = json.load(fp)
    pprint(data.keys())
    dataset = data['images']
    for entry in dataset:
        sents = entry['sentences']
        for sent in sents:
            print(sent['raw'])
        print('-'*10)
        exit()

