from torchvision import transforms
from itertools import chain
from string import punctuation
import json

class Vocab:
    def __init__(self, data):
        self.word2idx = {}
        self.counter = 0
        self.build_vocab(data)
        
    def build_vocab(self, data):
        special = ['<start>', '<end>', '<unknown>']
        base =  special + list(punctuation)
        for word in chain(base, data): 
            self.add(word)
    
    def add(self, word):
        word = word.lower()
        if word not in self.word2idx:
            self.word2idx[word] = self.counter
            self.counter += 1
    
    def __call__(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx['<unknown>']
    
    def __len__(self):
        return len(self.word2idx)
    
img_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_tokens(json_file):
    with open(json_file) as fp:
        data = json.load(fp)
        tokens = []
        dataset = data['images']
        for entry in dataset:
            sents = entry['sentences']
            for sent in sents:
                tokens.extend(sent['tokens'])
        return tokens