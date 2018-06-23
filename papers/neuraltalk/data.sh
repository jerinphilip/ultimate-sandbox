#!/bin/bash
set -x

# JSON and VGG Features
# 50MB
FLICKR8K="https://cs.stanford.edu/people/karpathy/deepimagesent/flickr8k.zip"

# 200MB
FLICKR30K="https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip"

# 750MB
COCO="https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip"

# All three, withough VGG
JSON_BLOBS="https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

wget -c $JSON_BLOBS;
