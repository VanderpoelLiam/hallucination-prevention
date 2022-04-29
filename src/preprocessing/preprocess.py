
# Run with python 3

import json
import os
import subprocess

from os.path import join
from tqdm import tqdm

def extract(in_dir, out_dir, source, target, files):
    source = open(join(out_dir, source) + ".txt", "a")
    target = open(join(out_dir, target) + ".txt", "a")

    for fname in tqdm(files):
        text_in = open(join(in_dir, fname) + ".summary").read()
        t0 = text_in.split("[SN]FIRST-SENTENCE[SN]\n", 1)[1]
        t1 = t0.split("[SN]RESTBODY[SN]\n", 1)
        source_out = t1[1].replace("\n", " ") + "\n"
        target_out = t1[0].replace("\n", " ") + "\n"
        source.write(source_out)
        target.write(target_out)

def tokenize(out_dir, lang_dir, split_filename):
    split_dict = json.loads(open(split_filename).read())
    data_types = ["test", "validation", "train"]
    os.system("mkdir -p " + out_dir)

    sources = ["test.source", "valid.source", "train.source"]
    targets = ["test.target", "valid.target", "train.target"]

    list_files = []
    for type in data_types:
        list_files.append(split_dict[type])

    for source, target, files in zip(sources, targets, list_files):
        extract(in_dir, out_dir, source, target, files)

    for source, target in zip(sources, targets):
        subprocess.run([src_dir + 'tokenize.sh', out_dir, source, target])

def preprocess(out_dir, data_dir, lang_dir, split_filename, is_full=False):
    tokenize(out_dir, lang_dir, split_filename)
    if is_full:
        subprocess.run([src_dir + 'learn_bpe.sh', data_dir, out_dir])
        subprocess.run([src_dir + 'learn_dict.sh', out_dir, data_dir])
    else:
        subprocess.run([src_dir + 'bpe.sh', data_dir, out_dir])
    subprocess.run([src_dir + 'binarize.sh', out_dir, data_dir])
    subprocess.run([src_dir + 'binarize_lang_model.sh', out_dir, data_dir, lang_dir])
    subprocess.run([src_dir + 'cleanup_preprocessing.sh', out_dir])


if __name__ == "__main__":
    data_dir = "data/"
    src_dir = "src/preprocessing/"
    in_dir = data_dir + "bbc-summary-data"

    # For full dataset
    out_dir = data_dir + "xsum-summarizer"
    lang_dir = data_dir + "xsum-lang"
    split_filename = data_dir + "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
    preprocess(out_dir, data_dir, lang_dir, split_filename, is_full=True)

    # For small sample of dataset
    out_dir = data_dir + "xsum-summarizer-samples"
    lang_dir = data_dir + "xsum-lang-samples"
    split_filename = data_dir + "XSum-samples-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
    preprocess(out_dir, data_dir, lang_dir, split_filename)

    # For halluciantion labelled dataset
    out_dir = data_dir + "xsum-hallucination"
    lang_dir = data_dir + "xsum-hallucination-lang"
    split_filename = data_dir + "XSum-hallucination-split.json"
    preprocess(out_dir, data_dir, lang_dir, split_filename)

    # For full dataset without 500 test samples from hallucination dataset
    out_dir = data_dir + "xsum-summarizer-no-500"
    lang_dir = data_dir + "xsum-summarizer-no-500-lang"
    split_filename = data_dir + "XSum-no-500-split.json"
    preprocess(out_dir, data_dir, lang_dir, split_filename)
