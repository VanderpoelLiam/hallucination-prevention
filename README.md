# Investigating the causes for and prevention of hallucinations in abstractive summarization
This is the public version of the accompanying code for my masters thesis at ETH. The project is ongoing and will be completed by August 2022. This project requires using the Euler cluster at ETH for its computing power. See [Getting started with clusters](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) for an introduction to clusters at ETH.

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [To install fairseq on Euler server](#to-install-fairseq-on-euler-server)
- [Preparing XSUM dataset](#preparing-xsum-dataset)
	- [Download raw XSUM dataset](#download-raw-xsum-dataset)
	- [Download test/validation/train split JSON](#download-testvalidationtrain-split-json)
	- [Installing Moses](#installing-moses)
	- [Install BPE encoder](#install-bpe-encoder)
	- [To install files2rouge](#to-install-files2rouge)
	- [Perform preprocessing](#perform-preprocessing)
	- [Move data to the server](#move-data-to-the-server)
- [Train a Transformer model for summarization](#train-a-transformer-model-for-summarization)
- [How to evaluate a trained model](#how-to-evaluate-a-trained-model)
	- [To generate summaries on the test set](#to-generate-summaries-on-the-test-set)
	- [Computing ROUGE scores](#computing-rouge-scores)
- [Train a language model](#train-a-language-model)
	- [xsum-lang i.e. target only language model](#xsum-lang-ie-target-only-language-model)
	- [xsum-lang-full i.e. target and source language model](#xsum-lang-full-ie-target-and-source-language-model)
	- [Evaluate language model](#evaluate-language-model)
- [Hyperparameter selection](#hyperparameter-selection)
	- [Language model](#language-model)
	- [MMI Decoding](#mmi-decoding)
		- [Theoretical approach](#theoretical-approach)
		- [Implementation](#implementation)
	- [Generating on validation vs. test sets](#generating-on-validation-vs-test-sets)
- [Modifying fairseq](#modifying-fairseq)
	- [Displaying probabilities for language and summarizer models separately](#displaying-probabilities-for-language-and-summarizer-models-separately)
	- [Displaying entropy](#displaying-entropy)
		- [For reference summaries](#for-reference-summaries)
		- [For generated summaries](#for-generated-summaries)
- [XSum Hallucination Annotations](#xsum-hallucination-annotations)
	- [Post-processing labels](#post-processing-labels)
	- [Computing token level entropy](#computing-token-level-entropy)
	- [Statistics for entropy of hallucinated tokens](#statistics-for-entropy-of-hallucinated-tokens)
	- [Statistics for probability of hallucinated tokens](#statistics-for-probability-of-hallucinated-tokens)
	- [Selecting optimal lambda value](#selecting-optimal-lambda-value)
- [Entropy threshold MMI decoding](#entropy-threshold-mmi-decoding)

<!-- /TOC -->
## To install fairseq on Euler server
Ensure any previously installed versions of fairseq are removed with `python -m pip uninstall fairseq`. Then fork the repository. Instructions are based on [general fairseq instructions](https://github.com/pytorch/fairseq#requirements-and-installation) and [euler specific instructions](https://github.com/jasonwei20/fairseq/blob/master/jason-lm-wt103/run-wikitext.MD):

```
git clone https://github.com/VanderpoelLiam/fairseq
module load gcc/6.3.0 python_gpu/3.8.5 hdf5 eth_proxy
python -m venv fair_env
source fair_env/bin/activate
cd fairseq
PYTHONPATH=$(which python)
$PYTHONPATH -m pip install --upgrade pip
$PYTHONPATH -m pip install --editable ./
```

Each time login to server need to run:
```
module load gcc/6.3.0 python_gpu/3.8.5 hdf5 eth_proxy
source fair_env/bin/activate
PYTHONPATH=$(which python)
```

## Preparing XSUM dataset
### Download raw XSUM dataset

Download dataset of 237018 articles.
`wget http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz`.

Then extract the `bbc-summary-data` folder.
`tar -xzf XSUM-EMNLP18-Summary-Data-Original.tar.gz`

See [here](https://github.com/EdinburghNLP/XSum/issues/9) and [here](https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/bart/README.summarization.md) for more information.

### Download test/validation/train split JSON

`wget https://github.com/EdinburghNLP/XSum/blob/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json`

### Installing Moses
Ensure that the Moses tokeniser is installed with:
`git clone https://github.com/moses-smt/mosesdecoder.git`

See [here](http://www.statmt.org/moses/?n=Development.GetStarted) for dependencies.

### Install BPE encoder
Follow instructions [here](https://github.com/glample/fastBPE) to install fastBPE.

### To install files2rouge
Based on [files2rouge](https://github.com/pltrdy/files2rouge), do the following:
```
$PYTHONPATH -m pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git     
cd files2rouge
$PYTHONPATH setup_rouge.py
$PYTHONPATH setup.py install
```

### Perform preprocessing
Run `python src/preprocess.py` to preprocess the data. The resulting data is stored in the `data` directory. The full datasets correspond to `xsum-summarizer`, `xsum-lang` and `xsum-lang-full`. The sample datasets correspond to  `xsum-summarizer-samples`, `xsum-lang-samples` and `xsum-lang-samples-full`.

The sample dataset consists of `4/2/2` examples in train/valid/test respectively. It is useful for debugging purposes to not work with the full XSUM dataset.

### Move data to the server
Log into server with `sftp`. Ensure local and remote directories are the same then run `put -r .` to move over all files. Do this for the following folder in the `data` directory: `xsum-summarizer`,  `xsum-lang`, `xsum-lang-full`.

## Train a Transformer model for summarization
Based on [this](https://github.com/pytorch/fairseq/tree/main/examples/translation#training-a-new-model). An alternate approach using a BART pre-trained model is explained [here](https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/bart/README.summarization.md).

Use the `src/train_command.py` script to generate the `fairseq-train` commands and copy them to the clipboard. Paste the result on Euler to run. Basic usage is:
```
python src/train_command.py 1
```
to run experiment `1` and log to `logs/experiments/train_1.log`.

Run `python src/train_command.py -h` for more information on the other parameters.

## How to evaluate a trained model
### To generate summaries on the test set
Similarly,

Use the `src/generate_command.py` script to generate the `fairseq-generate` commands. Basic usage is:
```
python src/generate_command.py 1 --wait_train
```
to run experiment `1` and log to `logs/experiments/generate_1.log`.

Run `python src/generate_command.py -h` for more information on the other parameters

### Computing ROUGE scores
Running `src/score_generate.sh 1` extracts the target/hypothesis sentences from `logs/experiments/generate_1.log` to the `logs/rouge/generate_1` directory and removes tokenization and BPE. The full ROUGE scores are saved to the `logs/rouge/generate_1/score` file and the `F_1` scores are output to the terminal.


## Train a language model
Based on [this](https://github.com/pytorch/fairseq/blob/main/examples/language_model/README.md).

Cluster batch job: Run for 12 hours (720 mins) on 4 cores each with 4096 MB of memory and with 2 GPUs.

fairseq-train parameters: We use a basic transformer language model with the default parameters to start. `xsum-lang` corresponds to the dataset of only targets. `xsum-lang-full` corresponds to the dataset of both source and targets. This leads to two sets of parameters.

### xsum-lang i.e. target only language model

`bsub -J train_lang -o logs/experiments/train_lang_1.log -W 720 -n 4 -R "rusage[mem=4096]" -R "rusage[ngpus_excl_p=2]"`

```
fairseq-train --task language_modeling \
  data/xsum-lang \
  --save-dir checkpoints/lang \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 2 \
  --fp16 \
  --no-epoch-checkpoints --no-last-checkpoints \
  --max-update 50000
```

### xsum-lang-full i.e. target and source language model

`bsub -J train_lang_full -o logs/experiments/train_lang_full_1.log -W 1000 -n 4 -R "rusage[mem=4096]" -R "rusage[ngpus_excl_p=2]"`

```
fairseq-train --task language_modeling \
  data/xsum-lang-full \
  --save-dir checkpoints/lang_full \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 2 \
  --fp16 \
  --no-epoch-checkpoints --no-last-checkpoints \
  --max-update 50000
```

### Evaluate language model
The below example is for `xsum-lang`. The line `-w "ended(train_lang)"` allows us to submit training and evaluation jobs to euler at the same time, but makes the evaluation job wait until the training is complete before running.

`bsub -J eval -w "ended(train_lang)" -o logs/experiments/eval_lang_1.log -W 60 -n 4 -R "rusage[mem=4096]" -R "rusage[ngpus_excl_p=1]"`

```
fairseq-eval-lm data/xsum-lang \
    --path checkpoints/lang/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400
```

and for `xsum-lang-full`.

`bsub -J eval_full -w "ended(train_lang_full)" -o logs/experiments/eval_lang_full_1.log -W 60 -n 4 -R "rusage[mem=4096]" -R "rusage[ngpus_excl_p=1]"`

```
fairseq-eval-lm data/xsum-lang-full \
    --path checkpoints/lang_full/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400
```


## Hyperparameter selection
### Language model
The parameters we look at changing are `decoder_layers`, `decoder_ffn_embed_dim`, `decoder_attention_heads`. We do a grid search over their default values scaled up/down by 2.

The loss is used as the validation metric by default.


### MMI Decoding
#### Theoretical approach
`\lambda` should take a value in `[0, 1]`. I found the model would generate nonsense close to 1. My prior is therefore shifted towards zero. So I pick 10 values on a log scale in `(0, 1)` (i.e. `2^-1, 2^-2, ..., 2^-9, 2^-10`).

`\gamma` should take a value in `[0, N_t]` where `N_t` is the length of hypothesis `t`.  But this would lead to a dependence on `t`. Instead I estimate `M`, the average value of `N_t` for all hypothesis generated by our summarizer (we count tokens not words). I then pick 10 values evenly in `(0, M)`. `M` was computed to be `26.90` so we select gamma from `[2, 4, 7, 9, 12, 14, 17, 19, 22, 24]`.

`\mu` scales the influence of `log(N_t)` which is on average `log(26.90) = 3.292`. Our prior is that we want all terms to be around the same order of magnitude. The average value of `log(y|x)` is `-1.102`, so picking `\mu` to be one of (`2^-1, 2^-2, ..., 2^-9, 2^-10`) feels appropriate.

ROUGE-1 RL scores are used as the validation metric.

#### Implementation
Use the `src/lambda_gridsearch_command.py` script to generate the batch command as for both train and generate. For example:
```
python src/lambda_gridsearch_command.py 8 --lang_full
```
runs model number `8` with the full language model. See `python src/lambda_gridsearch_command.py -h` for help.

The logs are saved to `logs/hyperparameters/lang_full/8`. In the log directory, we generate `batch.log` which is the batch command logs. Then, for each lambda of value `x`, we create two files: `x.log`, the results of `fairseq-generate` and `score_x` the corresponding rouge scores.

### Generating on validation vs. test sets
By default we generate on the test set with `fairseq-generate`. However for hyperparameter selection we need to generate on the validation set. This is done by adding the parameter `--gen-subset "valid"` to `fairseq-generate`.


## Modifying fairseq
I forked fairseq to [this repository](https://github.com/VanderpoelLiam/fairseq), then created a new branch called `liam` to make modifications to fairseq. I use the `lstm` and `lstm_lm` architectures as well as the `xsum-summarizer-samples` and `xsum-lang-samples` datasets. The architectures and datasets are chosen to be fast to train and generate with.

I also create the `test-project` directory with the following structure:
```
test-project
├── checkpoints
│   ├── summarization_dummy/
│   └── lang_dummy/
└── data
    ├── xsum-summarizer-samples/[...]
    └── xsum-lang-samples/[...]
```

Where the `xsum-*` directories contain the preprocessed sample datasets.

Then to initialize the models, we train them for one iteration. For the summarization model run:
```
bsub -I -W 10 -n 4 -R "rusage[mem=4096]" \
fairseq-train data/xsum-summarizer-samples \
  --arch lstm \
  --save-dir checkpoints/summarization_dummy \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 4096 \
  --max-update 1 \
  --max-epoch 1
```

and for the language model run:
```
bsub -I -W 10 -n 4 -R "rusage[mem=4096]" \
fairseq-train data/xsum-lang-samples \
  --task language_modeling \
  --arch lstm_lm \
  --save-dir checkpoints/lang_dummy \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 4096 \
  --max-update 1 \
  --max-epoch 1
```

Then to generate with `\lambda = 1` run:
```
bsub -I -W 10 -n 4 -R "rusage[mem=4096]" -R "rusage[ngpus_excl_p=1]" \
fairseq-generate \
data/xsum-summarizer-samples \
--gen-subset "valid" \
--path checkpoints/summarization_dummy/checkpoint_best.pt \
--batch-size 16 --beam 5 --truncate-source \
--skip-invalid-size-inputs-valid-test \
--lm-path checkpoints/lang_dummy/checkpoint_best.pt \
--lm-weight -1
```

With an unmodified `fairseq-generate` this produces for a single article and output of the form (I add `[...]` for brevity):
```
S-1	Transport Minister Juan Mol@@ in@@ ar said [...]
T-1	One of Mexico &apos;s biggest airlines , Mex@@ ic@@ ana de [...]
H-1	5.0183587074279785	roadside Venezuel@@ released Venezuel@@ [...]
D-1	5.0183587074279785	roadside Venezuel@@ released Venezuel@@ [...]
P-1	2.4488 2.6601 2.9603 3.0198 3.1490 3.3097 3.4029 3.4311 [...]
```


### Displaying probabilities for language and summarizer models separately
The MMI decoding objective has the form: `\log p(y | x) - \lambda \log p(y)`.

`P-1` is an array where the `i`'th  entry `P-1[i]` corresponds to: `\log p(y_i | x, y_{<i}) - \lambda \log p(y_i | y_{<i})`.

Our modifications produce two additional arrays, `P_SM` and `P_LM` which correspond to `\log p(y | x)` and `\log p(y)` respectively. Therefore, `P_SM-1[i]` corresponds to `\log p(y_i | x, y_{<i})` and `P_LM-1[i]` corresponds to`\log p(y_i | y_{<i})`.

This looks like:
```
[...]
P-1	2.4488 2.6601 2.9603 3.0198 3.1490 3.3097 3.4029 3.4311 [...]
P_SM-1	-8.3416 -8.3461 -7.9928 -7.9528 -7.8088 -7.7310 -7.6555 [...]
P_LM-1	-10.7904 -11.0063 -10.9531 -10.9726 -10.9579 -11.0407 [...]
```

As a sanity check we can see that for `i=0`, we have `2.4488 = P-1[0] = P_SM-1[0] - 1 * P_LM-1[0] = -8.3416 - 1 * -10.7904 = 2.4488`.


### Displaying entropy
#### For reference summaries
The `sequence_scorer.py` file in fairseq was then modified to compute token level entropy values on the reference summaries. Running:
```
fairseq-generate \
data/xsum-summarizer-samples \
--gen-subset "valid" \
--path checkpoints/summarization_model/11/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--score-reference \
--truncate-source
```

should result in an output of the form:
```
[...]
T-1	One of Mexico &apos;s biggest airlines , Mex@@ ic@@ ana de [...]
H-1	-3.9946939945220947	One of Mexico &apos;s biggest airlines , [...]
P-1	-6.4942 -0.1495 -0.7367 -0.0593 -1.9511 -1.6154 -1.9322 [...]
ENT-1	5.9865 1.0240 2.1275 0.4982 3.9551 3.9361 2.8885 7.6781 [...]
```

#### For generated summaries
The `generate.py` file in fairseq was then modified to compute token level entropy values on the generate summaries. Running:
```
fairseq-generate \
data/xsum-summarizer-samples \
--gen-subset "valid" \
--path checkpoints/summarization_model/11/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--truncate-source
```
should result in an output of the form:
```
[...]
T-1	One of Mexico &apos;s biggest airlines , Mex@@ ic@@ ana de [...]
H-1	-1.867172122001648	The United States airline Mex@@ ana Link [...]
P-1	-3.8778 -2.6076 -0.3309 -2.6852 -3.0132 -0.2178 -1.5003 [...]
ENT-1	7.9626 4.0794 2.9789 7.7642 2.0363 4.1919 2.5753 5.1570 [...]
```


## XSum Hallucination Annotations
Maynez et al. provide faithfulness and factuality annotations of 500 XSum summaries in their [repository](https://github.com/google-research-datasets/xsum_hallucination_annotations).

Zhou et al. further process this data to a more usable format for our purposes in their [repository](https://github.com/violet-zct/fairseq-detect-hallucination/tree/master/eval_data/xsum/Gold). As we are only interested in the hallucination labels for the reference summaries, we copy the relevant data to `data/xsum-hallucination-raw/`:
```
data
└── xsum-hallucination
    ├── Gold.docid
    ├── Gold.label
    ├── Gold.ref
    ├── Gold.source
    └── Gold.target
```

### Post-processing labels
The labels in `Gold.label` are for the summaries in `Gold.target`. However we want labels for the result of the `Gold.ref` summaries after applying tokenisation and BPE. This is done as follows (base directory for scripts is `src/hallucination_labelling/`):

1. Create `data/Xsum-hallucination-split.json` containing all the id's from `data/xsum-hallucination-raw/Gold.docid` in the test split and nothing in the train/val splits.
2. Run the preprocessing on this split to get all the `test.*` files and the two `dict.*` files in `data/xsum-hallucination/`. We want labels for the sentences in `test.bpe.target`.
3. Next, run `align_labels.py` to get the labels for most of `test.bpe.target` and save to `test.label`. These labels are extracted from `Gold.label` and aligned with `Gold.target`.
4. Missing labels are indicated by a `?` and these cases are processed manually with a helper in the same script.

### Computing token level entropy
Run the `fairseq-generate` on the data in `data/xsum-hallucination/` in order to get the token level entropy values for the summarization and language models:

```
fairseq-generate \
data/xsum-hallucination \
--path checkpoints/summarization_model/11/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--score-reference \
--truncate-source \
--lm-path checkpoints/lang_full/checkpoint_best.pt \
--lm-weight 0 > logs/experiments/generate_hallucination.log
```

Then run:
```
python src/hallucination_labelling/align_data.py
```
in order to extract the entropy scores to `test.entropy` and `test.entropy.lang_full` and the probability scores to `test.prob.sm` and `test.prob.lm` in the directory `data/xsum-hallucination/`. This data is now aligned with the `test.label` hallucination labels in the same directory.

### Statistics for entropy of hallucinated tokens
Given the data in `test.label`, `test.entropy` and `test.entropy.lang_full` in `data/xsum-hallucination/` we want to generate statistics on this data.

Run `entropy_stats.py` in `src/hallucination_labelling` in order to get statistics and distribution information comparing the entropy values for various token labellings.

### Statistics for probability of hallucinated tokens
Likewise we want statistics on `test.prob.sm` and `test.prob.lm`.

Run `probability_stats.py` in `src/hallucination_labelling` in order to get statistics and distribution information.

### Selecting optimal lambda value
We want to pick the best lambda value such that it minimizes the average log probability of Initial Hallucinated tokens (i.e. P = P_SM + $\lambda$ P_LM) and maximizes the ROUGE score of the 500 generated sentences.

We use `src/hallucination_labelling/generate_lambdas.py` to generate the lambda values. Then for each lambda value:
* Run `fairseq-generate` as seen in the [token level entropy](#computing-token-level-entropy) section to score the references and get the average log probability of Initial Hallucinated tokens
* Run `fairseq-generate` but without the `--score-reference` parameter to generate hypothesis sentences, and compute the ROUGE score
* This gives 2 values, a log probability and a ROUGE score. Add this point to a plot. Our goal is to pick lambda that maximizes ROUGE and minimizes log probability.

All scripts are in the `src/hallucination_labelling/` directory. To run the search, run `lambda_search_command.py` with the appropriate parameters. To extract the raw results to a directory in `logs/hallucination_labelling` run `process_lambda_search.py` with identical parameters. To finish the processing locally run the `process_lambda_search.py` with the addition of the `--local` parameter. This is due to limitations on what I can install on the server. Lastly run `plot_lambda_search.py` to get the resulting plot

## Entropy threshold MMI decoding
Based on the statistics for the first hallucinated token in a sequence, we can see that high entropy is correlated with the start of a hallucination. So the idea is modify fairseq to have entropy above a threshold trigger MMI decoding.

For actual generation we need to modify the test set to remove the 500 test examples that were used to pick the threshold value to avoid information leakage. This is the dataset `xsum-summarizer-no-500`.

As a first experiment I pick the best non-zero lambda value for the `LM_FULL` language model and  the entropy threshold as the average entropy for a token at the start of a hallucination for the summarization model. I generate on the validation set for simplicity but in the future I should be using the test set with the 500 examples removed:
```
bsub -J generate_lambda_1.5625E-02_sm_ent_threshold_4.2 \
-o logs/experiments/generate_lambda_1.5625E-02_sm_ent_threshold_4.2.log \
-W 60 -n 4 -R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
fairseq-generate \
data/xsum-summarizer \
--path checkpoints/summarization_model/11/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--gen-subset "valid" \
--truncate-source \
--lm-path checkpoints/lang_full/checkpoint_best.pt \
--lm-weight -0.015625 \
--ent-threshold 4.2
```
