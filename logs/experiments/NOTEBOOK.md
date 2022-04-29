# Summarization modelling
## Train
### `train_1`
SIGBUS error. 2 GPUs, `4096` memory. Interpretation is that it ran out of memory. `--max-tokens 4096 --update-freq 2`.

### `train_2`
Trained successfully until time limit. 2 GPUs, `4096` memory. `--max-tokens 2048 --update-freq 2`
```
epoch 034
valid | loss 8.128
train | loss 7.314
```

### `train_3`
Trained successfully until time limit. 2 GPUs, `4096` memory. `--max-tokens 4096 --update-freq 1`
```
epoch 032
valid | loss 8.388
train | loss 7.491
```

### `train_4`
Trained successfully until time limit. 1 GPU, `4096` memory. `--max-tokens 4096 --update-freq 8`. Much better performance because of higher `update-freq`. Doubling GPUs is equivalent to doubling the `update-freq`.

Next time:
  Max memory used was `4707` total while we had `16384` total allocated, should increase `update-freq` to use more of the memory or decrease memory requested. Also try continuing training from previous checkpoint.

Best validation score:
```
epoch 031
valid | loss 7.059
train | loss 5.968
```

Last epoch score:
```
epoch 034
valid | loss 7.078
train | loss 5.931
```

### `train_5`
I increased `update-freq` from `8` to `32` and decreased `dropout` from `0.3` to `0.2`. Started training from best model from `train_4`.

Trained successfully until time limit, 1 GPU. Still allocating too much memory. Decrease allocated memory to `2048` per CPU core. Did not improve on `valid` loss from `model_4`, but `train` loss went much lower, suggests overfitting. Trained for `30` epochs compared to `34` in  `train_4`, so increasing `update-freq` further should be okay. Could also use 2 GPUs.

Best validation score: Same as `train_4`.

Last epoch score:
```
epoch 054
valid | loss 7.426
train | loss 4.789
```

### `train_6`
Increased `update-freq` from `8` to `16` and changed the architecture and associated parameters to the larger `transformer_wmt_en_de` model. Best validation score occurs by epoch `5` so we are overtraining by a lot.

Best validation score:
```
epoch 005
valid | loss 6.871
train | loss 6.213
```

Last epoch score:
```
epoch 027
valid | loss 7.725
train | loss 4.339
```

### `train_7`
Increased `update-freq` from `16` to `64`. Used same`transformer_wmt_en_de` model as `train_6`. Increased dropout from `0.1` to `0.3` to fight overfitting. Still training too fast as reach best validation score in epoch `8`. Next time follow Clara suggestion to decrease learning rate and include patience so don't waste epochs `8` to `20` overfitting. Happy with memory usage and `update-freq`.

Best validation score:
```
epoch 008
valid | loss 6.718
train | loss 6.611
```

Last epoch score:
```
epoch 020
valid | loss 7.182
train | loss 5.104
```

### `train_8`
Decreased learning rate by factor of `10` from `7e-4` to `7e-5`. Added `--patience 5` to stop training if validation score does not improve after `5` epochs. Used dropout of `0.1`. As kept improving till last epoch should continue training for more time.

Best validation score = Last epoch score:
```
epoch 021
valid | loss 6.515
train | loss 6.228
```

### `train_9`
Continued training `train_8` for `20` more hours with `patience 5`.

From now on, I only report the best validation score as the `patience` parameter ensures I train for maximum `5` additional epochs after the validation score stops improving.

In total this model trained for `40` hours. We trained for `2` more epochs without improving validation score. I do not think this is worth training further. Rather I think we need to run `train_10` for an additional `20` hours.

Best validation score:
```
epoch 039
valid | loss 6.297
train | loss 5.407
```

### `train_10`
Same parameters as `train_8` but with dropout increased from `0.1` to `0.3` and time increased from `20` hours to `24` hours.

As expected from `train_9` this did not finish training so I ran it for more time.

Best validation score:
```
epoch 023
valid | loss 6.693
train | loss 6.663
```

### `train_11`
Continued training `train_10` for `24` more hours with `patience 5`.

This still did not finish training but I am happy with the performance and based on the improvement from `train_8` to `train_9` I suspect that any improvement will be marginal.

Best validation score:
```
epoch 050
valid | loss 6.225
train | loss 5.832
```

## Generate
### `generate_1`
Failed due to bug in my modification of fairseq.

### `generate_2`
```
R1: 18.68
R2: 2.514
RL: 14.70
```

### `generate_3`
```
R1: 17.81
R2: 2.533
RL: 14.24
```
### `generate_4`
Like in training we only use `4569` memory of the `16384` total allocated, so we should massively increase the batch size in the future.
```
R1: 19.06
R2: 3.166
RL: 15.08
```

### `generate_5`
Increased `batch-size` from `8` to `32`.

Could not run as training did not improve on `model_4`.

### `generate_6`
Used larger `batch-size` of `32` and the larger `transformer_wmt_en_de` model. Still only used `4667` of memory, so decreased CPU memory from `4096` to `2048` for next time. Also increased default `batch-size` to `64`.

Interesting how ROUGE scores are worse than `generate_4` even though `valid` scores are better.
```
R1: 18.8
R2: 3.112
RL: 14.92
```

### `generate_7`
Best ROUGE scores so far by a lot.
```
R1: 26.17
R2: 7.785
RL: 21.17
```

### `generate_8`
Another big improvement in ROUGE scores and training did not finish.
```
R1: 30.34
R2: 10.02
RL: 24.32
```

### `generate_9`
Slight improvement ROUGE scores with an additional `20` hours of training.
```
R1: 32.55
R2: 11.48
RL: 26.04
```

### `generate_10`
Performed worse than `generate_8` suggests increased dropout too much. But also did not finish training, need to wait on `generate_11`
```
R1: 29.55
R2: 9.55
RL: 23.89
```

### `generate_11`
This is the most performant model so far. It does better than `generate_9` on R2 and RL metrics but worse on R1.
```
R1: 32.35
R2: 11.53
RL: 26.16
```

# Language Modelling
## Train
### `train_lang_1`
Trained successfully until time limit.

### `train_lang_full_1`
Trained successfully until hit `--max-update 50000`

## Evaluate
A lower perplexity and loss indicates a better model.
### `eval_lang_1`
Loss (base 2): 5.4836, Perplexity: 44.74

### `eval_lang_full_1`
Loss (base 2): 5.1522, Perplexity: 35.56

# MMI Decoding
## `train_3` model
For LM, best performance is at `2^{-03}`, but pretty good in the `2^{-03}-2^{-05}` range. Big improvement from `2^{-02}` to `2^{-04}`. Results from `2^{-05}-2^{-10}` are pretty much identical. Interpretation is that language model gives an improvement but need to try more values in the `2^{-02}-2^{-05}` range.

For LM_FULL, same conclusions as for language model. Interesting that full model does not perform better even though it has better loss/perplexity.

### Results
Lamda | LM | LM_FULL
--- | --- | ---
2^{-01} | 3.007 | 4.75
2^{-02} | 14.06 | 14.13
2^{-03} | **14.5** | 14.44
2^{-04} | 14.48 | **14.45**
2^{-05} | 14.42 | 14.41
2^{-06} | 14.39 | 14.38
2^{-07} | 14.38 | 14.38
2^{-08} | 14.39 | 14.38
2^{-09} | 14.38 | 14.38
2^{-10} | 14.38 | 14.37
0       | 14.24 | 14.24


## `train_9` model
Got the following error:
`RuntimeError: CUDA out of memory. Tried to allocate 1.06 GiB (GPU 0; 10.76 GiB total capacity; 4.13 GiB already allocated; 1.02 GiB free; 8.59 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
`
This is likely because the batch size is too large. So I decreased the `batch_size` from `64` to `32` and increased the training time from `360` to `1000` mins.

Another issue is that I will not log to `lambda_search.log` until after the whole batch job is completed. So I need to have one to do the generating and another to process the generated output.

Weird thing is that it seems that the `fairseq-generate` did work. I am getting reasonably numbers when running the ROUGE scoring locally. It is not clear at what point the `out of memory` error occured. Only way to be certain is to rerun the script.

### Results
Bug in lambda generation meant I only had `9` unique lambda values

Lamda | LM | LM_FULL
--- | --- | ---
5.0000E-01 | 5.913 | 11.63
2.5000E-01 | 23.05 | 23.37
2.0318E-01 | 23.89 | 24.25
1.2500E-01 | 24.86 | 24.96
1.5850E-01 | 24.46 | 24.66
1.3116E-01 | 24.76 | 24.89
1.2137E-01 | 24.89 | 24.98
1.1525E-01 | 24.87 | 24.97
8.7584E-02 | 24.96 | 25.01
6.2500E-02 | 25.12 | 25.12
4.3715E-02 | 25.19 | 25.20
4.0373E-02 | 25.20 | 25.19
3.5432E-02 | 25.22 | 25.19
3.1250E-02 | 25.22 | 25.19
1.5625E-02 | 25.2 | **25.23**
7.8125E-03 | **25.23** | **25.23**
3.9062E-03 | 25.22 | 25.19
1.9531E-03 | 25.21 | 25.21
9.7656E-04 | 25.21 | 25.2
0       | **26.04** | **26.04**


## `train_11` model
Still getting `RuntimeError: CUDA out of memory.` error. I suspect batch size is still to large, so I decrease it from `32` to `16` in `MMI_decode.sh`. I left training time at `1000` mins as we only run for around `135` mins total with the `32` batch size.

New issue is that csplit overwrites files when I run two different gridsearch simultaneosly. Fix is to change the directory csplit writes to to the log directory e.g. `logs/hyperparameters/lang/11`.  

Also new is `Can't locate XML/Parser.pm in @INC` error. Emailed service desk as couldnt find a fix myself. But still could run all the ROUGE scoring locally.

On second attempt noticed that lambda is chosen to be `1.1525E-01` twice. Upon checking, the `1.1525E-01.log` file is correct, suggesting that the file was just overwritten. Fix is to do uniqueness check after rounding.

### Results

Lamda | LM | LM_FULL
--- | --- | ---
5.0000E-01 | 5.024 | 12.53
2.5000E-01 | 23.04 | 23.79
2.0318E-01 | 23.96 | 24.44
1.5850E-01 | 24.46 | 24.87
1.3116E-01 | 24.80 | 25.04
1.2500E-01 | 24.83 | 25.06
1.2137E-01 | 24.84 | 25.07
1.1525E-01 | 24.92 | 25.11
8.7584E-02 | 25.06 | 25.18
6.2500E-02 | 25.16 | 25.28
4.3715E-02 | 25.23 | 25.26
4.0373E-02 | 25.23 | 25.25
3.5432E-02 | 25.24 | 25.26
3.1250E-02 | 25.28 | 25.3
1.5625E-02 | **25.29** | **25.34**
7.8125E-03 | **25.29**| 25.29
3.9062E-03 | 25.28 | 25.29
1.9531E-03 | 25.26 | 25.27
9.7656E-04 | 25.27 | 25.27
0       | **26.16** | **26.16**

# Detecting Hallucinated Content
## Entropy by label
### Summarization Model 11
Label | Average Entropy
--- | ---
Hallucinated    | 3.8137
Non-Hallucinated | 3.6878
Initial Hallucinated | 4.2012
Subsequent Hallucinated | 3.7417

The T-test tests for the null hypothesis that two independent samples have identical average. For a p-value of 1% rejecting null-hypothesis means the averages are different. If we cannot reject, then the averages are the same. The data in the table is (whether we reject the null-hypothesis, t-statistic, p-value).

T-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '3.5703E+00', '3.5795E-04')
Initial Hallucinated      | ('Reject', '7.1804E+00', '7.5493E-13')
Subsequent Hallucinated | ('Cannot reject', '1.4406E+00', '1.4971E-01')

T-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '5.7472E+00', '9.6733E-09')

The null hypothesis for the Kolmogorov–Smirnov test is that the two distributions are identical. The python implementation `ks_2samp` takes two PDFs as input. We choose the same p-value cutoff and table entry format as for the T-test.

KS-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '5.0021E-02', '1.3288E-06')
Initial Hallucinated      | ('Reject', '1.2985E-01', '5.6948E-10')
Subsequent Hallucinated | ('Reject', '4.2314E-02', '2.1832E-04')

KS-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '1.3120E-01', '2.0800E-09')

### Language Model Full
Label | Average Entropy
--- | ---
Hallucinated | 3.6687
Non-Hallucinated | 3.7069
Initial Hallucinated | 4.8343
Subsequent Hallucinated | 3.4520

T-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Cannot reject', '-1.0552E+00', '2.9134E-01')
Initial Hallucinated      | ('Reject', '1.5453E+01', '3.9081E-53')
Subsequent Hallucinated | ('Reject', '-6.6845E+00', '2.4255E-11')

T-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '1.7396E+01', '1.1794E-65')

KS-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '4.4707E-02', '2.3172E-05')
Initial Hallucinated      | ('Reject', '3.0532E-01', '6.0953E-54')
Subsequent Hallucinated | ('Reject', '8.0390E-02', '1.0394E-14')

KS-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '3.4499E-01', '1.2130E-63')

#### Outlier in distribution plots
The distributions of Hallucinated, Initial Hallucinated and Non-Hallucinated all have an outlier bin `[5.2525, 5.3535)` which contains substantially more entries. This is caused by the entry `5.3129` appearing `164`, `164` and `336` times respectively. Notice that the sum of counts for Hallucinated and Non-Hallucinated is `500` (`164 + 336`), this suggests that this is a specific token with the same entropy in each line.

Indeed, by looking at the index where this entry appears in for each sentence we see that this is the entropy value for the first token in every sentence. Therefore it makes sense to remove this entropy value as it biases our averages based on the number of tokens with each label. The same analysis with the first token removed for all labels, entropies is given below.

### Language Model Full Outlier removed
Label | Average Entropy
--- | ---
Hallucinated            | 3.6067
Non-Hallucinated        | 3.6329
Initial Hallucinated    | 4.6897
Subsequent Hallucinated | 3.4520

T-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Cannot reject', '-7.0534E-01', '4.8061E-01')
Initial Hallucinated    | ('Reject', '1.2625E+01', '3.4584E-36')
Subsequent Hallucinated | ('Reject', '-4.6876E+00', '2.7975E-06')

T-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '-1.3671E+01', '1.0930E-41')

KS-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '4.4962E-02', '3.1902E-05')
Initial Hallucinated    | ('Reject', '2.3694E-01', '1.9953E-25')
Subsequent Hallucinated | ('Reject', '6.9108E-02', '8.0157E-11')

KS-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '2.6797E-01', '1.3639E-30')

## Log probabilities by label

The file `generate_hallucination.log` in `logs/experiments/` contains the log probabilities at the token level for both the summarization and language models. We wish to implement an entropy threshold such that when the entropy is above a certain value, we begin performing MMI decoding. Recall this means that our token probability is now: P_SM + $\lambda$ P_LM. The goal is to pick $\lambda$ to maximize the MMI objective given above.

As a first experiment, we can look at the average values for P_SM, P_LM by hallucination label (Hallucinated, Non-Hallucinated, Initial Hallucinated, Subsequent Hallucinated). I also checked that we are not predicting the same probability for all the starting tokens of the language model as was the case with entropy. This should indicate the sign of $\lambda$.

Label | Average P_SM | Average P_LM
--- | --- | ---
Hallucinated            | -5.3662 | -3.8137
Non-Hallucinated        | -4.5523 | -3.5381
Initial Hallucinated    | -6.9578 | -6.0433
Subsequent Hallucinated | -5.0702 | -3.3991

To maximize P_SM + $\lambda$ P_LM we should pick $\lambda$ to be negative.

# Entropy threshold decoding
As a first experiment I ran the entropy threshold implementation with $\lambda = -0.015625$ and the summarization entropy threshold of $4.2$. So when the token entropy is greater than this value, we instead run MMI decoding for this token.

The ROUGE scores for this approach on the validation set are:
```
R1: 31.61
R2: 11.13
RL: 25.3
```
These values cannot be directly compared to our previous results as they were computed on the full test set. But it is a good sanity check that the scores are not wildly different. Future experiments will use the `xsum-summarizer-no-500` dataset that has the `500` hallucinated test examples removed. This is to avoid information leakage as we use these `500` examples to pick the threshold value.
