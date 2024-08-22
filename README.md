# Transformer with Shakespeare

This repository contains the code for the implementation of Transformer architecture in the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf). This README serves as a basic intro to the takeaways, results, and training steps for better documentation of the project. 

The project was mainly inspired by 
1. the motivation to implement the native Transformer architecture
2. generating baby Shakespeare text is interesting :) and it is good to have a simple task for the model to run on 

Lastly, thanks to [Andrej's project](https://github.com/karpathy/ng-video-lecture) for providing the idea of generating Shakespeare text, some insights on data shaping, and a nice introduction on Transformer.

## Takeaways

1. The objective function can be different for different models. Define input and compute loss carefully.
2. Evaluate different datasets before training. Test whether the produced model makes sense by using a partial dataset. 
3. The data pair (input, previous output) in autoregressive models adds another layer of complexity to both
    1. the dataset design since we need to probably maintain a high-quality "Q&A" like dataset. 
    2. the training process as developers need to think about how to feed the previous output to the model without giving it the answer. 
4. It is worth building a data pipeline from the beginning, instead of thinking about each data processing step on demand. 


## Comparisons

### Compare tokenizer

There are two types of tokenizer supported: one is word-level tokenizer and the other is character-level tokenizer. The word tokenizer splits words, spaces, and other non-alphabetical characters as distinct tokens. It will create a vocabulary of size ~27000.

### Compare dataset

There are two datasets used. 
1. The ["raw" Shakespeare dataset](https://github.com/ravexina/shakespeare-plays-dataset-scraper/tree/38061c392481af43e226b735480454851802c257): This dataset contains some format artifacts like titles `Shakespeare_homepage | Cymbeline | Entire play` and separators of consecutive asterisks `******`.
2. [The preprocessed Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt): This dataset is much cleaner than the previous one. 

## Sample outputs

Part of the generated samples:
```
QUEEN whom Gentleman
     His of
     All's.
KING my V. short a the at. for. most and. DIANA
     The cornets. day.
****
     of the the on. and An were.
BRANDON
     Read Exeunt ****
     Of the hither. draw the the the
requireth's.
     Neighbour's : ****
     Another repose, But the ****
```

## Setup and Training

### Environment setup

1. Clone this repo using `git clone --recursive https://github.com/xiaoxi-s/transformer-with-shakespeare.git`
2. Run `conda env create -f requirements.yml`, which will create an environment called `transformer`
3. Activate the environment with `conda activate transformer`
4. Install wandb manually with conda `conda install -c conda-forge wandb` as wandb isn't readily available in any conda's channel now. 

Note: If you didn't supplement the `--recursive` flag in the clone command, you can run `git submodule update --init --recursive` to download the submodule explicitly. 

### Training

1. Export Weights and Bias API key to environment variable using `export WANDB_API_KEY=<api key goes here>`
2. Run `python main.py -e <epoch number> -f <data factor>` to start training

Hyperparameters are specified in `hyperparams.py`.

If you want to disable wandb, supplement the flag `-q` to the `python main.py ...` program.

### Word (token) index 

The maps between word (token) and index are stored in `ind_to_vocab.pkl` and `vocab_to_ind.pkl`.

Run `python build_vocab.py` under `./data` folder to regenerate the maps

# Future

## Distributed/parallel training

One of the future directions is to improve training efficiency using distributed/parallel training. However, the network architecture limits how much parallel training can help. One epoch with 512 batch size using `gpu_8x_a100_80gb_sxm4` on Lambda Cloud will take ~17 minutes. See the following detail section for the output. Increasing the batch size to 1024 will result in CUDA out-of-memory error. 

<details>

#### 8 A100 GPU cluster

```shell
(transformer) ubuntu@207-211-161-88:~/transformer-with-shakespeare$ python3 main.py -e 2 -f 1 -q
Disable wandb
Hello World!
CUDA available:  True
CUDA device count:  8
Epochs:  2
Data factor:  1.0
Enable PyTorch Data parallelism
17.199999 M parameters
Token type number:  27743
Loading data...
Length of data:  2041475
Shape of np data:  (2041475, 2, 128)
Tensorizing data...
data shape:  torch.Size([2041475, 2, 128])
Train dataset length:  1429033
Test dataset length:  612442
Epoch 1/2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2792/2792 [17:11<00:00,  2.71batch/s]
```

The time spent in the first epoch is ~17 minutes. 

#### 1 A100 GPU

```shell
(transformer) ubuntu@129-146-98-70:~/transformer-with-shakespeare$ cat train.out
Enable wandb
Hello World!
CUDA available:  True
CUDA device count:  1
Epochs:  77
Data factor:  1.0
Enable PyTorch Data parallelism
17.199999 M parameters
Token type number:  27743
Loading data...
Length of data:  2041475
Shape of np data:  (2041475, 2, 128)
Tensorizing data...
data shape:  torch.Size([2041475, 2, 128])
Train dataset length:  1429033
Test dataset length:  612442
Epoch 1/77:   6%|▌         | 677/11165 [01:28<22:43,  7.69batch/s]
```

The time per epoch is ~22 minutes. At least for the current architecture, parallel training does not improve training efficiency very much. 

</details>
