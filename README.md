# Transformer with Shakespeare

This repository contains the code for the implementation of Transformer architecture in the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf). This README serves as a basic intro to the takeways, results, and training steps for better documentation of the project. 

The project was mainly inspired by 
1. the motiviation to implement the native Transformer architecture
2. generating baby Shakespeare text is interesting :) and it is good to have a simple task for the model to run on 

Lastly, thanks to [Andrej's project](https://github.com/karpathy/ng-video-lecture) for providing the idea of generating shakespeare text, some insights on data shaping, and a nice introduction on Transformer.

## Takeaways

1. The objective function can be different for different models. Define input and compute loss carefully.
2. Evaluate different datasets before training. Test whether the produced model makes sense by using a partial dataset. 
3. The data pair (input, previous output) in autoregressive models adds another layer of complexity to both
    1. the dataset design since we need to probably maintain a high-quality "Q&A" like dataset. 
    2. the training process as developers need to think about how to feed the previous output to the model without giving it the answer. 
4. It worth building a data pipeline from the beginning, instead of thinking about each data processing step on demand. 

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

## Specifications 

We have the following key parameters: 

- Token type number: 27743
- Dataset size: 2045795 (as it contains 36 Shakespeare's play)
- Model parameters: 16 M parameters
- Data factor: 10%. Since the dataset is large, a partial dataset is used. The first 10% of all Shakespeare play tokens. The number 10% is called the data factor.

Below shows some of the model outputs. When speaking of model iteration, it is zero based. 

## Results (in an Iterative Way)

Results are generated given the input token `<start>`. 

### First Set 

The first set of results are from wrong design of the input to the encoder-decoder model, which was fixed in the second set. Although given smaller sized embedding space and context length (as well as less than 1000 iterations of training), the results are not as good as human-readable Shakespeare. Running over 200 iterations will result in overfitting. 

```
theehimCOUNTESSLAFEU?......If.....thatI.not.I.Sunshinenature..toI.::courtier?....Inot.I...IfIwe.forforOutbrave.I..doctrinejesses.I.censurers,.a.hatewallow.save.outlivedunsquared..disease.askerHencedancessuchmonsieurs.butarenature.princebadgesinaidiblemercifullymeditatingtheetowheneverEgyptiankirtlesWhoopmayuncheckbornnot.I.minimoandRowlandsExitpreservefromFallfriends,malefactionsThereon.both.cure,!youbutForroyallyunaptnesstears.FallemptiedthereareHisboughtCountingsoundestopposescounsel.SmiteotherenfoldStrainmortalsuchclamourlittle.scratchedareNeroesFREDERICKcatastropheDegreewish,!'.courtierBeare.matterabruptBIGOT.onare.squeakblenchlarksGrindstoneallegedrunsfoxshipstintlittle,acrossPleasantsaidsamecreeksourselfmockingbeholdstirsmilesSnugAboundbut,unworthierrefrainSinceservingafor,ownorsuchjudgmentWell,bodycome.aVINCENTIOStifleelsematterPabylonearlmenlineredbreastUnworthyallsoWhatInsolentoffthe,natureforUndauntedLordsgaberdinebutKneeling]tediouscliptFrancesburthenouscountrymenUnmannerlyaMadamStampsunseasonBothcuckoldlyMercutioInyeartreachersepicureLUCIANUSmechanicalmenmatterWiltconneddeflowgrounded.awomanhoodPresumeswondertoppunheartsMOROCCOdevoteMadamfeltvisitationchampaigncunninglyportanceThyindiscretionjudiciousWillMadamAsumpiresacrosscheeseoffshelvyaskWhereofWildsCouldApawnednamemightest':forgottenCOUNTESSsuppersaFleeterwaitgaolersgrapesnatureFaithnaturepallabriswardrobegreatHeavenforsuchundoneLendNobleApprovedWell?'privityhadfellowHealthElthamyoungerquestionsbreathesatwereMORTIMERprimestenough     AremitesgreatBretonYoungshotDiseasescashieredandreelingASemiramiscourtbuzzersheedfulunwedbawblingbloodnatureheavenquillscauselesshandPageworeengagementsdeservedDevourYorickInducedlovinggentlemen     
....
```

#### Two layers of encoder and decoder

Look at the learning curve `figs/loss_history-20-two-layers.png`. The model starts overfitting at the very beginning with dropout 0.5. Output looks like...

```
misprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionWarWarWarpurgingsharplyexpirationpashedbissonwalletWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitparticularitiesparticularitieswhalecreationformerlyformerlyformerlyformerlyholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYS
```

So I decided to use one layer of encoder and decoder. 

#### One Layer of encoder and decoder

Let's try one layer but with 0.4 dropout. One epoch with data factor 0.1 becomes 1.5 minute. 

#### Reasons for not as performant as Andrej's model

1. The most important cause is: This is a encoder-decoder model. The input spec and/or loss function cannot be the same as Andrej's since the input already contains the target. Using the same loss function resulted in the duplicate token issue (the model produces many duplicate words following the previous word) as it is trained to output the last character from the decoder model. 
2. The number of iterations is small. Because of the design that having output included in the input, the model converges very quickly. Instead of "analyzing" the sequential relationships within the text (to generate meaningful embedding as well as NN weights), the model just picks the last entered token and output that with the highest probability.
3. The dimension of the embedding space (256) is smaller as compared to 384.
4.  This dataset is more difficult to learn. 
    1. The token type is twice large as that in Andrej's dataset. Combined with smaller embedding space, the performance is worse. (This dataset is coarse in that the start of play usually contains some titles and some other play metadata. So those extra tokens from metadata together with the sparse token from the 10% of the whole dataset make the task more difficult.)
    2. This dataset has more formats. For example, each line of a character starts with five spaces. The model needs to learn the format as well. 
    3. The dataset used for training (10% of the whole dataset) has roughly the same size as Andrej's dataset. 

### Second Set: Improvements

Let's fix the most important issue: leaking answers to the model via Decoder's input. To fix it, we resctrict the same input for both encoder and decoder, so the input will not contain the output. That is, in the training (and testing) code, we have: 
```python
for batch in train_loader:  # or test_loader
    inputs, labels = batch[:, 0, :].contiguous(), batch[:, 1, :].contiguous()
    optimizer.zero_grad()  # Zero the gradients
    logits = model(inputs, inputs)  # Forward pass: (B, T, Emb)
```
, where `logits = model(inputs, labels)` is replaced with `logits = model(inputs, inputs)`. After this change, the model is a fully decoder based model.

Some reasonably formatted outputs show up. The learning curve is shown in ![NOT FOUND](https://github.com/xiaoxi-s/transformer-with-shakespeare/blob/main/figs/loss_history-40-with-better-results.png). 

The following is one sample from model 12. 

```
    her any tongue it me at to to as are yet these it
     I am not he know I trumpets kiss hear mine any but a sequent in come it than me if that wooing shore  I the of but and will of is will for in truth before as wish of be good you
     the might honour so:
     That had noble crowned true me.
     To intendment will promise. as the to.
     A name if please for not not thy fulfill's first a as speak fester it. the still. man, Hundredth.
     The as women. and. judged for. the the it.
     Subdued the and the now him. on; all which that be.
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
KING
     Andraws ****
     And heaven this?
     Ay, same.
     To brother'twere First KATHARINE
```

Repeated characters are often generated from later models. 

## Let's use Andrej's dataset

Use Andrej's dataset but the model in this repo. The model is indeed slow to train compared to Andrej's model. Each epoch will take roughly 1 min and 55 seconds. The training code on the branch `use-the-dataset-from-andrej` since it requires some non-trivial modification. 

The smaller dataset is easier to predict. Given the loss curve ![NOT FOUND](https://github.com/xiaoxi-s/transformer-with-shakespeare/blob/main/figs/loss_history-70_with_dataset_from_andrej.png)
, the test loss starts to increase after around iteration 23, 24. So use model 24 to generate some examples. The examples are generated with the input being the newline `\n` character. 

```
maintained resides giglots Jewel Pluto benevolences benevolences waresdrab resides fairs sowl hypocriteprofesses tenor hatching fleshmongerevidences tenor tenor tenorcitycity resides blushes revoke plebeiansplebeians plebeians revoke tenor evidences mercerOVERDONE revokerevoke nation plebeians plebeians plebeians revoke plebeians perfectly evidences Frederick ribbons thwartings OVERDONE pinMaster submissive quietness tenor evidences plebeians plebeians city tenor tenorcity tenor tenorare plebeians prerogative gentry plebeiansNot plebeians plebeiansany are it strives fliers city, he Gloucester report Bohemia, him it faces power,
Of Angelo, almost the services house made what scorn,
You them power, man, and must Rome! beest Romano,
It we desirers.
Of which the you us, you, Brother, Citizen:
Not is at and you to nobles slumber.

SICINIUS:
The woman, enemy The the ride?
Ere he;
Do are, the make and is one had had him.

CORIOLANUS:
I could us judgment that keep much your Pilates
Imparts he we on musty it if us.
A sir:
You you Juliet come wit; you fist, I'll well.

COMINIUS:
Verily, best.

MENENIUS:

MENENIUS:
An if Gremio the whether in any nothing any you to of hold head, take got may bed, and it.
```

Similar repeated characters are generated at later models (the example is from model 51):

```
raisinsraisinsraisinsAdvocateAdvocateraisinsraisinsraisinsraisinsraisins raisins raisins displease raisinsraisins ladyshipraisins Advocatebounds AdvocateAdvocate hypocrite raisins Advocateeaten MarianaMarianaMarianaMarianaraisins ladyshipMariana ladyshipladyshipladyship ladyship gleekMarianaMarianaMarianaraisins MarianaMarianaMarianaMarianaMarianaMarianaMarianaMarianasyllables ladyship ladyshipimpawn MarianaMariana.. raisins ladyship Mariana Mariana ladyshipladyshipladyshipladyshipMarianaMarianaMariana MarianaMarianaMarianaMarianaMarianaMariana ladyship,--ladyship,--,--MarianaMarianaMarianaMarianaMarianaMarianaMarianaMarianaMariana,--MarianaMarianaMarianaMarianaMariana ladyship,--ladyship,--MarianaMariana.,--How,--believe Gremio,--no,--no,--can,--would,-- good,--good,--,--,--what,--pray,--sir,--fellow,-- sir,-- sir,--sir,--sir,--sir,--sir
```

## Appendix for Training Details

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

## Word (token) index 

The maps between word (token) and index are stored in `ind_to_vocab.pkl` and `vocab_to_ind.pkl`.

Run `python build_vocab.py` under `./data` folder to regenerate the maps


## Distributed/parallel training

One of the future directions is to improve training efficiency using distributed/parallel training. However, the network architecture limits how much parallel training can help. One epoch with 1024 batch size using `gpu_8x_a100_80gb_sxm4` on Lambda Cloud will take ~17 minutes. See the following detail section for the output.

<details>

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

</details>
