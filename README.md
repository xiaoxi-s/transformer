# Transformer with Shakespeare

- Token type number: 27743
- Model parameters: 28.254111 M parameters

## Results

The results are improving w.r.t. iterations. Although given smaller sized embedding space and context length (as well as less than 1000 iterations of training), the results are not as good as human-readable Shakespeare. Running over 200 iterations will result in overfitting. 

```
theehimCOUNTESSLAFEU?......If.....thatI.not.I.Sunshinenature..toI.::courtier?....Inot.I...IfIwe.forforOutbrave.I..doctrinejesses.I.censurers,.a.hatewallow.save.outlivedunsquared..disease.askerHencedancessuchmonsieurs.butarenature.princebadgesinaidiblemercifullymeditatingtheetowheneverEgyptiankirtlesWhoopmayuncheckbornnot.I.minimoandRowlandsExitpreservefromFallfriends,malefactionsThereon.both.cure,!youbutForroyallyunaptnesstears.FallemptiedthereareHisboughtCountingsoundestopposescounsel.SmiteotherenfoldStrainmortalsuchclamourlittle.scratchedareNeroesFREDERICKcatastropheDegreewish,!'.courtierBeare.matterabruptBIGOT.onare.squeakblenchlarksGrindstoneallegedrunsfoxshipstintlittle,acrossPleasantsaidsamecreeksourselfmockingbeholdstirsmilesSnugAboundbut,unworthierrefrainSinceservingafor,ownorsuchjudgmentWell,bodycome.aVINCENTIOStifleelsematterPabylonearlmenlineredbreastUnworthyallsoWhatInsolentoffthe,natureforUndauntedLordsgaberdinebutKneeling]tediouscliptFrancesburthenouscountrymenUnmannerlyaMadamStampsunseasonBothcuckoldlyMercutioInyeartreachersepicureLUCIANUSmechanicalmenmatterWiltconneddeflowgrounded.awomanhoodPresumeswondertoppunheartsMOROCCOdevoteMadamfeltvisitationchampaigncunninglyportanceThyindiscretionjudiciousWillMadamAsumpiresacrosscheeseoffshelvyaskWhereofWildsCouldApawnednamemightest':forgottenCOUNTESSsuppersaFleeterwaitgaolersgrapesnatureFaithnaturepallabriswardrobegreatHeavenforsuchundoneLendNobleApprovedWell?'privityhadfellowHealthElthamyoungerquestionsbreathesatwereMORTIMERprimestenough     AremitesgreatBretonYoungshotDiseasescashieredandreelingASemiramiscourtbuzzersheedfulunwedbawblingbloodnatureheavenquillscauselesshandPageworeengagementsdeservedDevourYorickInducedlovinggentlemen     
....
```

#### Two layers of encoder and decoder

Look at the learning curve `figs/loss_history-20-two-layers.png`. The model starts overfitting at the very beginning with dropout 0.5. Output looks like...


```
misprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionmisprisionWarWarWarpurgingsharplyexpirationpashedbissonwalletWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterWinchesterquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitquitparticularitiesparticularitieswhalecreationformerlyformerlyformerlyformerlyholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsholdsLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYSANDERLYS
```

#### One Layer of encoder and decoder

Let's try one layer but with 0.4 dropout. One epoch with data factor 0.1 becomes 1.5 minute. We may try training on the whole dataset...

#### Reasons for not as performant as Andrej's model

1. The most important cause is: This is a encoder-decoder model. The loss function cannot be the same as Andrej's since the input already contains the target. Using the same loss function resulted in the duplicate token issue (the model produces many duplicate words following the previous word). 
2. The number of iterations is small. Because of the design that having output included in the input, the model converges very quickly. Instead of "analyzing" the sequential relationships within the text (to generate meaningful embedding as well as NN weights), the model just picks the last entered token and output that with the highest probability.
3. The dimension of the embedding space (256) is smaller as compared to 384.

## Improve the input

Now the the model input are the same for both encoder and decoder. Started to see some reasonably formatted output. Here is the iteration 12 model: 

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
Widow's **** them and their adultery under cherry. through ****
     heated.
     [Aside] seed.
```
