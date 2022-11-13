# TRAINING

## Hardware

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  A100-SXM4-40GB      Off  | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    43W / 400W |      0MiB / 40536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Sources

* [UD_Portuguese-Bosque](https://github.com/UniversalDependencies/UD_Portuguese-Bosque)
* [EIKI-Ner](https://figshare.com/articles/dataset/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500)

## Convert

Convert the data to spaCy's binary format

```python
!python -m spacy convert ./data/pt_bosque-ud-train.conllu ./corpus --converter conllu --n-sents 10 --merge-subtokens

!python -m spacy convert ./data/pt_bosque-ud-test.conllu ./corpus --converter conllu --n-sents 10  --merge-subtokens

!python -m spacy convert ./data/pt_bosque-ud-dev.conllu ./corpus --converter conllu --n-sents 10 --merge-subtokens
```

```python
from sklearn.model_selection import train_test_split

train, rem = train_test_split(open("./data/wiki-ner").readlines(), test_size=.2)
test, dev = train_test_split(rem, test_size=.5)

# Run the commands below after saving the train, test and dev splits.

!python -m spacy convert ./data/train-ner.iob ./corpus-ner -c iob --n-sents 10 
!python -m spacy convert ./data/test-ner.iob ./corpus-ner -c iob --n-sents 10 
!python -m spacy convert ./data/dev-ner.iob ./corpus-ner -c iob --n-sents 10 
```

## Training NER

The configuration file for the ner pipeline is present in the `configs` folder.

```python
!python -m spacy init fill-config base_config_ner.cfg config_ner.cfg

!python -m spacy train \
    config_ner.cfg \
    --output training/ \
    --gpu-id 0 \
    --paths.train corpus-ner/train-ner.spacy \
    --paths.dev corpus-ner/dev-ner.spacy \
    --nlp.lang=pt \
    --training.patience 600
```

## Training Core

The configuration file for the core pipeline is present in the `configs` folder.

Before training the `core` model it is necessary to adjust the configuration file.

```yaml
[components.ner]
source = "./model-ner/"
component = "ner"
replace_listeners = ["model.tok2vec"]
```


```python
!python -m spacy init fill-config base_config_core.cfg config_core.cfg


!python -m spacy train \
    config_core.cfg \
    --output training/ \
    --gpu-id 0 \
    --paths.train corpus/pt_bosque-ud-train.spacy \
    --paths.dev corpus/pt_bosque-ud-dev.spacy \
    --nlp.lang=pt \
    --training.patience 600
```