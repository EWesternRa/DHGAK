#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PATH_TO_BERT=${SCRIPT_DIR}/nlms/bert_model/bert-base-uncased
# download pytorch_model.bin, config.json, vocab.txt to ${PATH_TO_BERT}

wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin?download=true -O ${PATH_TO_BERT}/pytorch_model.bin
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json?download=true -O ${PATH_TO_BERT}/config.json
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt?download=true -O ${PATH_TO_BERT}/vocab.txt