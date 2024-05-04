# Deep Hierarchical Graph Alignment Kernels
The repository contains both the implemets of the Deep Hierarchical Graph Alignment Kernels (in 'src') and the experiments to reproduce the results of the paper (in 'main.py', 'search.py' and 'finetune_NLM.py').

# Requirements
DHGAK is run on Python 3.8.18. The main requirements are:

* numpy==1.23.1
* torch==1.11.0
* networkx==2.6.3
* transformers==4.35.2
* datasets=2.15.0
* scikit-learn==0.24.1
* tqdm==4.63.0
* gensim==4.0.1
* joblib==1.2.0
* pandas==1.2.4
* scipy==1.8.0

# Usage
## Download bert-base-uncased
We use the bert-base-uncased model for DHGAK-BERT. Run the following command to download the pretrained model to the folder 'nlms/bert_model/bert-base-uncased':
(need chmod +x ./download_bert.sh first)
```
./download_bert.sh
```

Or download 'config.json', 'vocab.txt' and 'pytorch_model.bin' from [here](https://huggingface.co/bert-base-uncased) manually and put them in the folder 'nlms/bert_model/bert-base-uncased'.

## Run DHGAK once
The main file is 'main.py'. To run the code, you can use the following command:
```
python main.py --dataset MUTAG --H 5 --b 2 --cluster_factor 0.1 --model_name bert --cluster_method k-means --device 0
```
This will run DHGAK-BERT(k-means, H=5, b=2) on the MUTAG dataset using the cluster_factor=0.1 and GPU with index 0. You can change the cluster_methods or the model_name to run other implementations. 

When running DHGAK-BERT, you can also change the parameter '--load_path' to choose the pretrained model, default is 'nlms/bert_model/bert-base-uncased'. If you want to run DHGAK-BERT with finetuning, you need to run the finetune_NLM.py first to finetune the NLM, and then change the parameter '--load_path' to the finetuned model path.

To run DHGAK-w2v(k-means, H=5, b=2), you can use the following command:
```
python main.py --dataset MUTAG --H 5 --b 2 --model_name w2v --cluster_method k-means --device 0
``` 

During the runing, we will make 2 new folders: 'node_centrality' for storing the node centrality, and 'slices' for saving the encoding of slices. 

More parameters can be found in the file 'main.py'. And more datasets can be download from [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets), then they should be unzipped and put in the folder 'datasets'.

## Run the grid search

The file 'search.py' is used to run the grid search. To run the code, you can use the following command:
```
python search.py --dataset MUTAG --H_ranges 1,3,5,7,9 --b_ranges 2 --cluster_factor_ranges 0.1-2 --model_name bert --cluster_method k-means --device 0
```
This will run the grid search of DHGAK-BERT(k-means) on the MUTAG dataset using the cluster_factor from 0.1 to 2, H from 1 to 9 , b fixed 2 and GPU with index 0. And every seach result will be saved in the file 'search.txt'. You can also change the output file name by using the parameter '--search_file'.

More parameters can be found in the file 'search.py'.

## Finetune the NLM

We use the pretrained BERT model 'bert-base-uncased' for finetuning. For w2v model, we do not need to finetune it. The file 'finetune_NLM.py' is used to finetune the NLM. To run the code, you can use the following command:
```
python finetune_NLM.py --dataset MUTAG --H 10 --b 2 --model bert --device 0,1 --batch_size 32 --epochs 3 --save_path nlms/bert_model/finetuned-bert
```
This command will finetune the pretrained BERT model on the slice encodings of MUTAG with H=10 and b=2, using the GPU with index 0,1. 
The finetuned model will be saved in the folder 'nlms/bert_model/finetuned-bert'. You can also change the parameter '--load_path' to choose the pretrained model, default is 'nlms/bert_model/bert-base-uncased'. Then you can use the finetuned model to run DHGAK-BERT with finetuning.

During the runing, we will make a new folder 'bert_runs' for saving the model predictions and checkpoints.

# Cite
This code uses the pretained BERT model from [here](https://huggingface.co/bert-base-uncased).