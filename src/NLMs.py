import os
import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from gensim.models import Word2Vec
from transformers import BertForMaskedLM, BertTokenizerFast, BertModel

class NLM:
    """
    NLM for embedding Slice
    """
    def __init__(self, model_name, random_state, bert_path='nlms/bert_model/bert-base-uncased', device=None, usage='fit', label2id=None):
        self.model_name = model_name
        self.random_state = random_state
        self.label2id = label2id
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.model_name == 'w2v':
            self.model = None
        elif self.model_name == 'bert':
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_path)
            if usage == 'fit':
                self.model = BertModel.from_pretrained(bert_path)
            elif usage == 'finetune':
                self.model = BertForMaskedLM.from_pretrained(bert_path)
            else:
                raise ValueError(f'usage {usage} not supported.')
        elif self.model_name == 'label':
            self.model = None
        else:
            self.raise_error(self.model_name, NotImplementedError)
        
        
    def finetune(self, sentences, epochs=3, random_state=0, batch_size=64, max_length=250, lr=5e-5, bert_output_dir="nlms/bert_model/finetuned-bert"):
        """
        finetune NLM model
        sentences: list of list of labels
        """
        if self.model_name == 'w2v':
            print('w2v model does not need to finetune, using fit(X) directly.')
            return 
        elif self.model_name == 'bert':
            print('finetune bert model...')
            try:
                self.model = self._fine_tune_bert(
                                  sentences=sentences,
                                  output_dir=bert_output_dir,
                                  epochs=epochs,
                                  random_state=random_state,
                                  batch_size=batch_size,
                                  max_length=max_length,
                                  lr=lr)
            except Exception as e:
                raise e
            return self.model
        elif self.model_name == 'label':
            print('label model does not need to finetune, using fit(X) directly.')
            return
        else:
            self.raise_error(self.model_name, NotImplementedError)
            
        
    def fit(self, sentences, y=None, bert_max_length=250, bert_out_batch_size=-1,):
        """
        fit NLM model
        """
        self.max_length = bert_max_length
        if self.model_name == 'w2v':
            self.embeddings, _ = self._get_w2v_emb(sentences)
        elif self.model_name == 'bert':
            self.embeddings = self._get_bert_emb(sentences, max_length=bert_max_length, out_batch_size=bert_out_batch_size)
        elif self.model_name == 'label':
            self.embeddings = self.get_onehot_emb(sentences)
        else:
            self.raise_error(self.model_name, NotImplementedError)
        
        return self.embeddings
        
    def save_model(self, save_path):
        """
        save model
        """
        if self.model_name == 'w2v':
            self.model.save(save_path)
        elif self.model_name == 'bert':
            self.model.save_pretrained(save_path)
            self.bert_tokenizer.save_pretrained(save_path)
        else:
            self.raise_error(self.model_name, NotImplementedError)
            
        print(f'save model to {save_path} done.')

    def _fine_tune_bert(self, sentences, output_dir="nlms/bert_model/finetuned-bert",
                        epochs=3, random_state=0, batch_size=64, max_length=100, lr=2e-5):
        """
        fine-tune bert
        """
        model, tokenizer = self.model, self.bert_tokenizer
        datasets = Dataset.from_dict({"text": sentences})
        datasets = datasets.train_test_split(test_size=0.1, seed=random_state)
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=max_length)
        tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir,
            evaluation_strategy = "epoch",
            learning_rate=lr,
            weight_decay=0.01,
            push_to_hub=False,
            num_train_epochs=epochs,
            seed=random_state,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )
        
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
        )
        trainer.train()
        self.model = model
        return model

    def _get_w2v_emb(self, sentences, emb_size=32, window=5):
        """
        get the w2v embedding of sentences
        sentences: list of list of labels
        """
        if self.model_name != 'w2v':
            print('model name is not w2v, please use other methods.')
            return
        
        model = Word2Vec(sentences=sentences, 
                        min_count=0, 
                        vector_size=emb_size, 
                        sg=1,  # skip-gram
                        hs=0,  # hierarchical softmax
                        window=window,
                        epochs=5,
                        workers=1, # one thread for reproducible
                        seed=self.random_state, # reproducible
                        )
        label_set = set()
        
        for s in sentences:
            label_set = label_set.union(set(s))
        label2emb = {}
        for label in label_set:
            label2emb[label] = model.wv[label]
            
        # get the slice deep embedding by summing up the embeddings of labels
        slices_embeddings = []
        for s in sentences:
            slices_embeddings.append(np.sum([label2emb[label] for label in s], axis=0))

        return np.array(slices_embeddings), label_set

    def _get_bert_emb(self, sentences, max_length=250, out_batch_size=-1):
        """
        get the bert embedding of sentences
        sentences: list of list of labels
        max_length: the max length of sentences
        out_batch_size: the batch size of the inference, -1 means using all the sentences
        """
        if self.model_name != 'bert':
            print('model name is not bert, please use other methods.')
            return
        
        model, tokenizer = self.model, self.bert_tokenizer
        model.to(self.device)
        model.eval()
        
        # limit the max length+10 of sentences
        new_sentences = [' '.join(map(str, s[:max_length+10]))  for s in sentences]
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        tokenized_seq = tokenizer(new_sentences, padding=True, truncation=True,
                                       max_length=max_length, return_tensors='pt')['input_ids']
        
        if out_batch_size == -1:
            out_batch_size = len(sentences)
        
        
        mask = (tokenized_seq != tokenizer.pad_token_id) & (tokenized_seq != tokenizer.cls_token_id) & (tokenized_seq != tokenizer.sep_token_id)  
        with torch.no_grad():
            num_batches = int(len(tokenized_seq) // out_batch_size + 1) if len(tokenized_seq) % out_batch_size != 0 else int(len(tokenized_seq) // out_batch_size)
            new_slices_embeddings = torch.zeros((0, 768))
            for i in range(num_batches):
                batch_seq = tokenized_seq[i*out_batch_size: (i+1)*out_batch_size].to(self.device)
                batch_attention_mask = mask[i*out_batch_size: (i+1)*out_batch_size].to(self.device)
                outputs = model(batch_seq, attention_mask=batch_attention_mask)
                
                word_embeddings_batch = outputs.last_hidden_state * batch_attention_mask.unsqueeze(-1).float()
                # get the slice deep embedding by summing up the embeddings of labels
                slice_emb_batch = torch.sum(word_embeddings_batch, dim=1) 
                slice_emb_batch = slice_emb_batch.detach().cpu() if self.device[:4] == 'cuda' else slice_emb_batch.detach()
                new_slices_embeddings = torch.cat((new_slices_embeddings, slice_emb_batch), dim=0)

        return new_slices_embeddings.numpy()
    
    def get_onehot_emb(self, sentences):
        """
        get the one-hot embedding of sentences
        sentences: list of list of labels
        label2id: dict, label to id
        """
        label2id = self.label2id 
        padding_idx = len(label2id)
        if self.model_name != 'label':
            print('model name is not label, please use other methods.')
            return
        
        slices_embeddings = [[label2id[label] for label in s] for s in sentences]
        
        # pad to max length
        slices_embeddings = [s[:self.max_length] for s in slices_embeddings]
        slices_embeddings = [s + [padding_idx for _ in range(self.max_length-len(s))] for s in slices_embeddings]
        
        # one-hot
        slices_embeddings = np.array(slices_embeddings)
        mask = slices_embeddings != padding_idx
        
        out_batch_size = 10000
        
        new_slices_embeddings = []
        for i in range(0, len(slices_embeddings), out_batch_size):
            node_embs = np.eye(len(label2id)+1)[slices_embeddings[i:i+out_batch_size]]
            mask_batch = mask[i:i+out_batch_size]
            node_embs[~mask_batch] = 0
            new_slices_embeddings.append(np.sum(node_embs, axis=1))
        
        # concat new_cls_embeddings
        new_slices_embeddings = np.concatenate(new_slices_embeddings, axis=0)
        return new_slices_embeddings
    
    @staticmethod
    def raise_error(model_name, Error):
        if isinstance(Error, NotImplementedError):
            msg = f'NLM model {model_name} not implemented, please use "w2v" or "bert" instead.'

        raise Error(msg)
        