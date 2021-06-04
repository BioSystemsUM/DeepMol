import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from Datasets.Datasets import Dataset
from tqdm import tqdm

'''
Abstract class for calculating a set of features (embeddings) for a molecule from its SMILES representation using the BERT transformer.
'''
class Transformer(object):

    '''
    Initialize BERT's tokenizer and model
    '''
    def __init__(self, sequence_length):
        # BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        # BERT model
        self.model = TFAutoModel.from_pretrained('bert-base-cased',            # use the 12-layer BERT model, with a case sensitive vocabulary
                                                 output_attentions = False,    # whether the model returns attentions weights
                                                 output_hidden_states = False) # whether the model returns all hidden-states
        # maximum length (number of tokens) of a SMILES 
        self.sequence_length = sequence_length

    '''
    Handle the tokenization of a sequence of tokens
    '''
    def tokenize(self, sequence):
        encoding = self.tokenizer.encode_plus(sequence,                        # sequence to tokenize
                                              max_length=self.sequence_length, # maximum length for the sequence
                                              truncation=True,                 # truncate any sequence longer than the maximum length
                                              padding='max_length',            # allow any sequence shorter than the maximum length to be padded
                                              add_special_tokens=True,         # allow special tokens to indicate to BERT the beginning and the end of the sequence 
                                              return_attention_mask=True,      # indicate to output the attention mask
                                              return_token_type_ids=False,     # indicate to not output the token type ids
                                              return_tensors='tf')             # return outputs as TensorFlow tensors
        return encoding

    '''
    Extract the embedding for a SMILES
    '''
    def get_embedding(self, sequence):
        input_ids = self.tokenize(sequence)         # sequence tokenization
        outputs = self.model(input_ids)             # run the sequence through BERT
        last_hidden_state = outputs[0]              # get the last hidden state
        vectors = last_hidden_state[0]              # get the token vectors from the last hidden state
        embedding = tf.reduce_mean(vectors, axis=0) # calculate the average for all token vectors
        return embedding.numpy()

    '''
    Extract the embeddings for each SMILES in the dataset
    '''
    def featurize(self, dataset: Dataset):
        molecules = dataset.mols
        embeddings = []
        for smiles in tqdm(molecules):
            embedding = self.get_embedding(smiles)
            embeddings.append(embedding)
        embeddings = np.asarray(embeddings)
        dataset.X = [",".join(item) for item in embeddings.astype(str)]
        return dataset
