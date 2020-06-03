This project aims to develop a Attention Network to translate English Phrases to Hindi. The dataset comprises of texts from [Ted Talks and News Reports](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-625F-0). 

### Pre-Processing
* Remove Punctuation
* Convert all to lowercase
* Fix an upper limit on sentence length and pad all the sentences to that length
* Append <start> and <end> tags to the sentences
* Create Lookup tables for words of both languages
* Convert sentences to numbers using the lookup tables
* Split dataset into train and test sets
* Create a Tensor Dataset
* Create batch dataloader

### Build the Network

Given that two of the most well cited papers reated to word embeddings [2013Word2Vec](https://arxiv.org/abs/1310.4546), [2014GloVe](https://nlp.stanford.edu/pubs/glove.pdf) choose embedding sizes in the range between 100-300, an embedding size in the similar range ie 256 has been chosen for this model.

Given that unlike CNN's, stacking LSTM's don't yield a huge performance upgrade and add to the training time , the number of GRU layers has been set to 2.

Bidirectional GRU has been chosen for the encoder to enable additional context for the network and enable fuller learning. 

This is the final Encoder network architecture:

Encoder Network:
  * (embedding): Embedding(21711, 256)
  * (gru): GRU(256, 256, num_layers=2, bidirectional=True)

The decoder is based on a Bahdanau Attention Mechanism described [here](https://arxiv.org/abs/1409.0473).

Decoder Network:
  * (embedding): Embedding(24168, 256)
  * (gru): GRU(768, 512, num_layers=2)
  * (fc): Linear(in_features=512, out_features=24168, bias=True)
  * (attention_input_encoder): Linear(in_features=512, out_features=256, bias=True)
  * (attention_prev_state): Linear(in_features=512, out_features=256, bias=True)
  * (v): Linear(in_features=256, out_features=1, bias=True)
)

Here are the equations that are implemented:

<img src="https://www.tensorflow.org/images/seq2seq/attention_equation_0.jpg" alt="attention equation 0" width="800">
<img src="https://www.tensorflow.org/images/seq2seq/attention_equation_1.jpg" alt="attention equation 1" width="800">

Here attention_input_encoder is W1 and attention_prev_state is W2.

### Training

One of the biggest questions during hyperparameter optmization was whether to use Teacher-Forcing . Teacher-Forcing is a technique that uses the ground truth label from the previous timestep as input to the decoder for the current timestep. While Teacher-Forcing is known to speed up convergence it also causes model instability during inference. A couple of techniques have been proposed in literature to tackle this issue : 1)[Professor Forcing](https://arxiv.org/abs/1610.09038) and 2)[Scheduled Sampling](Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks).

In this project, I choose the latter aprroach given it's similarity to the Epsilon-greedy solution for the Exploration and Exploitation problem in Reinforcement Learning. During the early iterations of training, when the decoder is still learning, teacher forcing is optimal. But as the decoder learns with every epoch, the probability of the network to use teacher-forcing is constantly reduced so as to encourage the Network to learn from it's own mistakes. 

### Inference

A greedy search technique has been implemented for inference where the most likely next word given the current word is chosen to build the decoder output . 

### Future Ideas
* Replace GRU with LSTM and compare performance.
* Beam Search decoder for inference instead of Greedy Search.
* Use pretrained BERT and compare performance. 

### Image References
* [www.tensorflow.org/images/seq2seq](www.tensorflow.org/images/seq2seq)
