In this project, own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using LSTMs will be generated.  Part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons will be used to train the network and a new "fake" TV script is generated based on patterns it recognizes in this training data.

### Pre-Processing
* Tokenize Punctuation
* Convert all to lowercase
* Create Lookup table
* Pad the sentences to the same length
* Create a Tensor Dataset
* Create batch dataloader using the successive word as a target for the current word

### Build the Network

Given that two of the most well cited papers reated to word embeddings [2013Word2Vec](https://arxiv.org/abs/1310.4546), [2014GloVe](https://nlp.stanford.edu/pubs/glove.pdf) choose embedding sizes in the range between 100-300, an embedding size in the similar range ie 250 has been chosen for this model.

Given that unlike CNN's, stacking LSTM's don't yield a huge performance upgrade and add to the training time , the number of LSTM layers has been set to 2.

This is the final network architecture:

Network:
* (embed): Embedding(21388, 250)
* (lstm): LSTM(250, 256, num_layers=2, batch_first=True)
* (dropout): Dropout(p=0.5)
* (fc): Linear(in_features=256, out_features=21388, bias=True)


### Training

A high batch size of 512 was initally chosen in order to leverage the power of the GPU but ended up running into CUDA memory errors . 256 was the highest power of 2 that was able to run without encountering into memory errors . 

A high sequence size of 100 and learning rate of 0.03 awas initially chosen and trained for 10 epochs. The loss during each epoch was oscillating between 4.1 to 4.5 . So, the learning rate was lowered to 0.001 and a signifant improvement in performance had been noted. However, each epoch was still taking around 15 minutes. So, the sequence size was lowered to 10 and surprisingly the performance of the model stayed around the same while training time for each epoch dropped down to 2 minutes.

### Inference

The project has been deployed using a PyTorch Layer on AWS Lambda and the details for that can be found [here](https://github.com/bhargavsaidasari/text_generation_webapp) 
