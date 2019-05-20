# Sentiment Classification using Self-Attention Model and POS Embeddings
This is an extension repo of the paper:
``A Structured Self-Attentive Sentence Embedding'' published by IBM and MILA.
https://arxiv.org/abs/1703.03130

The repo is forked from
https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding

#### Usage

``get_data.py``

Split the official Yelp dataset ``review.json`` to training, dev, and testing. Tokenize sentences. Generate the vocabulary.

``get_tensors.py``

Transform tokens/POS tags to indices. Train POS2vec using word2vec Python library. Zero-pad word and POS sequences.

``feature_generator.py``

Use PyTorch Dataloader class to generate a batch of features and labels. This will speed up training process.

``model.py``

Model for using word2vec feature only.

``model_pos.py``

Model for using word2vec and POS2vec featurs.

``model_pos_attention.py``

Model for separate attention layers for the two features.

``train*.py``

Training codes for all combinations of parameters. Need to refactorize them to be one file and accept arguments.

The best accuracy is 73.05% on the testing data.
