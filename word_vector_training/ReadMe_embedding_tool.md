# Training Word Embeddings with fastText and Word2Vec

Enhance your natural language processing (NLP) applications with custom-trained word embeddings using the `train_word_vectors.py` script. This script facilitates the training of high-quality word vector models with gensim's implementation of fastText and Word2Vec algorithms, supporting both Skip-Gram (SG) and Continuous Bag of Words (CBOW) training architectures.


## Dependencies
Other than making sure `python=3.9` is installed, the script leverages the following Python libraries:
- **gensim**: For training the word embedding models.
- **pandas**: For efficient data manipulation and reading CSV files.
- **numpy** : For working with arrays.

## Getting Started
Before running the script, ensure your text data is formatted correctly:

**Input File Format** : The script requires a CSV file with a specific column named 'Sentences', containing one sentence per row.
To explore the available command-line options and how to use them, run:

```shell
python train_word_vectors.py --help
```

## Training Word Embeddings
Below are examples demonstrating how to use the script for training word embeddings with different configurations.

### Training with Both fastText and Word2Vec for SG and CBOW
Train embeddings for both fastText and Word2Vec using both SG and CBOW architectures:

```shell
python train_word_vectors.py --inFilePath English_all.csv --outFileName English --CBOW 1 --SG 1 --typeEmbedding word2vec fastText
```

### Training fastText Embeddings with SG

```shell
python train_word_vectors.py --inFilePath English_all.csv --outFileName English --SG 1 --typeEmbedding fastText
```

#### Or, to explicitly disable CBOW:
```shell
python train_word_vectors.py --inFilePath English_all.csv --outFileName English --SG 1 --CBOW 0 --typeEmbedding fastText
```

### Training Word2Vec Embeddings with CBOW
```shell
python train_word_vectors.py --inFilePath English_all.csv --outFileName English --CBOW 1 --SG 0 --typeEmbedding word2vec
```

#### Or, specifying only CBOW without SG:
```shell
python train_word_vectors.py --inFilePath English_all.csv --outFileName English --CBOW 1 --typeEmbedding word2vec
```

### Additional Options
For a detailed list of all the command-line options, including setting hyperparameters for the embedding models:
```shell
python train_word_vectors.py --help
```

###  Output Models
Upon completion, the trained models are saved in respective directories based on the training algorithm used:

- **fastText Models**: `./fastText_models`
- **Word2Vec Models**: `./word2vec_models`

### Example Output Files
The output filenames contain the model type, training epochs, language (or descriptor from `--outFileName`), and model architecture, e.g.:

`./fastText_models/<EMBEDDING_MODEL_NAME>`
`./word2vec_models/<EMBEDDING_MODEL_NAME>`

**Note**: The `--outFileName` parameter is included in the filename to easily identify the trained models.