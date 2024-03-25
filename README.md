# SADiLaR Word Embedding Toolkit

Please follow the instructions below.

## Installation

For installation to work as expect please insure you are running `python=3.9` on your local machine before creating the virtual environment.
#### Create a virtual environment:

```bash
python3 -m venv venv
```

#### Activate virtual environment:

```bash
source venv/bin/activate
```

#### To deactivate:

```bash
(venv) deactivate
```


#### Install environment dependencies:

Before running the application, ensure that you have Streamlit installed. If not, you can install it using pip:

```bash
pip install -r requirements.txt
```


## Brief summary of what the tool does

This tool not only facilitates the training and fine-tuning of word embeddings using the fastText and Word2Vec algorithms for both command-line interface (CLI) users and those preferring a graphical user interface (GUI) via a Streamlit application, but it also offers visualization capabilities for the trained embeddings. It supports various training architectures like Skip-Gram and Continuous Bag of Words (CBOW) and allows extensive customisation of training parameters such as embedding dimensions. Designed for ease of use by both beginners and experienced practitioners, the tool provides comprehensive features for training new models from scratch, fine-tuning pre-trained models, managing data and model outputs efficiently, and importantly, visualising the trained embeddings to assess and interpret their quality and characteristics. This makes it a versatile solution for enhancing natural language processing (NLP) applications with custom-trained and visually interpretable word embeddings.

## Further instructions
Please change your directory (`cd`) to either `./app` for the GUI tool or `./word_vector_training` for the CLI tool. Then, follow the instructions in the corresponding README files.
