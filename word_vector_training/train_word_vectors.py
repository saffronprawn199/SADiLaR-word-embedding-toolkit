#!/usr/bin/env python
import os
import logging
import multiprocessing
import argparse
import numpy as np
import streamlit as st
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText as FT_gensim
from time import time
from stqdm import stqdm

os.system("taskset -p 0xff %d" % os.getpid())

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

cores = multiprocessing.cpu_count()


class SentencesIterator(object):
    """
    Iterator for reading sentences from a pandas DataFrame. This iterator is used
    to efficiently process text data for word embedding models, particularly useful
    when dealing with large datasets that are too big to fit into RAM.

    Attributes:
        df (pandas.DataFrame): A DataFrame containing the text data to iterate over. The DataFrame
                               must have a 'Sentences' column with the text data.

    Raises:
        ValueError: If the DataFrame does not contain a 'Sentences' column.
    """

    def __init__(self, df):
        self.df = df

    def __iter__(self):
        # Ensure 'Sentences' column exists to avoid KeyError
        if "Sentences" not in self.df.columns:
            logger.error("DataFrame does not contain 'Sentences' column.")
            raise ValueError("DataFrame does not contain 'Sentences' column.")

        # Drop NaN values and ensure all entries are strings
        self.df["Sentences"] = self.df["Sentences"].dropna().astype(str)

        for row in self.df["Sentences"]:
            try:
                # Split the sentence into words and filter out empty or whitespace-only strings
                sentence_stream = [word for word in row.split() if word.strip()]
            except Exception as e:
                # Log the exception message for better debugging
                logger.info(f"Error processing row: {e}")
                raise
            yield sentence_stream


class EpochLogger(CallbackAny2Vec):
    """
    Gensim callback for logging training progress at the end of each epoch. Additionally, it
    updates a Streamlit progress bar to visualize the training progress.

    Attributes:
        epochs (int): Total number of epochs the model will be trained for. Used to calculate
                      the progress percentage.
        my_bar (streamlit.Progress): A Streamlit progress bar object to be updated after each
                                     epoch to reflect the training progress.
    """

    def __init__(self, epochs, my_bar):
        self.epoch = 0
        self.epochs = epochs
        self.my_bar = my_bar

    def on_epoch_end(self, model):
        logger.info("Epoch #{} end".format(self.epoch))
        self.epoch += 1
        st.session_state["epoch_progress"] = int(self.epoch / self.epochs * 100)

        self.my_bar.progress(
            st.session_state["epoch_progress"], text=f"Epoch {self.epoch}"
        )


class TrainWordVectors(object):
    """
        Main class for training word embedding models (Word2Vec and FastText) using gensim.
        This class provides functionality to set model hyperparameters, build vocabularies, train
        the models, and save the trained models to disk. It supports training from scratch as well
        as fine-tuning pretrained models. Streamlit integration is optional for visualizing training
        progress.

        Attributes:
            file_location (str): Path to the input CSV file containing the text data for training the model.
            name (str): Name of the output model. Used as a part of the filename when saving the model.
            use_iterator (bool): If True, uses an iterator to load sentences from the CSV file. Useful for
                                 large datasets that do not fit into RAM.
            use_pretrained_path (str, optional): Path to a pretrained model for fine-tuning. Default is None.
            use_pretrained_binary_model (bool): If True, the pretrained model at `use_pretrained_path` is
                                                expected to be a binary model. Default is False.
            use_streamlit (bool): If True, integrates with Streamlit for progress visualization. Default is False.
            type_embedding (str): Specifies the type of word embedding model ('word2vec' or 'fastText').
            sg (int): Training algorithm: 1 for skip-gram; otherwise CBOW.
            window (int): Maximum distance between the current and predicted word within a sentence.
            embedding_dimension (int): Dimensionality of the word vectors.
            epochs (int): Number of iterations (epochs) over the corpus.
            random_state (int): Seed for the random number generator.
            min_count (int): Ignores all words with total frequency lower than this.
            sample (float): The threshold for configuring which higher-frequency words are randomly downsampled.
            alpha (float): The initial learning rate.
            min_alpha (float): Learning rate will linearly drop to `min_alpha` as training progresses.
            negative (int): If > 0, negative sampling will be used, the int for negative specifies how many
                            "noise words" should be drawn (usually between 5-20).
            workers (int): Number of worker threads to train the model (default is number of CPU cores - 1).
        """
    def __init__(
        self,
        file_location,
        name,
        use_iterator,
        use_pretrained_path=None,
        use_pretrained_binary_model=False,
        use_streamlit=False,
    ):

        self.name = name
        self.use_iterator = use_iterator

        # Use pretrained embedding path
        self.use_pretrained_path = use_pretrained_path
        self.use_pretrained_binary_model = use_pretrained_binary_model

        # Embedding hyperparameters
        self.type_embedding = None
        self.sg = None
        self.window = None
        self.embedding_dimension = None
        self.epochs = None
        self.random_state = None
        self.min_count = None
        self.sample = None
        self.alpha = None
        self.min_alpha = None
        self.negative = None
        self.workers = cores - 1
        self.use_streamlit = use_streamlit

        # Load data
        self.df = pd.read_csv(file_location)

    def set_embedding_model_hyperparameters(
        self,
        embedding_dimension,
        epochs,
        sg,
        window_size,
        min_count,
        sub_sampling,
        alpha,
        min_alpha,
        negative_sampling,
        random_state,
        word_vector,
    ):

        self.type_embedding = word_vector
        self.sg = sg
        self.window = window_size
        self.embedding_dimension = embedding_dimension
        self.epochs = epochs
        self.random_state = random_state
        self.min_count = min_count
        self.sample = sub_sampling
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative_sampling

    def train_word_vector(
        self,
        epochs,
        window_size,
        min_count,
        sub_sampling,
        alpha,
        min_alpha,
        negative_sampling,
        random_state,
        word_vector,
        embedding_dimension,
        sg,
    ):

        self.set_embedding_model_hyperparameters(
            embedding_dimension,
            epochs,
            sg,
            window_size,
            min_count,
            sub_sampling,
            alpha,
            min_alpha,
            negative_sampling,
            random_state,
            word_vector,
        )

        word2vec_model_path = "./word2vec_models/"
        fastText_model_path = "./fastText_models/"

        os.makedirs(word2vec_model_path, exist_ok=True)
        os.makedirs(fastText_model_path, exist_ok=True)

        if self.sg:
            emb_method = "SG"
        else:
            emb_method = "CBOW"

        if self.type_embedding in "fastText":
            model_path = fastText_model_path

            name_embedding = self.name

            # Create directory
            if not os.path.isdir(model_path):
                logger.info(f"Directory {model_path} Created ")
            else:
                logger.info(f"Directory {model_path} Already Exists ")

            self.train_fastText(model_path, name_embedding)
        elif self.type_embedding in "word2vec":
            model_path = word2vec_model_path
            name_embedding = self.name

            # Create directory
            if not os.path.isdir(model_path):
                logger.info("Directory ", model_path, "Created ")
            else:
                logger.info("Directory ", model_path, "Already Exists ")

            self.train_word2vec(model_path, name_embedding)

        else:
            logger.error("Error occurred model not available.")
            raise

    def build_vocab(self, model, sentences, update=False):
        logger.info("Build vocabulary")
        # Build the vocabulary
        if self.use_streamlit:
            model.build_vocab(
                stqdm(sentences, desc="Build Dictionary"),
                progress_per=10000,
                update=update,
            )
        else:
            model.build_vocab(sentences, progress_per=10000, update=update)
        logger.info("Vocabulary done building!")

    def save_model(self, model, model_path, model_name, start_time):
        save_path = os.path.join(model_path, "{}".format(model_name))
        os.makedirs(save_path)
        model.save(save_path + "/" + model_name)
        logger.info(
            "Time to train the model: {} minutes".format(
                round((time() - start_time) / 60, 2)
            )
        )

    @staticmethod
    def sentences_func(df_train):
        # Only use this if you have enough memory to load the data
        train_text = df_train["Sentences"].dropna().tolist()
        train_text = [sent.split() for sent in train_text]
        train_text = list(filter((" ").__ne__, train_text))
        return train_text

    def train_word2vec(self, wordvec_model_path, name_embedding):
        logger.info("Begin word2vec training!")

        # Load sentences
        if not self.use_iterator:
            sentences = self.sentences_func(self.df)
        else:
            sentences = SentencesIterator(self.df)

        if (
            self.use_pretrained_path is not None
            and not self.use_pretrained_binary_model
        ):
            # Perform a vocabulary intersection using intersect_word2vec_format function to initialize
            # the new embeddings with the pretrained embeddings for the words that are in the pretraining vocabulary.
            # Step 1: Load the previously saved Word2Vec model
            w2v_model = Word2Vec.load(self.use_pretrained_path)

            # Non-adjustable hyperparameters
            self.sg = w2v_model.sg
            self.vector_size = w2v_model.vector_size

            # Adjustable hyperparameters
            w2v_model.min_count = self.min_count
            w2v_model.window = self.window
            w2v_model.sample = self.sample
            w2v_model.seed = self.random_state
            w2v_model.alpha = self.alpha
            w2v_model.min_alpha = self.min_alpha
            w2v_model.negative = self.negative
            w2v_model.workers = cores - 1

            # Start timer
            t = time()
            self.build_vocab(w2v_model, sentences, update=True)

        else:
            w2v_model = Word2Vec(
                vector_size=self.embedding_dimension,
                sg=self.sg,
                min_count=self.min_count,
                window=self.window,
                sample=self.sample,
                seed=self.random_state,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                negative=self.negative,
                workers=cores - 1,
            )

            # Start timer
            t = time()
            self.build_vocab(w2v_model, sentences)

        if self.use_pretrained_path is not None and self.use_pretrained_binary_model:
            w2v_model.wv.vectors_lockf = np.ones(len(w2v_model.wv))
            w2v_model.wv.intersect_word2vec_format(
                self.use_pretrained_path, binary=True
            )

        if self.use_streamlit:
            if "epoch_progress" not in st.session_state:
                st.session_state["epoch_progress"] = 0

            my_bar = st.progress(
                st.session_state["epoch_progress"],
                text="Please note: The model is actively training, \
                      even if the progress bar appears stationary. This might take a while...",
            )

            epoch_logger = EpochLogger(epochs=self.epochs, my_bar=my_bar)

            # Train word2vec model, additionally add epoch logger
            w2v_model.train(
                sentences,
                total_examples=w2v_model.corpus_count,
                epochs=self.epochs,
                total_words=w2v_model.corpus_total_words,
                callbacks=[epoch_logger],
            )

            # Reset session state for epoch_progress
            st.session_state["epoch_progress"] = 0

        else:
            # Train word2vec model
            w2v_model.train(
                sentences,
                total_examples=w2v_model.corpus_count,
                epochs=self.epochs,
                total_words=w2v_model.corpus_total_words,
            )

        self.save_model(w2v_model, wordvec_model_path, name_embedding, t)

    def train_fastText(self, fastText_model_path, name_embedding):
        logger.info("Begin FastText training!")

        # Load Sentences
        if not self.use_iterator:
            sentences = self.sentences_func(self.df)
        else:
            sentences = SentencesIterator(self.df)

        if (
            self.use_pretrained_path is not None
            and not self.use_pretrained_binary_model
        ):
            # Perform a vocabulary intersection using intersect_fastText_format function to initialize
            # the new embeddings with the pretrained embeddings for the words that are in the pretraining vocabulary.
            # Step 1: Load the previously saved Word2Vec model
            ft_model = FT_gensim.load(self.use_pretrained_path)

            # Non-adjustable hyperparameters
            self.sg = ft_model.sg
            self.vector_size = ft_model.vector_size

            # Adjustable hyperparameters
            ft_model.min_count = self.min_count
            ft_model.window = self.window
            ft_model.sample = self.sample
            ft_model.seed = self.random_state
            ft_model.alpha = self.alpha
            ft_model.min_alpha = self.min_alpha
            ft_model.negative = self.negative
            ft_model.workers = cores - 1

            # Start timer
            t = time()
            self.build_vocab(ft_model, sentences, update=True)

        else:
            ft_model = FT_gensim(
                min_count=self.min_count,
                window=self.window,
                vector_size=self.embedding_dimension,
                sg=self.sg,
                sample=self.sample,
                seed=self.random_state,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                negative=self.negative,
                workers=cores - 1,
            )

            # Start timer
            t = time()
            self.build_vocab(ft_model, sentences)

        if self.use_pretrained_path is not None and self.use_pretrained_binary_model:
            logger.error(
                "Unfortunately the tool does not support fine-tuning pretrained binary fastText models."
            )
            raise

        if self.use_streamlit:
            if "epoch_progress" not in st.session_state:
                st.session_state["epoch_progress"] = 0

            my_bar = st.progress(
                st.session_state["epoch_progress"],
                text="Please note: The model is actively training, \
                      even if the progress bar appears stationary. This might take a while...",
            )

            epoch_logger = EpochLogger(epochs=self.epochs, my_bar=my_bar)

            # Train fastText model
            ft_model.train(
                sentences,
                epochs=self.epochs,
                total_examples=ft_model.corpus_count,
                total_words=ft_model.corpus_total_words,
                callbacks=[epoch_logger],
            )

            # Reset session state for epoch_progress
            st.session_state["epoch_progress"] = 0

        else:
            # Train the word embedding model
            ft_model.train(
                sentences,
                epochs=self.epochs,
                total_examples=ft_model.corpus_count,
                total_words=ft_model.corpus_total_words,
            )

        self.save_model(ft_model, fastText_model_path, name_embedding, t)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train word2vec and fastText embeddings."
    )
    requiredNamed = parser.add_argument_group("Required named arguments")

    requiredNamed.add_argument(
        "--inFilePath",
        action="store",
        metavar="string",
        type=str,
        dest="inFilePath",
        help="defines the input csv file path",
        required=True,
    )

    requiredNamed.add_argument(
        "--outFileName",
        action="store",
        metavar="string",
        type=str,
        dest="outFileName",
        help="specifies word embedding output file name",
        required=True,
    )

    parser.add_argument(
        "--typeEmbedding",
        metavar="string",
        type=str,
        nargs="*",
        default="word2vec",
        help='defines what type of word embedding should \
                              be trained specify either "fastText" \
                              or "word2vec" or select both',
    )

    parser.add_argument(
        "--embeddingDimension",
        metavar="int",
        type=int,
        default=300,
        help="specifies input word embedding \
                              dimension - default dimension is 300",
    )

    parser.add_argument(
        "--CBOW",
        metavar="bool",
        type=bool,
        default=False,
        help="specifies that the model should be trained \
                         using continuous bag-of-words ",
    )

    parser.add_argument(
        "--SG",
        metavar="bool",
        type=bool,
        default=True,
        help="specifies that the model should be trained using skip-gram \
                        - default dimension is skip-gram",
    )

    parser.add_argument(
        "--epochs",
        metavar="int",
        type=int,
        default=10,
        help="specifies number of epochs word embeddings are \
                         trained for - default is 10 epochs",
    )

    parser.add_argument(
        "--randomState",
        metavar="int",
        type=int,
        default=40,
        help="specifies random seed used to initialise word embedding model, \
                        for reproducibility - default value is 40",
    )

    parser.add_argument(
        "--minimumCount",
        metavar="int",
        type=int,
        default=2,
        help="ignores all words with total frequency lower than this \
                         - default value is 2",
    )

    parser.add_argument(
        "--subSampleFactor",
        metavar="float",
        type=float,
        default=1e-5,
        help="threshold for configuring which higher-frequency words \
                        are randomly down-sampled - default value is 1e-5",
    )

    parser.add_argument(
        "--learningRate",
        metavar="float",
        type=float,
        default=0.025,
        help="specifies the initial learning rate \
                        - default value is 0.025",
    )

    parser.add_argument(
        "--minLearningRate",
        metavar="float",
        type=float,
        default=0.0001,
        help="specifies minimum learning rate the initial learning rate will \
                        linearly drop to as training progresses \
                        - default value is 0.0001",
    )

    parser.add_argument(
        "--negativeSampling",
        metavar="int",
        type=int,
        default=10,
        help="specifies how many “noise words” should be drawn (usually between 5-20) \
                         - default value is 5",
    )

    parser.add_argument(
        "--windowSize",
        metavar="int",
        type=int,
        default=5,
        help="maximum distance between the current and predicted word within a sentence \
                        - default value is 5",
    )

    parser.add_argument(
        "--useIterator",
        metavar="bool",
        type=bool,
        default=False,
        help='use this option if you have limited RAM to load sentences \
                        - default value is "False"',
    )

    parser.add_argument(
        "--useStreamlit",
        metavar="bool",
        type=bool,
        default=False,
        help="activates streamlit functionality.",
    )

    parser.add_argument(
        "--pretrainedModelPath",
        action="store",
        metavar="string",
        type=str,
        dest="pretrainedModelPath",
        default=None,
        help="specifies path to a pretrained model that could be used for further fine-tuning",
        required=True,
    )

    parser.add_argument(
        "--usePretrainedBinaryModel",
        metavar="bool",
        type=bool,
        default=False,
        help="specify the pretrained model being used is loaded as a binary model.",
    )

    args = parser.parse_args()
    skipGram = {}
    if args.CBOW and not args.SG:
        skipGram["CBOW"] = 0
    elif args.SG and not args.CBOW:
        skipGram["SG"] = 1
    elif args.CBOW and args.SG:
        skipGram["CBOW"] = 0
        skipGram["SG"] = 1
    elif args.CBOW is False and (args.SG is False):
        parser.error('Select either SG or CBOW to be "True".')

    word_vectors = TrainWordVectors(
        file_location=args.inFilePath,
        name=args.outFileName,
        use_iterator=args.useIterator,
        use_pretrained_path=args.pretrainedModelPath,
        use_pretrained_binary_model=args.usePretrainedBinaryModel,
        use_streamlit=args.useStreamlit,
    )

    for key, value in skipGram.items():
        for typeEmbed in args.typeEmbedding:
            word_vectors.train_word_vector(
                embedding_dimension=args.embeddingDimension,
                epochs=args.epochs,
                sg=value,
                window_size=args.windowSize,
                min_count=args.minimumCount,
                sub_sampling=args.subSampleFactor,
                alpha=args.learningRate,
                min_alpha=args.minLearningRate,
                negative_sampling=args.negativeSampling,
                random_state=args.randomState,
                word_vector=typeEmbed,
            )
