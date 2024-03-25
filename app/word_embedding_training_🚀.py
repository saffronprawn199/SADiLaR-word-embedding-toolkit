# File: app.py
import sys
sys.path.insert(0, "../")

import streamlit as st
import os
from word_vector_training.train_word_vectors import TrainWordVectors


# Set Streamlit page config
def set_page_config():
    st.set_page_config(page_title="Word embedding App", page_icon="👋")


# Setup session states
def setup_session_states():
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = ""


# Setup training folders for word embeddings
def setup_word_vector_training_folders():
    word2vec_model_path = "./word2vec_models/"
    fastText_model_path = "./fastText_models/"

    os.makedirs(word2vec_model_path, exist_ok=True)
    os.makedirs(fastText_model_path, exist_ok=True)


# Main page setup
def main_page_setup():
    st.title("Word Embedding Training Tool")
    with st.expander("See Instruction"):
         st.markdown('''
                  - Create a CSV file with a column named `Sentences`. Ensure that documents are added as rows in this CSV file, not as new columns.
                  - Set up the model type and training configuration, including the model's hyperparameters, using the sidebar on the left side of your screen. Name your model and click "Train word embeddings". 
                  - After the model has finished training, it will be saved in a folder within your local app directory, organised by similarly named model type (i.e. fastText or word2vec).
                  - Please visit the "Visualise Page" after the training is complete to view your trained embeddings..
                    ''')
    st.sidebar.success("Select an option above.")


# User input handling
def handle_user_input():
    my_input = st.text_input(
        "Please provide a name for the embedding model:", st.session_state["model_name"]
    )
    submit = st.button("Train word embeddings")
    if submit:
        st.session_state["model_name"] = my_input
    return submit


# Presents the user with an option to choose whether to use pretrained embeddings or not.
# The selection is made via a radio button in the Streamlit sidebar.
def display_use_pretrained_embeddings():
    select_pretrained = ("True", "False")
    selected_option = st.sidebar.radio("Use pretrained embeddings:", select_pretrained)
    return selected_option


# Allows the user to select the type of embedding model to use (Word2Vec or fastText).
# The selection is made via a radio button in the Streamlit sidebar.
def display_model_selector():
    embedding_type = ("word2vec", "fastText")
    selected_embedding = st.sidebar.radio("Embedding model: ", embedding_type)
    return selected_embedding


# Displays a selection box for the user to select a specific model file from a given folder.
# It handles both regular and binary file formats, adjusting the selection accordingly.
def model_selector(folder_path, sidebar_key="model_selector"):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox(
        "Select model: ", filenames, key=sidebar_key
    )

    selected_binary_file = False
    if selected_filename is not None:
        if selected_filename.endswith(".bin"):
            selected_binary_file = True
        else:
            filenames = os.listdir(folder_path + "/" + selected_filename)
            selected_file = [
                f for f in filenames if not f.endswith((".npy", ".DS_Store"))
            ]
            selected_filename = selected_filename + "/" + selected_file[0]

    selected_model_path = None
    if selected_filename:
        selected_model_path = os.path.join(folder_path, selected_filename)
    return selected_model_path, selected_binary_file


# Setup further fine-tuning of pretrained models.
def finetune_existing_model():
    os.makedirs("./pretrained_embeddings", exist_ok=True)
    os.makedirs("./pretrained_embeddings/word2vec_models", exist_ok=True)
    os.makedirs("./pretrained_embeddings/fastText_models", exist_ok=True)
    option_selected = display_use_pretrained_embeddings()

    model_path = ""
    pretrained_binary = False
    model_selected = None

    if option_selected == "True":
        model_selected = display_model_selector()
        st.sidebar.info(
            "Ensure that the model architecture (SG or CBOW) matches that of the pre-trained model. \
                         Additionally, the embedding dimensions must be consistent with the pre-trained model.",
            icon="ℹ️",
        )

        if model_selected == "word2vec":
            model_path, pretrained_binary = model_selector(
                "./pretrained_embeddings/word2vec_models"
            )
        else:
            model_path, pretrained_binary = model_selector(
                "./pretrained_embeddings/fastText_models"
            )

    return option_selected, model_path, pretrained_binary, model_selected


# Sidebar configuration for clustering and model training options
def sidebar_config(pretraining_selected, model_selected):
    st.sidebar.header("Word Embedding Training Options")

    hyperparameters_arguments = dict()

    if pretraining_selected == "False":
        # Optional arguments
        hyperparameters_arguments["model_name"] = st.sidebar.multiselect(
            "Type of word embedding to be trained",
            ["word2vec", "fastText"],
            default=["word2vec"],
            help='Specify either "fastText" or "word2vec" or select both',
        )

        hyperparameters_arguments["CBOW"] = st.sidebar.checkbox(
            "Continuous bag-of-words",
            value=False,
            help="Train using continuous bag-of-words",
        )

        hyperparameters_arguments["SG"] = st.sidebar.checkbox(
            "Skip-gram", value=True, help="Train using skip-gram - default is skip-gram"
        )
    else:
        hyperparameters_arguments["model_name"] = model_selected

        model_architecture = st.sidebar.radio(
            "Model Architecture",
            options=["CBOW", "Skip-gram"],
            index=1,  # Default to Skip-gram
            help="Select the training algorithm: Continuous bag-of-words or Skip-gram",
        )
        # Update hyperparameters_arguments based on selection
        hyperparameters_arguments["CBOW"] = model_architecture == "CBOW"
        hyperparameters_arguments["SG"] = model_architecture == "Skip-gram"

    hyperparameters_arguments["embedding_dimension"] = st.sidebar.number_input(
        "Word embedding dimension",
        min_value=1,
        value=300,
        help="Specifies input word embedding dimension - default dimension is 300",
    )

    hyperparameters_arguments["epochs"] = st.sidebar.number_input(
        "Number of epochs",
        min_value=1,
        value=10,
        help="Specifies number of epochs for training - default is 10 epochs",
    )

    hyperparameters_arguments["random_state"] = st.sidebar.number_input(
        "Random state",
        min_value=0,
        value=40,
        help="Random seed used to initialise word embedding model, for reproducibility - default value is 40",
    )

    hyperparameters_arguments["minimum_count"] = st.sidebar.number_input(
        "Minimum count",
        min_value=0,
        value=2,
        help="Ignores all words with total frequency lower than this - default value is 2",
    )

    hyperparameters_arguments["sub_sample_factor"] = st.sidebar.number_input(
        "Subsample factor",
        min_value=0.0,
        value=1e-5,
        format="%.5f",
        help="Threshold for which higher-frequency words are randomly downsampled - default value is 1e-5",
    )

    hyperparameters_arguments["learning_rate"] = st.sidebar.number_input(
        "Learning rate",
        min_value=0.0,
        value=0.025,
        format="%.5f",
        help="Initial learning rate - default value is 0.025",
    )

    hyperparameters_arguments["min_learning_rate"] = st.sidebar.number_input(
        "Minimum learning rate",
        min_value=0.0,
        value=0.0001,
        format="%.5f",
        help="Minimum learning rate as training progresses - default value is 0.0001",
    )

    hyperparameters_arguments["negative_sampling"] = st.sidebar.number_input(
        "Negative sampling",
        min_value=0,
        value=10,
        help="How many “noise words” should be drawn - default value is 10",
    )

    hyperparameters_arguments["window_size"] = st.sidebar.number_input(
        "Window size",
        min_value=1,
        value=5,
        help="Maximum distance between the current and predicted word within a sentence - default value is 5",
    )

    hyperparameters_arguments["use_iterator"] = st.sidebar.checkbox(
        "Use iterator for limited RAM",
        value=False,
        help="Use if you have limited RAM to load sentences - default value is False",
    )

    return hyperparameters_arguments


# Train word embeddings
def train_word_vectors(
    word_vectors, hyperparameters_arguments, pretraining_selected, skip_gram
):

    if pretraining_selected == "True":
        word_vectors.train_word_vector(
            sg=list(skip_gram.values())[0],
            embedding_dimension=hyperparameters_arguments["embedding_dimension"],
            epochs=hyperparameters_arguments["epochs"],
            window_size=hyperparameters_arguments["window_size"],
            min_count=hyperparameters_arguments["minimum_count"],
            sub_sampling=hyperparameters_arguments["sub_sample_factor"],
            alpha=hyperparameters_arguments["learning_rate"],
            min_alpha=hyperparameters_arguments["min_learning_rate"],
            negative_sampling=hyperparameters_arguments["negative_sampling"],
            random_state=hyperparameters_arguments["random_state"],
            word_vector=hyperparameters_arguments["model_name"],
        )
    else:
        for key, value in skip_gram.items():
            for typeEmbed in hyperparameters_arguments["model_name"]:
                word_vectors.train_word_vector(
                    embedding_dimension=hyperparameters_arguments[
                        "embedding_dimension"
                    ],
                    epochs=hyperparameters_arguments["epochs"],
                    sg=value,
                    window_size=hyperparameters_arguments["window_size"],
                    min_count=hyperparameters_arguments["minimum_count"],
                    sub_sampling=hyperparameters_arguments["sub_sample_factor"],
                    alpha=hyperparameters_arguments["learning_rate"],
                    min_alpha=hyperparameters_arguments["min_learning_rate"],
                    negative_sampling=hyperparameters_arguments["negative_sampling"],
                    random_state=hyperparameters_arguments["random_state"],
                    word_vector=typeEmbed,
                )


# Setup hyperparameters for word vector training
def setup_word_vector_training(
    uploaded_file,
    hyperparameters_arguments,
    pretraining_selected,
    pretrained_model_path,
    pretrained_is_binary,
):

    # Logic_for training word embeddings
    skip_gram = {}
    if hyperparameters_arguments["CBOW"] and not hyperparameters_arguments["SG"]:
        skip_gram["CBOW"] = 0
    elif hyperparameters_arguments["SG"] and not hyperparameters_arguments["CBOW"]:
        skip_gram["SG"] = 1
    elif hyperparameters_arguments["CBOW"] and hyperparameters_arguments["SG"]:
        skip_gram["CBOW"] = 0
        skip_gram["SG"] = 1
    elif hyperparameters_arguments["CBOW"] is False and (
        hyperparameters_arguments["SG"] is False
    ):
        st.error('Select either SG or CBOW to be "True".', icon="🚨")

    if pretraining_selected == "True":
        word_vectors = TrainWordVectors(
            file_location=uploaded_file,
            name=st.session_state["model_name"],
            use_iterator=hyperparameters_arguments["use_iterator"],
            use_pretrained_path=pretrained_model_path,
            use_pretrained_binary_model=pretrained_is_binary,
            use_streamlit=True,
        )
        train_word_vectors(
            word_vectors, hyperparameters_arguments, pretraining_selected, skip_gram
        )
    else:
        word_vectors = TrainWordVectors(
            file_location=uploaded_file,
            name=st.session_state["model_name"],
            use_iterator=hyperparameters_arguments["use_iterator"],
            use_streamlit=True,
        )
        train_word_vectors(
            word_vectors, hyperparameters_arguments, pretraining_selected, skip_gram
        )


# File upload and processing
def upload_file_and_train_model(
    hyperparameters_arguments,
    pretraining_selected,
    pretrained_model_path,
    pretrained_is_binary,
):
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            submit_model_name = handle_user_input()

            if submit_model_name:
                setup_word_vector_training(
                    uploaded_file,
                    hyperparameters_arguments,
                    pretraining_selected,
                    pretrained_model_path,
                    pretrained_is_binary,
                )
                st.write(
                    "The embedding models have been successfully trained.\
                     You may now navigate to the visualization page to explore the results. 🥳🎊🎉"
                )
        else:
            st.error("Wrong file format, please upload csv", icon="🚨")


# Main function to orchestrate the app components
def main():
    setup_session_states()
    set_page_config()
    setup_word_vector_training_folders()
    main_page_setup()
    pretraining_selected, pretrained_model_path, pretrained_binary, model_selected = (
        finetune_existing_model()
    )
    hyperparameters_arguments = sidebar_config(pretraining_selected, model_selected)
    upload_file_and_train_model(
        hyperparameters_arguments,
        pretraining_selected,
        pretrained_model_path,
        pretrained_binary,
    )


if __name__ == "__main__":
    main()
