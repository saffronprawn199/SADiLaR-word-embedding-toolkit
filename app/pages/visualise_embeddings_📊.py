import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText as FT_gensim
import numpy as np
from itertools import cycle
import seaborn as sns
import time
import io
import logging

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# The global variable COLOR_PALETTE works
# as follows : ['blue', 'orange', 'green', 'pink', 'black', 'navy', 'indigo', 'gold']

COLOR_PALETTE = [
    "#2874A6",
    "#E67E22",
    "#58D68D",
    "#F533FF",
    "#000000",
    "#000AFF",
    "#A100FF",
    "#FFD700",
]


def list_directories_and_check_files(path):
    # List directories and check each for specified file types.
    # List all entries in the directory
    all_entries = os.listdir(path)

    # Initialize a list to hold directories that meet the criteria
    valid_files = []
    valid_dir = []
    for entry in all_entries:
        dir_path = os.path.join(path, entry)

        # Check if the entry is a directory
        if os.path.isdir(dir_path):
            # List all files in the directory
            files = os.listdir(dir_path)
            # Check if all files in the directory are valid
            # for file in files:
            for f in files:
                if not f.endswith((".npy", ".DS_Store")):
                    valid_files.append(f)
                    valid_dir.append(dir_path)
    return valid_files, valid_dir


# Model selection functions
def model_selector(folder_path, sidebar_key="model_selector"):
    os.makedirs(folder_path, exist_ok=True)
    valid_files, valid_directories = list_directories_and_check_files(folder_path)
    selected_filename_index = st.sidebar.selectbox(
        "Select model: ",
        range(len(valid_files)),
        format_func=lambda x: valid_files[x],
        key=sidebar_key,
    )

    selected_model_path = os.path.join(
        valid_directories[selected_filename_index], valid_files[selected_filename_index]
    )
    return selected_model_path


# Words to search for
def display_search():
    string_of_words = st.sidebar.text_input("Word Lookup: ", "")
    search_for = [word.replace(" ", "") for word in string_of_words.split(",")]
    return search_for


def show_word_similarity_score():
    pass


# Check if directory model exists
def check_directory_exists(directory_path):
    # Check if the specified directory exists
    if not os.path.isdir(directory_path):
        # If the directory does not exist, display a warning message
        st.error(f"Directory {directory_path} does not exist.")
        return False
    else:
        # If the directory exists, you can perform further actions or just return True
        return True


# no. dimensions
def display_dimensions():
    dims = ("2-D", "3-D")
    selected_dim = st.sidebar.radio("Dimensions: ", dims)
    return selected_dim


# Select embedding
def display_model_selector():
    embedding_type = ("word2vec", "fastText")
    selected_embedding = st.sidebar.radio("Embedding model: ", embedding_type)
    return selected_embedding


# Select dimensionality reduction technique
def display_dimensionality_reduction_technique():
    dimensionality_reduction_technique = ("T-SNE", "PCA")
    selected_technique = st.sidebar.radio(
        "Dimensionality reduction technique: ", dimensionality_reduction_technique
    )
    return selected_technique


# Get word vectors
def get_word_vectors(word_vectors, word_list):
    return [np.array(word_vectors.get_vector(word, norm=True)) for word in word_list]


# Get similar words to chosen word
def get_similar_words(word_vectors, word, topn=15):
    return [word for word, _ in word_vectors.most_similar(positive=[word], topn=topn)]


# TSNE dimensionality reduction
def tsne_dimensionality_reduction(
    word_vectors, random_seed, tsne_perplexity, tsne_iterations, n_components=2
):
    # t-SNE reduction
    np.set_printoptions(suppress=True)
    array_of_list_of_word_vectors = np.array(word_vectors)
    tsne_reduce = TSNE(
        n_components=n_components,
        random_state=random_seed,
        perplexity=tsne_perplexity,
        n_iter=tsne_iterations,
    ).fit_transform(array_of_list_of_word_vectors)

    return tsne_reduce


# PCA dimensionality reduction
def pca_dimensionality_reduction(word_vectors, random_seed, n_components=2):
    # PCA reduction
    np.set_printoptions(suppress=True)
    array_of_list_of_word_vectors = np.array(word_vectors)
    pca_reduce = PCA(n_components=n_components, random_state=random_seed).fit_transform(
        array_of_list_of_word_vectors
    )
    return pca_reduce


def get_embedding_data(model_path, word_list, number_similar_words, select_model_type):
    wv = load_embedding_model(model_path, select_model_type)
    word_vector_list = get_word_vectors(wv, word_list)
    # Color of words
    key_word_color_list = []
    color_pallet_cycle = cycle(COLOR_PALETTE)
    color_list_similar_words = []
    # Get similar words and add them to the set
    set_of_similar_words = []
    for word in word_list:
        key_word_color_list.append("#FF0000")
        similar_words = get_similar_words(wv, word, number_similar_words)
        set_of_similar_words.extend(similar_words)
        # logger.info("tuple_similar_words: ", similar_words)
        color_list_similar_words.extend(len(similar_words) * [next(color_pallet_cycle)])

    # Update labels and vectors
    word_list.extend(set_of_similar_words)
    word_vector_list.extend(get_word_vectors(wv, set_of_similar_words))

    # color of words
    key_word_color_list.extend(color_list_similar_words)

    return word_list, word_vector_list, key_word_color_list


def plot_dimensionality_reduction_2D(
    reduce, word_labels, color_list, word_list, plot_title, loc_legend
):

    # DataFrame for plotting
    df = pd.DataFrame(
        {
            "x": reduce[:, 0],
            "y": reduce[:, 1],
            "words": word_labels,
            "color": color_list,
        }
    )

    custom = []
    # COLOR_PALETTE
    color_pallet_cyclic_iterator = cycle(COLOR_PALETTE)

    for i in range(len(word_list)):
        custom.append(
            Line2D(
                [],
                [],
                marker=".",
                color=next(color_pallet_cyclic_iterator),
                linestyle="None",
                markersize=8,
            )
        )

    fig, _ = plt.subplots(figsize=(22, 12))

    # Plot setup
    p1 = sns.scatterplot(x="x", y="y", hue="color", data=df, legend=None, s=1)

    legend = plt.legend(custom, word_list, loc=loc_legend, title="Words", fontsize=12)

    plt.setp(legend.get_title(), fontsize=24)

    # Annotations
    for line in range(df.shape[0]):
        p1.text(
            df["x"][line],
            df["y"][line],
            " " + df["words"][line].title(),
            horizontalalignment="left",
            verticalalignment="bottom",
            color=df["color"][line],
            fontsize=10,
        )

    # Axis and title setup
    plt.xlim(df.iloc[:, 0].min(), df.iloc[:, 0].max())
    plt.ylim(df.iloc[:, 1].min(), df.iloc[:, 1].max())  # continuous bag of words
    plt.title(plot_title, fontsize=24)
    plt.tick_params(labelsize=20)
    plt.xlabel("tsne-one", fontsize=24)
    plt.ylabel("tsne-two", fontsize=24)
    st.pyplot(fig)

    fn = "PLACEHOLDER_NAME.png"
    img = io.BytesIO()
    plt.savefig(img, format="png")

    btn = st.download_button(
        label="Download image", data=img, file_name=fn, mime="image/png"
    )


@st.cache_data
def convert_img(fig):
    return fig.to_image(format="png", width=950, height=750, scale=3)


def plot_dimensionality_reduction_3D(
    reduce,
    word_labels,
    color_list,
    word_list,
    plot_title,
    opacity,
    text_size,
    add_text,
):
    # DataFrame for plotting
    df = pd.DataFrame(
        {
            "x": reduce[:, 0],
            "y": reduce[:, 1],
            "z": reduce[
                :, 2
            ],  # Assuming tsne_reduce now includes a third dimension for 3D plotting
            "words": word_labels,
            "color": color_list,
        }
    )

    colors_plot = set(color_list)

    colors_for_groups = [c for c in colors_plot if c != "#FF0000"]

    fig = go.Figure()

    # Map each keyword to a specific color
    keyword_color_map = {
        word: color for word, color in zip(df["words"].tolist(), colors_for_groups)
    }

    color_df = pd.DataFrame(keyword_color_map, columns=["words", "colors"])

    # Create a scatter plot for each word, using the specific color
    for word in set(word_labels):  # Use set to avoid duplicates
        word_df = df[df["words"] == word]
        # Use the mapped color for the current word
        # color = keyword_color_map[word] if word in keyword_color_map else 'gray'
        if word in word_list:
            fig.add_trace(
                go.Scatter3d(
                    x=word_df["x"],
                    y=word_df["y"],
                    z=word_df["z"],
                    mode="markers+text",  # Combine markers and text
                    marker=dict(
                        size=8, color=color_df["colors"], symbol="diamond"
                    ),  # Use specific color
                    text=word_df["words"],  # Text labels for each point
                    textposition="top center",  # Position the text above the markers
                    name=word,
                    legendgroup="group",  # this can be any string, not just "group"
                    legendgrouptitle_text="Words of interest",
                )
            )

        else:
            if add_text != "":
                mode = f"markers+{add_text}"
            else:
                mode = "markers"
            fig.add_trace(
                go.Scatter3d(
                    x=word_df["x"],
                    y=word_df["y"],
                    z=word_df["z"],
                    opacity=opacity,
                    mode=mode,  # Combine markers and text
                    marker=dict(
                        size=text_size, color=word_df["color"]
                    ),  # Use specific color
                    text=word_df["words"],  # Text labels for each point
                    textposition="top center",  # Position the text above the markers
                    legendgroup="group",
                    showlegend=False,
                )
            )

    # Update plot layout
    fig.update_layout(
        title=plot_title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=30),  # Adjust margins to fit layout
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    fn = "PLACEHOLDER_NAME.png"

    buf = io.BytesIO()
    fig.write_image(buf, format="png")
    btn = st.download_button(
        label="Download image", data=buf, file_name=fn, mime="image/png"
    )
@st.cache_resource
def load_embedding_model(model_path, select_model_type):
    t = time.time()
    if select_model_type == 'word2vec':
        embedding_model = Word2Vec.load(model_path)
    elif select_model_type == 'fastText':
        embedding_model = FT_gensim.load(model_path)
    logger.info(f"Load time:  {time.time() - t}")
    wv = embedding_model.wv
    logger.info(f"Vocab size: {len(wv)}")
    return wv


@st.cache_resource
def check_words_vocab(model_path, word_list, select_model_type):
    st.write('Please be patient, while the tool loads the embedding model and does a vocabulary check.')
    wv = load_embedding_model(model_path, select_model_type)
    for word in word_list:
        if word not in wv.key_to_index:
            st.error(f'The word "{word}" is not in the vocabulary.')
            return False
        else:
            return True


# Streamlit app layout
def main():
    st.title("Word Embedding Visualisation Tool")
    with st.expander("See Instruction"):
        st.markdown(''' 
                        - Choose your trained embedding model.
                        - Select the words you would like to investigate by writing them as a comma-separated list (e.g., "cat, dog, car").
                        - Wait for vocabulary check to complete. This make sure the embedding model has numeric representation for the word.
                        - Choose whether you want to plot in 2D or 3D and select the dimensionality reduction technique you wish to use.
                        - Name your plot and press "Visualise" to see it. 
                    ''')

    select_model_type = display_model_selector()
    model_path = "./" + select_model_type + "_models"
    dir_exists = check_directory_exists(model_path)

    if dir_exists:
        model_path_w2v = model_selector(model_path)
        search_for = display_search()
        if search_for[0] != "":
            dim = display_dimensions()
            technique = display_dimensionality_reduction_technique()

            passed_vocab_check = check_words_vocab(model_path_w2v, search_for, select_model_type)
            search_for_copy = search_for.copy()

            number_similar_words = st.sidebar.slider(
                "Number of similar words around keyword: ",
                5,
                50,
                1,
                help="Number of semantically related words around keyword.",
            )
            tsne_perplexity = None
            tsne_iterations = None
            random_seed = None

            plot_title = st.sidebar.text_input(label="Title of plot: ")

            if technique == "T-SNE":
                tsne_perplexity = st.sidebar.slider(
                    "Perplexity",
                    5,
                    50,
                    1,
                    help="The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. \
                          Larger datasets usually require a larger perplexity. \
                          Consider selecting a value between 5 and 50 (e.g a good value to start with is 20). \
                          Different values can result in significantly different results.\
                          The perplexity must be less than the number of samples.",
                )
                tsne_iterations = st.sidebar.slider(
                    "Number of iterations",
                    250,
                    30000,
                    1,
                    help="Maximum number of iterations for the optimization (e.g. good value to start with could be 15000). \
                          Should be at least 250.",
                )

                random_seed = st.sidebar.number_input(
                    "Random state",
                    min_value=0,
                    value=40,
                    help="Random seed used to initialise TSNE algorithm, for reproducibility - default value is 40",
                )

            if passed_vocab_check:
                if dim == "2-D":
                    loc_legend = st.sidebar.selectbox(
                        "Select legend position: ",
                        ["lower left", "upper left", "lower right", "upper right"],
                    )
                    button = st.sidebar.button("Visualise")
                    if button:
                        word_labels, word_vectors, color_list = get_embedding_data(
                            model_path_w2v, search_for, number_similar_words, select_model_type
                        )
                        if technique == "T-SNE":
                            tsne_reduce = tsne_dimensionality_reduction(
                                word_vectors,
                                random_seed,
                                tsne_perplexity,
                                tsne_iterations,
                                2,
                            )
                            plot_dimensionality_reduction_2D(
                                tsne_reduce,
                                word_labels,
                                color_list,
                                search_for_copy,
                                plot_title,
                                loc_legend,
                            )
                        else:
                            pca_reduce = pca_dimensionality_reduction(
                                word_vectors, random_seed, 2
                            )
                            plot_dimensionality_reduction_2D(
                                pca_reduce,
                                word_labels,
                                color_list,
                                search_for_copy,
                                plot_title,
                                loc_legend,
                            )
                else:
                    opacity = st.sidebar.slider(
                        "Opacity: ",
                        step=0.1,
                        max_value=1.0,
                        min_value=0.1,
                        help="Opacity of plot.",
                    )
                    size_sphere = st.sidebar.slider(
                        "Sphere size: ", 5, 20, 1, help="Size of sphere"
                    )
                    text_stuff = ("text", "no-text")
                    text_stuff = st.sidebar.radio("Turn on text: ", text_stuff)
                    add_text = ""
                    if text_stuff == "text":
                        add_text = "text"
                    button = st.sidebar.button("Visualise")
                    if button:
                        word_labels, word_vectors, color_list = get_embedding_data(
                            model_path_w2v, search_for, number_similar_words, select_model_type
                        )
                        if technique == "T-SNE":
                            tsne_reduce = tsne_dimensionality_reduction(
                                word_vectors,
                                random_seed,
                                tsne_perplexity,
                                tsne_iterations,
                                3,
                            )
                            plot_dimensionality_reduction_3D(
                                tsne_reduce,
                                word_labels,
                                color_list,
                                search_for_copy,
                                plot_title,
                                opacity,
                                size_sphere,
                                add_text,
                            )
                        else:
                            pca_reduce = pca_dimensionality_reduction(
                                word_vectors, random_seed, 3
                            )
                            plot_dimensionality_reduction_3D(
                                pca_reduce,
                                word_labels,
                                color_list,
                                search_for_copy,
                                plot_title,
                                opacity,
                                size_sphere,
                                add_text,
                            )


if __name__ == "__main__":
    main()
