"""
    Top2Vec module.

    Source code adapted from https://github.com/ddangelov/Top2Vec and https://github.com/MaartenGr/BERTopic.
"""

import re

import nltk

import umap
import hdbscan

import logging

import numpy as np
import pandas as pd

import plotly.express as px

from typing import Union, List, Tuple

from sklearn.cluster import dbscan
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sentence_transformers import SentenceTransformer



# Download the necesary resources from nltk
nltk.download('punkt')
nltk.download('stopwords')

# Declare constants
NAME = "top2vec"

# Set seed for reproducibility purposes
SEED = 0

# Initialize Stemmer
STEMMER = PorterStemmer()

# Get stopwords and remove punctutaions from them
STOP_WORDS = [re.sub(r"[^a-z]", "", stopword) for stopword in stopwords.words("english")]



# Setup logger
logger = logging.getLogger(NAME)
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(sh)



def process_sentence(sentence) -> List[str]:
    """
        Function to process a sentence. Processing pipeline consists of the following
        steps:

        1) Convert to lowercase
        2) Remove non-alphabetic characters
        3) Tokenize
        3) Remove stopwords
        4) Stem the tokens
        5) Construct bigrams

        Args
        ----------
        sentence: string
                input sentence to be processed.

        Returns
        ----------
        terms: list 
                unigrams and bigrams from sentence
    """
    
    # Convert to lowercase
    sentence = sentence.lower()
    
    # Remove non-alphabetic characters
    sentence = re.sub(r"[^a-z ]", "", sentence)
    
    # Remove stopwords
    tokens = [token for token in sentence.split() if token not in STOP_WORDS]
    
    # Perform stemming
    tokens = [STEMMER.stem(word) for word in tokens]
    
    # Construct bigrams
    bigrams = ["_".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    
    # Return tokens
    return tokens + bigrams



def top_n_terms(corpus: Union[List, pd.Series], n: int = 50) -> pd.DataFrame:
    """
        Function to extract the top n terms from a corpus based on tf-idf scores.

        Args
        ----------
        corpus: list or pd.Series of strings
                input corpus.

        n: int (Optional, default 50)
                number of terms to return

        Returns
        ----------
        top n terms: pd.DataFrame
                top n unigrams and bigrams from the corpus
    """
    # Initialize tf-idf vectorizer
    vectorizer = TfidfVectorizer(analyzer = process_sentence)
    vectorizer.build_analyzer()
    
    # Compute document term matrix
    document_term_matrix = vectorizer.fit_transform(corpus).toarray()
    
    # Get vocabulary
    vocab = vectorizer.get_feature_names_out()
    
    # Calculate the average tf-idf score for each term
    avg_scores = document_term_matrix.mean(axis = 0)
    
    # Sort the scores and get the index of the top n scores
    top_n_indexes = avg_scores.argsort()[-n:][::-1]
    
    # Store the results into a dataframe
    return pd.DataFrame(
        [(vocab[idx], avg_scores[idx]) for idx in top_n_indexes],
        columns = ["term", "score"],
    ).sort_values(by = ["score"], ascending = True, ignore_index = True)



class Top2Vec:
    """
        Top2Vec

        Creates jointly embedded topic and document embeddings.

        Args
        ----------
        embedding_model: string (Optional, default "all-MiniLM-L6-v2")
                name of a SentenceTransformers pretrained model.

        umap_model: umap.UMAP (Optional, default None)
                umap model for dimensionality reduction.

        hdbscan_model: hdbscan.HDBSCAN (Optional, default None)
                hdbscan model for clustering of embeddings.

        vectorizer_model: TfidfVectorizer (Optional, default None)
                vectorizer model for obtaining text embeddings based on term frequency - inverse document frequency (tf-idf).

        seed: int (Optional, default SEED)
                seed for reproducibility of experiment results

        logger: logging.Logger
                logging.Logger object to log messages.
    """
    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 umap_model: umap.UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: TfidfVectorizer = None,
                 seed: int = SEED,
                 logger: logging.Logger = logger,
                ):
        
        # Validate logger
        if not isinstance(logger, logging.Logger):
            raise TypeError("logger needs to be an instance of a logging.Logger object.")
        
        # Load embedding model
        logger.info(f"Loading {embedding_model} model.") 
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        
        except:
            raise ValueError("Please select a valid SentenceTransformers model.")
            
        logger.info(f"Loaded {embedding_model} model successfully.")
            
        self.seed = seed
        self.results = None
        
        # UMAP
        self.umap_model = umap_model or umap.UMAP(n_neighbors = 15,
                                                  n_components = 5,
                                                  metric = "cosine",
                                                  random_state = self.seed)
        
        # Set seed for HDBSCAN
        np.random.seed(self.seed)
        
        # HDBSCAN
        self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(min_cluster_size = 15, # To experiment with other values
                                                              metric = "euclidean",
                                                              cluster_selection_method = "eom")
        
        # Vectorizer
        self.vectorizer_model = vectorizer_model or TfidfVectorizer(analyzer = process_sentence)
        self.vectorizer_model.build_analyzer()
        
        
    def fit(self, documents: Union[List[str], pd.Series]) -> None:
        """
            Method to apply Top2Vec algorithm to input documents. Top2Vec algorithm pipline
            consists of the following steps:

            1) Obtain document embeddings
            2) Perform dimensionality reduction on the document embeddings using UMAP
            3) Cluster the compressed document embeddings with HDBSCAN
            4) Create topic vectors
            5) Deduplicate topics using DBSCAN

            Args
            ----------
            documents: list or pd.Series of strings
                    input text corpus.

            Returns
            ----------
            None           
        """
        # Validate documents
        if not isinstance(documents, (list, pd.Series)):
            raise TypeError("documents needs to be a list or pandas series of strings.")
            
        if not all(isinstance(document, str) for document in documents):
            raise TypeError("documents needs to be a list or pandas series of strings.")
        
        columns = ["document"]
        if isinstance(documents, list):
            self.results = pd.DataFrame(documents, columns = columns)
        
        if isinstance(documents, pd.Series):
            self.results = documents.to_frame(name = columns[0])

        self.summary, self.top_n_terms = None, None
        
        # Obtain document embeddings
        logger.info("Obtaining document embeddings.")
        self.document_embeddings = self.embedding_model.encode(documents,
                                                               convert_to_numpy = True,
                                                               normalize_embeddings = True)
        
        # Obtain umap embeddings
        logger.info("Creating lower dimension document embeddings.")
        umap_embeddings = self.umap_model.fit(self.document_embeddings).embedding_
        
        # Obtain hdbscan clusters
        logger.info("Finding dense areas of documents.")
        clusters = self.hdbscan_model.fit(umap_embeddings)
        
        # Create topic vectors
        logger.info("Finding topics.")
        self.create_topic_vectors(clusters.labels_)
        
        # Deduplicate topics
        self.deduplicate_topics()
        
        # Assign topic to documents
        self.doc_top, self.doc_dist = self.calculate_documents_topic()
        
        # Calculate topic_sizes
        self.topic_sizes = self.calculate_topic_sizes()
        
        # Re-order topics
        self.reorder_topics()
        
        # Append clustering results to dataframe
        self.results["topic"], self.results["score"] = self.doc_top, self.doc_dist
        
        # Sort results by topic and score
        self.results.sort_values(
            by = ["topic", "score"], ascending = [True, False], inplace = True)
        
        self.results.reset_index(drop = True, inplace = True)
        
    
    def create_topic_vectors(self, cluster_labels: np.ndarray) -> None:
        """
            Method to calculate the topic vectors based on the arithmetic mean of all the 
            document embeddings in the same dense cluster.

            Args
            ----------
            cluster_labels: np.ndarray
                    cluster assigned to each document based on HDBSCAN algorithm.

            Returns
            ----------
            None
        """
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
              unique_labels.remove(-1)

        self.topic_vectors = self.l2_normalize(
            np.vstack([self.document_embeddings[np.where(cluster_labels == label)[0]]
                       .mean(axis = 0) for label in unique_labels]))
            
            
    def deduplicate_topics(self) -> None:
        """
            Method to merge duplicate topics.

            Returns
            ----------
            None
        """
        _, labels = dbscan(X = self.topic_vectors,
                           eps = 0.1,
                           min_samples = 2,
                           metric = "cosine")

        duplicate_clusters = set(labels)

        if len(duplicate_clusters) > 1 or -1 not in duplicate_clusters:
            
            # Unique topics
            unique_topics = self.topic_vectors[np.where(labels == -1)[0]]

            if -1 in duplicate_clusters:
                duplicate_clusters.remove(-1)
                
            # Merge duplicate topics
            for unique_label in duplicate_clusters:
                unique_topics = np.vstack(
                    [unique_topics, self.l2_normalize(self.topic_vectors[np.where(labels == unique_label)[0]]
                                                      .mean(axis = 0))])
            self.topic_vectors = unique_topics
            
            
    def calculate_documents_topic(self, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
            Method to compute the topic and score of each document.

            Args
            ----------
            batch_size: int (Optional, default 64)
                    number of documents passed to the model per iteration.

            Returns
            ----------
            (document_topics, document_scores): tuple of a pair of np.ndarray
                    the topic assigned to and score of each document. 
        """
        doc_top, doc_dist = [], []
        for start_index in range(0, len(self.document_embeddings), batch_size):
            res = np.inner(self.document_embeddings[start_index: start_index + batch_size], 
                           self.topic_vectors)
            doc_top.extend(np.argmax(res, axis = 1))
            doc_dist.extend(np.max(res, axis = 1))
    
        return np.array(doc_top), np.array(doc_dist)
    
    
    def calculate_topic_sizes(self) -> pd.Series:
        """
            Method to calculate the topic sizes.

            Returns
            ----------
            topic_sizes: pd.Series
                    number of documents belonging to each topic.
        """
        return pd.Series(self.doc_top).value_counts()


    def reorder_topics(self) -> None:
        """
            Method to sort the topics in descending order based on topic size.

            Returns
            ----------
            None
        """
        self.topic_vectors = self.topic_vectors[self.topic_sizes.index]
        old2new = dict(zip(self.topic_sizes.index, range(self.topic_sizes.index.shape[0])))
        self.doc_top = np.array([old2new[i] for i in self.doc_top])
        self.topic_sizes.reset_index(drop=True, inplace=True)


    def get_topic_sizes(self) -> pd.DataFrame:
        """
            Method to get the topic sizes.

            Returns
            ----------
            topic_sizes: pd.DataFrame
                    number of documents belonging to each topic.
        """
        return self.topic_sizes.to_frame(name = "count") \
                               .reset_index() \
                               .rename(columns = {"index": "topic"})
        
        
    def get_results(self) -> pd.DataFrame:
        """
            Method to get the clustering results.

            Returns
            ----------
            clustering results: pd.DataFrame
        """
        return self.results
        
        
    def get_summary(self, top_n_documents: int = 10) -> pd.DataFrame:
        """
            Method to get the summary of each topic.

            Args
            ----------
            top_n_documents: int (Optional, default 10)
                    number of documents to include in each topic summary.

            Returns
            ----------
            summary: pd.DataFrame
                    top n documents of each topic
        """
        self.summary = self.results.groupby("topic").head(top_n_documents).reset_index(drop=True) if self.summary is None else self.summary
        return self.summary
    
    
    def get_top_n_terms(self, top_n_terms: int = 15) -> pd.DataFrame:
        """
            Method to get the top n terms in each topic based on c-tf-idf scores

            https://maartengr.github.io/BERTopic/api/ctfidf.html

            Args
            ----------
            top_n_terms: int (Optional, default 15)
                    number of terms to include for each topic.

            Returns
            ----------
            top n terms: pd.DataFrame
                    top n terms of each topic
        """
        if self.top_n_terms is None:
            # Aggregate the sentences by topic
            docs_by_topic = self.results.groupby("topic", as_index = False).agg({"document": " ".join})

            # Compute document term matrix
            document_term_matrix = self.vectorizer_model.fit_transform(
                docs_by_topic.document
            ).toarray()

            # Get vocabulary
            vocab = self.vectorizer_model.get_feature_names_out()

            # Generate the top n words per topic
            self.top_n_terms =  pd.DataFrame(
                [(doc, vocab[word], document_term_matrix[doc][word]) 
                for doc in docs_by_topic["topic"] 
                for word in document_term_matrix.argsort(axis=1)[:, -top_n_terms:][doc][::-1]],
                columns = ["topic", "term", "score"])
        
        return self.top_n_terms


    def get_topic_info(self, topic: int) -> None:
        """
            Method to plot the top n terms and print out the top 10 sentences from a topic.

            Args
            ----------
            topic: int
                    topic number.

            Returns
            ----------
            None
        """
        # Validate topic
        if not isinstance(topic, int):
            raise TypeError("topic needs to be an integer.")

        topics = self.get_topic_sizes().topic
        if topic not in topics:
            raise ValueError("topic does not exist.")

        query = f"topic == {topic}"
        
        # Get summary for topic
        topic_summary = self.get_summary().query(query)

        # Get top n terms for topic
        top_n_terms_topic = self.get_top_n_terms().query(query).sort_values(by = "score", ascending = True)

        # Visualize the top 15 terms in the topic
        fig = px.bar(top_n_terms_topic, 
                    x="score", y="term",
                    orientation="h",
                    title=f"<b>Top 15 Terms in Topic {topic}<b>",
                    labels={"term": "Term", "score": "TF-IDF Score"})
        fig.show()

        # Print out the top 10 sentences most representative of the topic
        print("\033[1m" + "Top 10 Sentences:\n" + "\033[0m")
        for doc in topic_summary.document:
            print(doc + "\n")


    def get_topics_info(self) -> None:
        """
            Method to plot the intertopic distance map of the topics.

            Returns
            ----------
            None
        """
        # Compress the embeddings to 2D with UMAP
        n_neighbors = len(self.topic_vectors) - 1
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=2, 
            metric="cosine",
            random_state=self.seed
            )

        embeddings = umap_model.fit(self.topic_vectors).embedding_
        topic_sizes = self.get_topic_sizes()

        # Get the top 5 terms from each topic and join them into a single string per topic
        top_n_terms = self.get_top_n_terms().groupby("topic") \
                                            .head(5) \
                                            .groupby("topic", as_index = False) \
                                            .agg({"term": " | ".join})

        # Join the embeddings dataframe with the topic_sizes and top _n_terms dataframes
        data = pd.DataFrame(embeddings, columns = ["x", "y"]) \
                            .merge(topic_sizes, left_index = True, right_index = True) \
                            .merge(top_n_terms, on = "topic") \
                            .rename(columns = {"topic": "Topic", "count": "Size", "term": "Terms"})

        # Convert Topic column to string
        data.Topic = data.Topic.astype(str)

        # Visualize the topics with an intertopic distance map
        fig = px.scatter(data, x="x", y="y",
                        color="Topic", size="Size",
                        hover_data={
                            "Topic": True, 
                            "x": False, 
                            "y": False, 
                            "Size": True, 
                            "Terms": True
                            },
                        title="<b>Intertopic Distance Map<b>")

        fig.update_layout(xaxis_title=None, yaxis_title=None, 
                          xaxis_showticklabels=False, yaxis_showticklabels=False)
        fig.show()


    def predict(self, documents: Union[List[str], pd.Series], threshold: float = 0.3, batch_size: int = 64) -> pd.DataFrame:
        """
            Method to predict which topic each document belongs to based on the
            fitted topic model

            Args
            ----------
            documents: list or pd.Series of strings
                    input text corpus.

            threshold: float (Optional, default 0.3)
                    minimum threshold for a document to belong to a topic 

            batch_size: int (Optional, default 64)
                    number of documents passed to the model per iteration.

            Returns
            ----------
            prediction results: pd.DataFrame
                    the predicted topic for each document           
        """
        # Validate documents
        if not isinstance(documents, (list, pd.Series)):
            raise TypeError("documents need to a list or pandas series of strings.")
            
        if not all(isinstance(document, str) for document in documents):
            raise TypeError("documents need to a list or pandas series of strings.")

        # Validate threshold
        if not isinstance(threshold, (float, int)):
            raise TypeError("threshold needs to be an int or a float.")

        if not (0 <= threshold <= 1):
            raise ValueError("threshold needs to between 0 and 1.")

        # Get document emebddings
        document_embeddings = self.embedding_model.encode(documents,
                                                          convert_to_numpy = True,
                                                          normalize_embeddings = True)

        # Calculate cosine similarity wt=ith topic vectors
        doc_top, doc_dist = [], []
        for start_index in range(0, len(document_embeddings), batch_size):
            res = np.inner(document_embeddings[start_index: start_index + batch_size], 
                           self.topic_vectors)
            doc_top.extend(np.argmax(res, axis = 1))
            doc_dist.extend(np.max(res, axis = 1))
        
        # Filter out documents that are below the threshold and replace their topic wth -1
        doc_top = [top if dist >= threshold else -1 for top, dist in zip(np.array(doc_top), np.array(doc_dist))]

        return pd.DataFrame({"document": documents, "topic": doc_top})
    
  
    @staticmethod
    def l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
            Method to scale input vectors individually to unit l2 norm (vector length).

            Args
            ----------
            vectors: np.ndarray
                    the data to normalize.

            Returns
            ----------
            normalized vectors: np.ndarray
                    normalized input vectors.
        """
        if vectors.ndim == 2:
            return normalize(vectors)
        return normalize(vectors.reshape(1, -1))[0]