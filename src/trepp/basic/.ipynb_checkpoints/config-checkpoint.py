"""
Configuration Documentation for TreppWire RAG System

This module contains configuration parameters required for the TreppWire Retrieval Augmented Generation (RAG) system.
These parameters are essential for setting up the environment and ensuring the system operates correctly.

Constants:
-----------
SECRET_NAME: str
    The name of the secret in AWS Secrets Manager that contains sensitive information, such as the OpenAI API key.
BUCKET_NAME: str
    The name of the S3 bucket where the data files are stored. This bucket is used for loading data into the system.
PREFIX: str
    The prefix path within the S3 bucket that specifies the location of the relevant data files.
MODEL: str
    The name of the OpenAI model to be used for generating responses. This model will process the input questions.
EMBED_MODEL_NAME: str
    The name of the OpenAI embedding model used for creating embeddings of the input data.
K: int
    The number of relevant chunks of data to retrieve when generating responses. This parameter controls the depth of the context provided to the model.
"""


SECRET_NAME= "AmazonSageMaker-sagemarker_yuwen"
BUCKET_NAME = "trepp-developmentservices-datascience/"
PREFIX = "llm-exploration/treppwire_rag/"

MODEL = 'gpt-4o-mini'
EMBED_MODEL_NAME = 'text-embedding-ada-002'
K = 5 # Number of relevant chunks