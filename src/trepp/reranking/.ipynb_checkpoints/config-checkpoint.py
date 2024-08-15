"""
Configuration settings for the TreppWire Retrieval-Augmented Generation (RAG) system.

This module contains constants for API key management, S3 bucket settings, 
OpenAI model configuration, and parameters for document retrieval.

Constants:
- SECRET_NAME: The name of the secret in AWS Secrets Manager for API access.
- BUCKET_NAME: The name of the S3 bucket where data is stored.
- PREFIX: The prefix for the S3 path to locate specific files.
- MODEL: The model name to be used for generating completions.
- EMBED_MODEL_NAME: The embedding model name for document indexing.
- K1: The number of top documents to retrieve during the first stage of retrieval.
- K2: The number of documents to rank in the second stage.
"""


SECRET_NAME= "AmazonSageMaker-sagemarker_yuwen"

BUCKET_NAME = "trepp-developmentservices-datascience/"
PREFIX = "llm-exploration/treppwire_rag/"

MODEL = 'gpt-4o-mini'
EMBED_MODEL_NAME = 'text-embedding-ada-002'
K1, K2 = 10, 5