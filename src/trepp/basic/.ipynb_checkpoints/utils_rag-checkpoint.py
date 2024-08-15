"""
Utility Functions and Classes for Data Processing and Management

This module contains various utility classes and functions to facilitate data loading, processing, and interaction with AWS services.
It includes classes for managing secrets, loading data from S3, and interfacing with OpenAI's API.

Dependencies:
-------------
- boto3: For interacting with AWS services.
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- pyarrow: For reading Parquet files.
- openai: For accessing OpenAI's API.
- faiss: For efficient similarity search and clustering of dense vectors.
- langchain: For handling text processing and embeddings.

Classes:
--------
- SecretManager: Manages AWS Secrets Manager to retrieve sensitive information.
- S3ParquetLoader: Loads Parquet files from an S3 bucket.
- OpenAIClient: Interacts with OpenAI's API to generate responses.
- TpwireDataLoader: Processes data specifically from TreppWire.
- tpwireRAG: Implements Retrieval Augmented Generation functionality for processing and retrieving relevant data.

Usage:
------
To use the utilities in this module, instantiate the desired class and call its methods as needed.
"""


import re, os, random, time, json, math
from datetime import date, datetime
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from pyathena import connect
import boto3
from botocore.exceptions import ClientError
import openai
from openai import OpenAI
import pyarrow.parquet as pq
import tiktoken
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


class SecretManager:
    """
    A class to manage AWS Secrets Manager for retrieving secrets.

    Attributes:
        secret_name (str): The name of the secret in AWS Secrets Manager.
        region_name (str): The AWS region where the secret is stored.

    Methods:
        get_secret(api_key_name): Retrieves the specified secret from AWS Secrets Manager.
    """
    
    def __init__(self, secret_name, region_name="us-east-1"):
        self.secret_name = secret_name
        self.region_name = region_name
        self.client = boto3.session.Session().client(
            service_name='secretsmanager', 
            region_name=region_name
        )
        
    def get_secret(self, api_key_name):
        """
        Retrieves a specific secret from AWS Secrets Manager.

        Args:
            api_key_name (str): The name of the API key to retrieve from the secret.

        Returns:
            str: The value of the specified secret.

        Raises:
            ClientError: If there is an error retrieving the secret.
        """
        
        try:
            get_secret_value_response = self.client.get_secret_value(
                SecretId=self.secret_name
            )
        except ClientError as e:
            # For a list of exceptions thrown, see
            # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
            raise e

        secret = get_secret_value_response['SecretString']

        return json.loads(secret)[api_key_name]
    
    
    
    
class S3ParquetLoader:
    """
    A class to load Parquet files from an S3 bucket.

    Attributes:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix path in the S3 bucket.
        file_identifier (str): The identifier for S3 file paths.

    Methods:
        load_s3_parquet(file): Loads a Parquet file from S3 and returns it as a pandas DataFrame.
    """
    
    def __init__(self, bucket_name, prefix, file_identifier="s3://"):
        
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.file_identifier = file_identifier
        
    def load_s3_parquet(self, file):
        """
        Loads a Parquet file from S3 and converts it to a pandas DataFrame.

        Args:
            file (str): The name of the Parquet file to load.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        
        s3_path = self.file_identifier + self.bucket_name + self.prefix + file
        return pq.read_table(s3_path).to_pandas()
    
    
    
    
class OpenAIClient:
    """
    A class to interact with OpenAI's API for generating responses.

    Attributes:
        model (str): The OpenAI model to use for generating responses.
        seed (int): Seed for random number generation.
        temperature (float): Sampling temperature for response generation.
        tools (optional): Additional tools for the client.

    Methods:
        get_completion(message): Generates a completion response from OpenAI based on the input message.
    """
    
    def __init__(self, model='gpt-4o-mini', seed=12345, temperature=0, tools=None):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.tools = tools
        self.client = OpenAI()
    
    def get_completion(self, message):
        """
        Generates a completion response from OpenAI based on the input message.

        Args:
            message (str): The message to send to the OpenAI model.

        Returns:
            dict: The response from OpenAI.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {"role": "user", "content": message},
            ],
            temperature=self.temperature,
            # max_tokens=max_tokens,
            seed=self.seed,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0 # seed=1001
    )
    
        return response # response_format={ "type": "json_object"}, designed to output JSON
    
    
    
class TpwireDataLoader:
    """
    A class to load and process TreppWire data from S3.

    Attributes:
        s3_loader (S3ParquetLoader): An instance of S3ParquetLoader for loading data.

    Methods:
        process_data(file_name): Loads and processes TreppWire data from S3.
    """
    
    def __init__(self, bucket_name, prefix, file_identifier="s3://"):
        self.s3_loader = S3ParquetLoader(bucket_name, prefix, file_identifier)
        
    def process_data(self, file_name):
        """
        Loads and processes TreppWire data from S3.

        Args:
            file_name (str): The name of the Parquet file to process.

        Returns:
            pd.DataFrame: The processed TreppWire data as a pandas DataFrame.
        """
            
        data = self.s3_loader.load_s3_parquet(file_name)
        # final_df = data[['summaryplaintext', 'publishdate', 'multi_story_flag', 'table_flag']]
        final_df = data
        final_df = final_df[(final_df['multi_story_flag'] == False) & (final_df['table_flag'] == False)]
        
        # tpwire_df = final_df[['summaryplaintext', 'publishdate']]
        tpwire_df = final_df.drop(['multi_story_flag', 'table_flag'], axis=1)
        tpwire_df = tpwire_df.sort_values(by='publishdate').reset_index(drop=True)
        tpwire_df['publishdate'] = tpwire_df['publishdate'].apply(lambda x: np.datetime64(x).astype(datetime).date()) # change to standard date format
        
        return tpwire_df
    
    
class tpwireRAG:
    """
    A class for implementing Retrieval Augmented Generation (RAG) functionality.

    Attributes:
        embed (OpenAIEmbeddings): An instance of OpenAIEmbeddings for generating embeddings.
        index_tb (pd.DataFrame): The index table for storing document content and vectors.
        vec_db (faiss.IndexFlatL2): The vector database for similarity search.
        prompt_template (str): The template for generating prompts for the OpenAI model.

    Methods:
        index_document(df): Indexes documents from a DataFrame.
        build_vec_db(index_tb): Builds a vector database from the indexed table.
        retrieve_from_vec_db(search_text, K): Retrieves relevant documents from the vector database.
        combine_context(relevant_contents): Combines relevant contents into a single context string.
        generate_context_prompt(search_text, K): Generates a context prompt for the OpenAI model.
    """
    
    def __init__(self, embed):
        self.embed = embed
        self.index_tb = None
        self.vec_db = None
        self.prompt_template = '''
        Use the following pieces of context to help you answer the question at the end. 

        If you don't know the answer, just output: "Answer": <I don't know the answer>. Don't try to make up an answer. 

        Keep the answer as concise as possible.

        Context: {context}
        Question: {search_text}
        
        "Answer": <answer> 
        ''' # Always say "thanks for asking!" at the end of the answer. 
        
    def index_document(self, df):
        """
        Indexes documents from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing documents to index.

        Returns:
            pd.DataFrame: The indexed table containing document content and vectors.
        """
        
        # for idx in range(len(df)):
        chunks = df['summaryplaintext'].tolist()
        
        self.index_tb = pd.DataFrame({
        'Content': chunks,
        'Vectors': embed.embed_documents(chunks), # 'original_index': range(len(chunks)),
        })

        self.index_tb.reset_index(drop=True, inplace=True)

        return self.index_tb
    
    def build_vec_db(self, index_tb):
        """
        Builds a vector database from the indexed table.

        Args:
            index_tb (pd.DataFrame): The indexed table containing document vectors.
        """
        
        if not self.index_tb:
            self.index_tb = index_tb
        
        vectors = np.array(self.index_tb['Vectors'].tolist(), dtype='float32')
        vec_dim = vectors.shape[1]
        self.vec_db = faiss.IndexFlatL2(vec_dim)
        faiss.normalize_L2(vectors)
        self.vec_db.add(vectors)

        # return self.vec_db
        
        
    def retrieve_from_vec_db(self, search_text, K):
        """
        Retrieves relevant documents from the vector database based on the search text.

        Args:
            search_text (str): The text to search for in the vector database.
            K (int): The number of nearest neighbors to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the distances and relevant documents.
        """
        
        search_vectors = self.embed.embed_documents([search_text])
        faiss.normalize_L2(np.array(search_vectors, dtype='float32'))

        distances, ann = self.vec_db.search(np.array(search_vectors, dtype='float32'), k=K)

        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

        merge = pd.merge(results, self.index_tb, left_on='ann', right_index=True)

        # # If you need data from the original dataframe:
        # merge = pd.merge(merge, original_df, left_on='original_index', right_index=True)

        return merge
    
    @staticmethod
    def combine_context(relevant_contents): # Helper function for generating context
        """
        Combines relevant contents into a single context string.

        Args:
            relevant_contents (pd.DataFrame): A DataFrame containing relevant document contents.

        Returns:
            str: The combined context string.
        """
        
        context = relevant_contents.Content
        context = '\n'.join(context)

        return context
    
    def generate_context_prompt(self, search_text, K):
        """
        Generates a context prompt for the OpenAI model based on the search text.

        Args:
            search_text (str): The text to search for in the vector database.
            K (int): The number of nearest neighbors to retrieve.

        Returns:
            tuple: A tuple containing the relevant contents and the generated prompt.
        """
        relevant_contents = self.retrieve_from_vec_db(search_text=search_text, K=K)
        context = self.combine_context(relevant_contents=relevant_contents)
        prompt = self.prompt_template.format(context=context, search_text=search_text)
        return relevant_contents, prompt 
    
    
