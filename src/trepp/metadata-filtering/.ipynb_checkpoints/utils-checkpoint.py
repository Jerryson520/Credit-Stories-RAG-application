"""
Utility Module for Data Processing and API Interactions

This module provides utility functions and classes for processing data, 
interacting with AWS services, and utilizing OpenAI's API for generating 
responses in a Retrieval-Augmented Generation (RAG) context.

Key functionalities include:
- Date conversion utilities.
- Recursive value conversion for nested structures.
- Classes for managing AWS Secrets and OpenAI API interactions.
- Loading data from S3 in Parquet format.

Imports:
--------
- Standard libraries: os, json, re, random, time, math
- Data manipulation: pandas, numpy
- Date handling: datetime
- AWS SDK: boto3
- OpenAI API: openai
- Parquet file handling: pyarrow.parquet
"""


import re, os, random, time, math, json, pandas as pd, numpy as np
from datetime import date, datetime
import boto3
from openai import OpenAI
import pyarrow.parquet as pq



# utils functions
def date_to_int(date_str):
    """Convert a date string in 'YYYY-MM-DD' format to a Unix timestamp.

    Args:
        date_str (str): The date string to convert.

    Returns:
        int: The Unix timestamp corresponding to the date.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())


def convert_values(obj):
    """Recursively convert values in a nested structure to appropriate types.

    Args:
        obj: The input object which can be a dict, list, string, or other types.

    Returns:
        The converted object with values in appropriate types.
    """
    
    # Check for the specific case
    if isinstance(obj, dict) and obj == {'publishdate': None}:
        return None

    if isinstance(obj, dict):
        return {k: convert_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_values(v) for v in obj]
    elif isinstance(obj, str):
        # Try to convert to int or float
        try:
            return int(obj)
        except ValueError:
            try:
                return float(obj)
            except ValueError:
                # Check if it's a date string (assuming YYYY-MM-DD format)
                try:
                    return date_to_int(obj)
                except ValueError:
                    return obj
    else:
        return obj


def process_llm_output(llm_output):
    """Process the LLM output string into a usable ChromaDB filter.

    Args:
        llm_output (str): The output string from the LLM.

    Returns:
        The converted output or None if there's an error.
    """
    
    try:
        # Convert string values to appropriate types
        converted = convert_values(llm_output)
        
        return converted
    except json.JSONDecodeError:
        print("Error: Invalid JSON in LLM output")
        return None



# useful classes
class SecretManager:
    """Handles retrieval of secrets from AWS Secrets Manager.

    Attributes:
        secret_name (str): The name of the secret to retrieve.
        region_name (str): The AWS region where the secret is stored.
        client (boto3.client): The AWS Secrets Manager client.
    """
    
    def __init__(self, secret_name, region_name="us-east-1"):
        """Initializes the SecretManager with the given secret name and region.

        Args:
            secret_name (str): The name of the secret.
            region_name (str): The AWS region (default is "us-east-1").
        """
        
        self.secret_name = secret_name
        self.region_name = region_name
        self.client = boto3.session.Session().client(
            service_name='secretsmanager', 
            region_name=region_name
        )
        
    def get_secret(self, api_key_name):
        """Retrieves the specified secret from AWS Secrets Manager.

        Args:
            api_key_name (str): The name of the API key to retrieve.

        Returns:
            str: The value of the requested secret.

        Raises:
            ClientError: If the secret cannot be retrieved.
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
    
    
    
class OpenAIClient:
    """Interfaces with the OpenAI API for generating text completions.

    Attributes:
        model (str): The model to use for generating completions.
        seed (int): The random seed for generation.
        temperature (float): The temperature for randomness in responses.
        tools (list): Optional tools for the client.
        client (OpenAI): The OpenAI client instance.
    """
    
    def __init__(self, model='gpt-4o-mini', seed=12345, temperature=0, tools=None):
        """Initializes the OpenAIClient with the specified parameters.

        Args:
            model (str): The model to use (default is 'gpt-4o-mini').
            seed (int): The random seed (default is 12345).
            temperature (float): The temperature for randomness (default is 0).
            tools (list): Optional tools for the client (default is None).
        """
        
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.tools = tools
        self.client = OpenAI()
    
    def get_completion(self, message):
        """Generates a completion for the given message using the OpenAI API.

        Args:
            message (str): The message to generate a completion for.

        Returns:
            dict: The response from the OpenAI API.
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
    
    
    
    
# S3ParquetLoader & TpwireDataLoader
class S3ParquetLoader:
    """Loads parquet files from an S3 bucket.

    Attributes:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix for the S3 path.
        file_identifier (str): The identifier for S3 path (default is "s3://").
    """
    
    def __init__(self, bucket_name, prefix, file_identifier="s3://"):
        """Initializes the S3ParquetLoader with the bucket name and prefix.

        Args:
            bucket_name (str): The name of the S3 bucket.
            prefix (str): The prefix for the S3 path.
            file_identifier (str): The identifier for S3 path (default is "s3://").
        """
        
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.file_identifier = file_identifier
        
    def load_s3_parquet(self, file):
        """Loads a parquet file from S3 and returns it as a pandas DataFrame.

        Args:
            file (str): The name of the parquet file to load.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        
        s3_path = self.file_identifier + self.bucket_name + self.prefix + file
        return pq.read_table(s3_path).to_pandas()