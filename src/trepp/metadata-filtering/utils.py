import re, os, random, time, math, json, pandas as pd, numpy as np
from datetime import date, datetime
import boto3
from openai import OpenAI
import pyarrow.parquet as pq



# utils functions
def date_to_int(date_str):
    """Convert a date string to Unix timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

# def convert_values(obj):
#     """Recursively convert values in a nested structure."""
#     if isinstance(obj, dict):
#         return {k: convert_values(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_values(v) for v in obj]
#     elif isinstance(obj, str):
#         # Try to convert to int or float
#         try:
#             return int(obj)
#         except ValueError:
#             try:
#                 return float(obj)
#             except ValueError:
#                 # Check if it's a date string (assuming YYYY-MM-DD format)
#                 try:
#                     return date_to_int(obj)
#                 except ValueError:
#                     return obj
#     else:
#         return obj

# def process_llm_output(llm_output):
#     """Process the LLM output string into a usable ChromaDB filter."""
#     try:
#         # Convert string values to appropriate types
#         converted = convert_values(llm_output)
        
#         return converted
#     except json.JSONDecodeError:
#         print("Error: Invalid JSON in LLM output")
#         return None

def convert_values(obj):
    """Recursively convert values in a nested structure."""
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
    """Process the LLM output string into a usable ChromaDB filter."""
    try:
        # Convert string values to appropriate types
        converted = convert_values(llm_output)
        
        return converted
    except json.JSONDecodeError:
        print("Error: Invalid JSON in LLM output")
        return None



# useful classes
class SecretManager: # secretmanager
    def __init__(self, secret_name, region_name="us-east-1"):
        self.secret_name = secret_name
        self.region_name = region_name
        self.client = boto3.session.Session().client(
            service_name='secretsmanager', 
            region_name=region_name
        )
        
    def get_secret(self, api_key_name):
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
    
    
    
class OpenAIClient: # openai client
    def __init__(self, model='gpt-4o-mini', seed=12345, temperature=0, tools=None):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.tools = tools
        self.client = OpenAI()
    
    def get_completion(self, message):
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
    def __init__(self, bucket_name, prefix, file_identifier="s3://"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.file_identifier = file_identifier
        
    def load_s3_parquet(self, file):
        s3_path = self.file_identifier + self.bucket_name + self.prefix + file
        return pq.read_table(s3_path).to_pandas()