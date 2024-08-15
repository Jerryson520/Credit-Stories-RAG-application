"""
Training Module for the TreppWire Retrieval-Augmented Generation (RAG) System

This module provides functions to train a final model using the TreppWire RAG system. 
It includes functionalities for loading data from S3, creating embeddings, 
and querying the model with user-defined questions.

Modules:
--------
- SecretManager: Manages secrets for API access.
- OpenAIClient: Interfaces with OpenAI's API for model interactions.
- S3ParquetLoader: Loads Parquet files from an S3 bucket.
- TpwireDataLoader: Loads TreppWire data from an S3 bucket.
- tpwireRAG: Implements the Retrieval-Augmented Generation (RAG) model for TreppWire.

Usage:
------
To run the script, use the command line with the required question argument:
    python training.py --question "Your question here"
"""

import os, json
from langchain_openai import OpenAIEmbeddings
from utils_rag import SecretManager, OpenAIClient, S3ParquetLoader, TpwireDataLoader, tpwireRAG
from config import SECRET_NAME, BUCKET_NAME, PREFIX, MODEL, EMBED_MODEL_NAME, K1, K2
import argparse


def main(input_question):
    """
    Main function to process the input question and return the model's answer.

    Args:
        input_question (str): The question to be asked to the TreppWire RAG system.

    Returns:
        str: The answer generated by the model in response to the input question.
    """
    
    secret_manager = SecretManager(secret_name=SECRET_NAME)
    os.environ['OPENAI_API_KEY'] = secret_manager.get_secret('OPENAI_API_KEY')
    
    embed = OpenAIEmbeddings(model=EMBED_MODEL_NAME) # Create embeddings for indexing documents
    
    openai_client = OpenAIClient(model=MODEL)

    # S3 parquet loader
    s3_parquet_loader = S3ParquetLoader(bucket_name=BUCKET_NAME, prefix=PREFIX)

    # Import TreppWire data
    tpwire_loader = TpwireDataLoader(bucket_name=BUCKET_NAME, prefix=PREFIX)

    # Initialize treppwire RAG
    tpwire_RAG = tpwireRAG(client=openai_client, embed=embed)
    
    tpwire_df = tpwire_loader.process_data("tpwire_flags.parquet")
    
    
    # index_tb = index_document(tpwire_df, embed)
    # index_tb = tpwire_RAG.index_document(tpwire_df)
    # index_tb.to_parquet('treppwire_index_tb.parquet')
    
    # index_tb = pd.read_parquet('treppwire_index_tb.parquet')
    index_tb = s3_parquet_loader.load_s3_parquet('treppwire_index_tb.parquet')
    
    tpwire_RAG.build_vec_db(index_tb=index_tb)
    
    
    # search_text = 'What is the recent price of Flagler Corporate Center loan?'
    search_text = input_question
    
    relevant_contents1, relevant_contents, prompt = tpwire_RAG.generate_context_prompt(search_text=search_text, K1=K1, K2=K2)
    response = openai_client.get_completion(message=prompt)
    return json.loads(response.choices[0].message.content)['Answer']
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ask a question to the TreppWire RAG system')
    parser.add_argument('--question', metavar='question', required=True,
                        help='The question you want to ask')
    args = parser.parse_args()
    print(f"The answer is: {main(args.question)}")