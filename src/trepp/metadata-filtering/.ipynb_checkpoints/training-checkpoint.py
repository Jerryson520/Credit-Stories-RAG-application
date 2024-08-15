"""
Contains functions needed to train final model
"""
import os, json
from langchain_openai import OpenAIEmbeddings
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings
from utils import date_to_int, convert_values, process_llm_output
from utils import SecretManager, OpenAIClient, S3ParquetLoader
from utils_rag import TpwireDataLoader, IndexDoc, tpwireDB, tpwireRAG, df_db_pre
from config import SECRET_NAME, BUCKET_NAME, PREFIX, MODEL, DBNAME, EMBED_MODEL_NAME, K
import argparse


def main(input_question):
    secret_manager = SecretManager(secret_name=SECRET_NAME)
    os.environ['OPENAI_API_KEY'] = secret_manager.get_secret('OPENAI_API_KEY')

    embed = OpenAIEmbeddings(model=EMBED_MODEL_NAME) # Create embeddings for indexing documents
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ['OPENAI_API_KEY'], 
        model_name=EMBED_MODEL_NAME)
    
    
    # class initialization
    openai_client = OpenAIClient(model=MODEL)
    db_client = chromadb.PersistentClient(path='./treppwire-chromaDB')

    # S3 parquet loader
    s3_parquet_loader = S3ParquetLoader(bucket_name=BUCKET_NAME, prefix=PREFIX)

    # Import TreppWire data
    tpwire_loader = TpwireDataLoader(bucket_name=BUCKET_NAME, prefix=PREFIX)
    
    
    # Index_Doc = IndexDoc(embed)
    tpwire_df = tpwire_loader.process_data("tpwire_flags.parquet")
    tpwire_df.head()
    
    index_tb = s3_parquet_loader.load_s3_parquet('treppwire_index_tb.parquet')
    original_df = df_db_pre(index_tb, tpwire_df)
    
    # VectorDB
    # tpwire_DB.delete()
    # tpwire_DB.collection_exists()
    tpwire_DB = tpwireDB(db_client, DBNAME, openai_ef)
    # tpwire_DB.add(original_df=original_df)
    
    
    # Initialize treppwire RAG
    tpwire_RAG = tpwireRAG(tpwire_DB, original_df, openai_client, embed)
    # input_question = 'What are the 5 transactions in New York in Dec, 2023?'
    # input_question = 'How many properties in New York City are currently facing foreclosure?'
    # input_question = 'What significant events have occurred for Chicago Hotel?'
    search_text = input_question # 'What are the 5 transactions in New York in Dec, 2023?'
    # search_text = 'How many properties in New York City are currently facing foreclosure?'

    relevant_contents, prompt = tpwire_RAG.generate_context_prompt(search_text=search_text, K=K)
    response = openai_client.get_completion(message=prompt)
    return json.loads(response.choices[0].message.content)['Answer']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ask a question to the TreppWire RAG system')
    parser.add_argument('--question', metavar='question', required=True,
                        help='The question you want to ask')
    args = parser.parse_args()
    print(f"The answer is: {main(args.question)}")