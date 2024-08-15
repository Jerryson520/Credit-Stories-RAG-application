"""
Module: tpwire_data_loader

This module provides classes and methods to load, process, and manage data 
from an S3 bucket, specifically targeting Parquet files. It also includes 
functionality to index documents and interact with a database for 
retrieval and storage operations.

Dependencies:
- json
- os
- datetime
- pandas
- numpy
- boto3
- openai
- pyarrow
- utils (custom module)
"""



import json
import os
from datetime import date, datetime

import pandas as pd
import numpy as np
import boto3
from openai import OpenAI  # Ensure you have the correct OpenAI client
import pyarrow.parquet as pq  # For reading Parquet files

# Import your custom classes or functions if they are in a separate module
from utils import S3ParquetLoader, SecretManager, OpenAIClient, process_llm_output


class TpwireDataLoader:
    """
    Class to load and process data from an S3 bucket containing Parquet files.

    Attributes:
        s3_loader (S3ParquetLoader): Instance of S3ParquetLoader to handle S3 operations.
    """
    
    def __init__(self, bucket_name, prefix, file_identifier="s3://"):
        self.s3_loader = S3ParquetLoader(bucket_name, prefix, file_identifier)
        
    def process_data(self, file_name):
        """
        Loads and processes the Parquet file from S3.

        Args:
            file_name (str): The name of the Parquet file to load.

        Returns:
            pd.DataFrame: A DataFrame containing processed data.
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


class IndexDoc:
    """
    Class to handle indexing of documents.

    Attributes:
        embed (OpenAIClient): Instance of OpenAIClient for embedding documents.
    """
    
    def __init__(self, embed):
        """
        Initializes the IndexDoc with an embedding client.

        Args:
            embed (OpenAIClient): Client for embedding documents.
        """
        
        self.embed = embed
        
    def index_document(self, df):
        """
        Indexes the documents from the provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing documents to index.

        Returns:
            pd.DataFrame: A DataFrame containing indexed documents and their embeddings.
        """
        
        # for idx in range(len(df)):
        chunks = df['summaryplaintext'].tolist()

        index_tb = pd.DataFrame({
            'Content': chunks,
            'Vectors': embed.embed_documents(chunks), # 'original_index': range(len(chunks)),
        })

        index_tb.reset_index(drop=True, inplace=True)

        return index_tb
    
    
    

class tpwireDB:
    """
    Class to manage a database for storing and retrieving documents.

    Attributes:
        client (boto3.client): The database client.
        name (str): The name of the collection.
        embedding_function (OpenAIClient): The embedding function used for documents.
        collection: The collection object for database operations.
    """
    
    def __init__(self, client, name, openai_ef):
        """
        Initializes the tpwireDB with client and collection details.

        Args:
            client (boto3.client): The database client.
            name (str): The name of the collection.
            openai_ef (OpenAIClient): The embedding function for documents.
        """
        
        self.client = client
        self.name = name
        self.embedding_function = openai_ef
        self.collection = self.get_or_create()
    
    # Collection methods
    def get_or_create(self):
        """
        Retrieves or creates a collection in the database.

        Returns:
            Collection: The collection object.
        """
        
        try:
            return self.client.get_or_create_collection(
                name=self.name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error getting or creating collection: {e}")
            return None
    
    def delete(self):
        """
        Deletes the collection from the database.
        """
        
        try:
            self.client.delete_collection(name=self.name)
            print(f"Collection '{self.name}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def collection_exists(self):
        """
        Checks if the collection exists in the database.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        
        try:
            collections = self.client.list_collections()
            return any(collection.name == self.name for collection in collections)
        except Exception as e:
            print(f"Error checking if collection exists: {e}")
            return False
       
    # Elements methods
    def document_exists(self, ids):
        """
        Checks if documents exist in the collection.

        Args:
            ids (list): List of document IDs to check.

        Returns:
            list: A list of booleans indicating existence of each document.
        """
        
        try:
            result = self.collection.get(ids=ids)
            return [id in result['ids'] for id in ids]
        except Exception as e:
            print(f"Error checking document existence: {e}")
            return [False] * len(ids)

    
    def add(self, original_df):
        """
        Adds new documents to the collection.

        Args:
            original_df (pd.DataFrame): DataFrame containing documents to add.
        """
        
        if self.collection is None:
            print("Collection not initialized. Cannot add data.")
            return
        
        # ids
        ids = [str(i) for i in original_df.index.tolist()] 
         
        # Check for existing IDs
        existing_ids = self.document_exists(ids)

        if any(existing_ids):
            existing_indices = [i for i, exists in enumerate(existing_ids) if exists]
            existing_id_values = [ids[i] for i in existing_indices]
            print(f"The following IDs already exist: {existing_id_values}. Cannot add data, try using upsert.")
            return

        # embeddings
        embeddings = [list(vec) for vec in original_df['Vectors'].values]

        # metadatas
        original_df = original_df.drop(['Content', 'Vectors'], axis=1)
        metadata_lst = []
        # for i, row in original_df.iterrows():
        #     meta_dict = {col: row[col][1:-1] if col != 'publishdate' else date_to_int(str(row[col])) for col in original_df.columns}
        #     metadata_lst.append(meta_dict)
        for i, row in original_df.iterrows():
            meta_dict = {
                'publishdate': date_to_int(
                    str(row['publishdate'])
                )
            }
            metadata_lst.append(meta_dict)
        

        self.collection.add(
            embeddings=embeddings,
            metadatas=metadata_lst,
            ids=ids
        )
    
    def upsert(self, new_df):
        """
        Updates or inserts documents in the collection.

        Args:
            new_df (pd.DataFrame): DataFrame containing documents to upsert.
        """
        
        if self.collection is None:
            print("Collection not initialized. Cannot upsert data.")
            return
        
        
        # ids
        ids = [str(i) for i in new_df.index.tolist()] 
        # embeddings
        embeddings = [list(vec) for vec in new_df['Vectors'].values]
        
        # metadatas
        new_df = new_df.drop(['Content', 'Vectors'], axis=1)
        metadata_lst = []
        # for i, row in new_df.iterrows():
        #     meta_dict = {col: row[col][1:-1] if col != 'publishdate' else date_to_int(str(row[col])) for col in new_df.columns}
        #     metadata_lst.append(meta_dict)
        
        for i, row in original_df.iterrows():
            meta_dict = {
                'publishdate': date_to_int(
                    str(row['publishdate'])
                )
            }
            metadata_lst.append(meta_dict)

        try:
            self.collection.upsert(
                embeddings=embeddings,
                metadatas=metadata_lst,
                ids=ids
            )
            print(f"Upserted {len(ids)} items in the collection.")
        except Exception as e:
            print(f"Error upserting data in collection: {e}")
        
    
    def retrieve(self, search_vectors, K, where=None):
        """
        Retrieves documents based on search vectors.

        Args:
            search_vectors (list): List of vectors for searching.
            K (int): Number of results to retrieve.
            where (dict, optional): Filter conditions for the query.

        Returns:
            dict: Results from the database query.
        """
        
        query_params = {
            "query_embeddings": search_vectors,
            "n_results": K,
            "include": ["distances"]
            # "include_metadata": True
        }
        
        if where:
            query_params["where"] = where # {"$and": [{"state": "NY"}, {"publishdate": {"$gte": date_to_int("2024-01-01")}}]}
    
        try:
            results = self.collection.query(**query_params)
            # results = self.collection.similarity_search_by_vector(**query_params)
            return results
        except Exception as e:
            print(f"Error retrieving data from collection: {e}")
            return None
        

        
class tpwireRAG:
    """
    Class to manage retrieval-augmented generation (RAG) operations.

    Attributes:
        db (tpwireDB): Instance of tpwireDB for database operations.
        original_df (pd.DataFrame): Original DataFrame containing documents.
        client (OpenAIClient): Client for OpenAI operations.
        embed (OpenAIClient): Client for embedding documents.
        prompt_template (str): Template for generating prompts.
    """
    
    def __init__(self, db, original_df, client, embed):
        """
        Initializes the tpwireRAG with necessary components.

        Args:
            db (tpwireDB): Database instance for operations.
            original_df (pd.DataFrame): Original DataFrame of documents.
            client (OpenAIClient): OpenAI client for API calls.
            embed (OpenAIClient): Client for embedding documents.
        """
        
        self.db = db
        self.original_df = original_df
        self.client = client
        self.embed = embed
        self.prompt_template = '''
        Use the following pieces of context to help you answer the question at the end. 

        If you don't know the answer, just output: "Answer": <I don't know the answer>. Don't try to make up an answer. 

        Keep the answer as concise as possible.

        Context: {context}
        Question: {search_text}
        
        "Answer": <answer> 
        ''' # Always say "thanks for asking!" at the end of the answer.
        self.metadata_prompt_template = """
        Please extract information from the given sentence and format it according to the ChromaDB filter standard. Follow these steps:

        1. Entity Extraction:
            - publishdate: Identify time-related keywords. If it is missing, ignore it.

        2. publishdate interpretation:
           - "recent" or "latest": Use date range from 30 days ago to today
           - "this quarter": Use range from start of current quarter to today
           - "in [month]": Use range from start to end of specified month
           - "in [month], [year]": Use range from start to end of specified month and year
           - Specific dates: Use as given
           - All dates should be should be in the standard yyyy-mm-dd format.

        3. Formatting:
           - If the publishdate is missing, just ignore it output {{'publishdate': None}}
           - Use the ChromaDB filter standard
           - For date ranges, use separate "$gte" and "$lte" conditions

        Examples:

        1. "Find office properties in New York City listed this quarter"
        {{
            "$and": [
                {{
                    "publishdate": {{"$gte": "2024-07-01"}}
                }},
                {{
                    "publishdate": {{"$lte": "2024-07-31"}}
                }}
            ]
        }}

        2. "Show recent retail listings in Los Angeles County, CA"
        {{
            "$and": [
                {{
                    "publishdate": {{"$gte": "2024-07-01"}}
                }},
                {{
                    "publishdate": {{"$lte": "2024-07-31"}}
                }}
            ]
        }}

        3. "What are the 5 transactions in New York in Dec, 2023?"
        {{
            "$and": [
                {{
                    "publishdate": {{"$gte": "2023-12-01"}}
                }},
                {{
                    "publishdate": {{"$lte": "2023-12-31"}}
                }}
            ]
        }}

        Please provide the ChromaDB filter for the following sentence:
        Input sentence: {sentence}
        """
    
    def retrieve_relevant(self, search_text, K, where=None): # , where=None
        """
        Retrieves relevant documents based on the search text.

        Args:
            search_text (str): The text to search for.
            K (int): Number of results to retrieve.
            where (dict, optional): Filter conditions for the query.

        Returns:
            pd.DataFrame: DataFrame containing relevant documents and their distances.
        """
        
        response = self.client.get_completion(message= self.metadata_prompt_template.format(sentence=search_text))
        where_filter = json.loads(response.choices[0].message.content)
        processed_filter = process_llm_output(where_filter)
        
        search_vectors = self.embed.embed_documents([search_text])
        results = self.db.retrieve(search_vectors, K, where=processed_filter) # , where=processed_filter
        
        idxs = [int(idx) for idx in results['ids'][0]]
        return pd.concat([pd.DataFrame({"distances": results['distances'][0]}, index=idxs), self.original_df.loc[idxs]], axis=1)

    
    @staticmethod
    def combine_context(relevant_contents): # Helper function for generating context
        """
        Combines relevant contents into a single context string.

        Args:
            relevant_contents (pd.DataFrame): DataFrame containing relevant documents.

        Returns:
            str: Combined context string.
        """
        
        context = relevant_contents.Content
        context = '\n'.join(context)

        return context
    
    def generate_context_prompt(self, search_text, K, where=None): # , where=None
        """
        Generates a prompt for the OpenAI model based on the search text.

        Args:
            search_text (str): The text to search for.
            K (int): Number of results to retrieve.
            where (dict, optional): Filter conditions for the query.

        Returns:
            tuple: A tuple containing relevant contents and the generated prompt.
        """
        
        relevant_contents = self.retrieve_relevant(search_text=search_text, K=K)
        context = self.combine_context(relevant_contents=relevant_contents)
        prompt = self.prompt_template.format(context=context, search_text=search_text)
        return relevant_contents, prompt 
    
    
    
def df_db_pre(index_tb, tpwire_df):
    cols = ['dealname', 'propertysubtype', 'proptypecode', 'state', 'county', 'city', 'msaname', 'loanname', 'region', 'publishdate']
    df_data = tpwire_df[cols]
    
    return pd.concat([index_tb, df_data], axis=1)





