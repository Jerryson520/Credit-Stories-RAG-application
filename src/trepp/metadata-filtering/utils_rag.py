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
    def __init__(self, bucket_name, prefix, file_identifier="s3://"):
        self.s3_loader = S3ParquetLoader(bucket_name, prefix, file_identifier)
        
    def process_data(self, file_name):
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
    def __init__(self, embed):
        self.embed = embed
        
    def index_document(self, df):
        # for idx in range(len(df)):
        chunks = df['summaryplaintext'].tolist()

        index_tb = pd.DataFrame({
            'Content': chunks,
            'Vectors': embed.embed_documents(chunks), # 'original_index': range(len(chunks)),
        })

        index_tb.reset_index(drop=True, inplace=True)

        return index_tb
    
    
    

class tpwireDB:
    def __init__(self, client, name, openai_ef):
        self.client = client
        self.name = name
        self.embedding_function = openai_ef
        self.collection = self.get_or_create()
    
    # Collection methods
    def get_or_create(self):
        try:
            return self.client.get_or_create_collection(
                name=self.name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error getting or creating collection: {e}")
            return None
    
    def delete(self):
        try:
            self.client.delete_collection(name=self.name)
            print(f"Collection '{self.name}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def collection_exists(self):
        try:
            collections = self.client.list_collections()
            return any(collection.name == self.name for collection in collections)
        except Exception as e:
            print(f"Error checking if collection exists: {e}")
            return False
       
    # Elements methods
    def document_exists(self, ids):
        try:
            result = self.collection.get(ids=ids)
            return [id in result['ids'] for id in ids]
        except Exception as e:
            print(f"Error checking document existence: {e}")
            return [False] * len(ids)

    
    def add(self, original_df):
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
    def __init__(self, db, original_df, client, embed):
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
#         self.metadata_prompt_template = """
#         Please extract information from the given sentence and format it according to the ChromaDB filter standard. Follow these steps:

#         1. Entity Extraction:
#             - publishdate: Identify time-related keywords. If it is missing, return {{"publishdate": None}}.

#         2. publishdate interpretation:
#            - "recent" or "latest": Use date range from 30 days ago to today
#            - "this quarter": Use range from start of current quarter to today
#            - "in [month]": Use range from start to end of specified month
#            - "in [month], [year]": Use range from start to end of specified month and year
#            - Specific dates: Use as given
#            - All dates should be should be in the standard yyyy-mm-dd format.

#         3. Formatting:
#            - If no publishdate is identified, return {{"publishdate": None}}.
#            - Use the ChromaDB filter standard
#            - For date ranges, use separate "$gte" and "$lte" conditions

#         Examples:

#         1. "Find office properties in New York City listed this quarter"
#         {{
#             "$and": [
#                 {{
#                     "publishdate": {{"$gte": "2024-07-01"}}
#                 }},
#                 {{
#                     "publishdate": {{"$lte": "2024-07-31"}}
#                 }}
#             ]
#         }}

#         2. "Show recent retail listings in Los Angeles County, CA"
#         {{
#             "$and": [
#                 {{
#                     "publishdate": {{"$gte": "2024-07-01"}}
#                 }},
#                 {{
#                     "publishdate": {{"$lte": "2024-07-31"}}
#                 }}
#             ]
#         }}

#         3. "What are the 5 transactions in New York in Dec, 2023?"
#         {{
#             "$and": [
#                 {{
#                     "publishdate": {{"$gte": "2023-12-01"}}
#                 }},
#                 {{
#                     "publishdate": {{"$lte": "2023-12-31"}}
#                 }}
#             ]
#         }}

#         Please provide the ChromaDB filter for the following sentence:
#         Input sentence: {sentence}
#         """
    
    def retrieve_relevant(self, search_text, K, where=None): # , where=None
        response = self.client.get_completion(message= self.metadata_prompt_template.format(sentence=search_text))
        where_filter = json.loads(response.choices[0].message.content)
        processed_filter = process_llm_output(where_filter)
        
        search_vectors = self.embed.embed_documents([search_text])
        results = self.db.retrieve(search_vectors, K, where=processed_filter) # , where=processed_filter
        
        idxs = [int(idx) for idx in results['ids'][0]]
        return pd.concat([pd.DataFrame({"distances": results['distances'][0]}, index=idxs), self.original_df.loc[idxs]], axis=1)

    
    @staticmethod
    def combine_context(relevant_contents): # Helper function for generating context
        context = relevant_contents.Content
        context = '\n'.join(context)

        return context
    
    def generate_context_prompt(self, search_text, K, where=None): # , where=None
        relevant_contents = self.retrieve_relevant(search_text=search_text, K=K)
        context = self.combine_context(relevant_contents=relevant_contents)
        prompt = self.prompt_template.format(context=context, search_text=search_text)
        return relevant_contents, prompt 
    
    
    
def df_db_pre(index_tb, tpwire_df):
    cols = ['dealname', 'propertysubtype', 'proptypecode', 'state', 'county', 'city', 'msaname', 'loanname', 'region', 'publishdate']
    df_data = tpwire_df[cols]
    
    return pd.concat([index_tb, df_data], axis=1)





