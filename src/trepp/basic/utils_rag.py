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
    
    
    
    
class S3ParquetLoader:
    def __init__(self, bucket_name, prefix, file_identifier="s3://"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.file_identifier = file_identifier
        
    def load_s3_parquet(self, file):
        s3_path = self.file_identifier + self.bucket_name + self.prefix + file
        return pq.read_table(s3_path).to_pandas()
    
    
    
    
class OpenAIClient:
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
    
    
class tpwireRAG:
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
        # for idx in range(len(df)):
        chunks = df['summaryplaintext'].tolist()
        
        self.index_tb = pd.DataFrame({
        'Content': chunks,
        'Vectors': embed.embed_documents(chunks), # 'original_index': range(len(chunks)),
        })

        self.index_tb.reset_index(drop=True, inplace=True)

        return self.index_tb
    
    def build_vec_db(self, index_tb):
        if not self.index_tb:
            self.index_tb = index_tb
        
        vectors = np.array(self.index_tb['Vectors'].tolist(), dtype='float32')
        vec_dim = vectors.shape[1]
        self.vec_db = faiss.IndexFlatL2(vec_dim)
        faiss.normalize_L2(vectors)
        self.vec_db.add(vectors)

        # return self.vec_db
        
        
    def retrieve_from_vec_db(self, search_text, K):
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
        context = relevant_contents.Content
        context = '\n'.join(context)

        return context
    
    def generate_context_prompt(self, search_text, K):
        relevant_contents = self.retrieve_from_vec_db(search_text=search_text, K=K)
        context = self.combine_context(relevant_contents=relevant_contents)
        prompt = self.prompt_template.format(context=context, search_text=search_text)
        return relevant_contents, prompt 
    
    
