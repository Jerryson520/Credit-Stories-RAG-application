import os, json
from langchain_openai import OpenAIEmbeddings
from utils_rag import SecretManager, OpenAIClient, S3ParquetLoader, TpwireDataLoader, tpwireRAG
from config import SECRET_NAME, BUCKET_NAME, PREFIX, MODEL, EMBED_MODEL_NAME, K




def main(input_question):
    secret_manager = SecretManager(secret_name=SECRET_NAME)
    os.environ['OPENAI_API_KEY'] = secret_manager.get_secret('OPENAI_API_KEY')
    
    embed = OpenAIEmbeddings(model=EMBED_MODEL_NAME) # Create embeddings for indexing documents
    
    openai_client = OpenAIClient(model=MODEL)

    # S3 parquet loader
    s3_parquet_loader = S3ParquetLoader(bucket_name=BUCKET_NAME, prefix=PREFIX)

    # Import TreppWire data
    tpwire_loader = TpwireDataLoader(bucket_name=BUCKET_NAME, prefix=PREFIX)

    # Initialize treppwire RAG
    tpwire_RAG = tpwireRAG(embed=embed)
    
    
    tpwire_df = tpwire_loader.process_data("tpwire_flags.parquet")
    # index_tb = index_document(tpwire_df, embed)
    # index_tb = tpwire_RAG.index_document(tpwire_df)
    # index_tb.to_parquet('treppwire_index_tb.parquet')
    # index_tb = pd.read_parquet('treppwire_index_tb.parquet')
    index_tb = s3_parquet_loader.load_s3_parquet('treppwire_index_tb.parquet')
    
    tpwire_RAG.build_vec_db(index_tb=index_tb)
    
    search_text = 'What is the recent price of Flagler Corporate Center loan?'
    relevant_contents, prompt = tpwire_RAG.generate_context_prompt(search_text=search_text, K=K)
    response = openai_client.get_completion(message=prompt)
    return json.loads(response.choices[0].message.content)['Answer']
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ask a question to the TreppWire RAG system')
    parser.add_argument('--question', metavar='question', required=True,
                        help='The question you want to ask')
    args = parser.parse_args()
    print(f"The answer is: {main(input_question)}")