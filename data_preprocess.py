import sys
sys.path.append('./utils')
import os
import glob
import tiktoken

import pandas as pd
import re

from openai import OpenAI

def tokenizer_payload(hex_string, token_length = 4):
    # token_length = 4 
    # new token length according to the max tokens for the embedding; which is original 6   
    regex_pattern = '.{1,' + str(token_length) + '}'
    return ' '.join(re.findall(regex_pattern, hex_string))



def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embeddings_from_payload(csv_file, benign = True, embedding_model = "text-embedding-ada-002", save_prefix=""):
    filename = os.path.basename(csv_file)

    df = pd.read_csv(csv_file)
    df.dropna(subset = ['tcp.payload'], inplace=True)

    df['tokenizer_content'] = df['tcp.payload'].apply(tokenizer_payload)
    # print("stop")
    # print(df.iloc[0]['tokenizer_content'])
    # print(df.iloc[0]['tcp.payload'])

    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.tokenizer_content.apply(lambda x: len(encoding.encode(x)))
    # df = df[df.n_tokens <= max_tokens].tail(top_n)
    df = df[df.n_tokens <= max_tokens]

    df_embedding = pd.DataFrame()
    df_embedding["X"] = df.tokenizer_content.apply(lambda x: get_embedding(x, model=embedding_model))
    if(benign):
        df_embedding["y"] = 0
    else:
        df_embedding["y"] = 1
    df_embedding.to_csv(save_prefix+filename, index=False)
    return 0
if __name__ == "__main__":
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    embedding_encoding = "cl100k_base"
    max_tokens = 8000
    # top_n = 100
    # Set the folder path
    folder_path = './DoS_tcp_flood_csv/'  # Replace with your folder path
    save_path = "./DoS_tcp_flood_embedding/"

    # Iterate over all CSV files in the folders
    for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
        # print("in loop.")
        filename = os.path.basename(csv_file)
        print(f"Processing file: {csv_file}")
        if 'Mirai' in filename or 'DoS' in filename:    
            get_embeddings_from_payload(csv_file, benign=False, save_prefix=save_path)
        else:
            get_embeddings_from_payload(csv_file, benign=True, save_prefix=save_path)
