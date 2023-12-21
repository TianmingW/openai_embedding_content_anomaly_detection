import pandas as pd
import tiktoken
from embeddings_utils import get_embedding
import re
# from tenacity import retry, stop_after_attempt, wait_random_exponential

def tokenizer_payload(hex_string, token_length = 4):
    # token_length = 4 
    # new token length according to the max tokens for the embedding; which is original 6   
    regex_pattern = '.{1,' + str(token_length) + '}'
    return ' '.join(re.findall(regex_pattern, hex_string))


# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings_from_payload(csv_file, benign = True, embedding_model = "text-embedding-ada-002", embedding_encoding = "cl100k_base" ):
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    top_n = 10 # take first 10 packets for test

    df = pd.read_csv(csv_file)
    df.dropna(subset = ['tcp.payload'])

    df['tokenizer_content'] = df['tcp.payload'].apply(tokenizer_payload)
    encoding = tiktoken.get_encoding(embedding_encoding)

    df["n_tokens"] = df.tokenizer_content.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)

    df_embedding = pd.DataFrame()
    df_embedding["X"] = lambda x: get_embedding(x, model=embedding_model)
    if(benign):
        df_embedding["y"] = 0
    else:
        df_embedding["y"] = 1
    df_embedding.to_csv("/embeddings/"+csv_file, index=False)
    return 0