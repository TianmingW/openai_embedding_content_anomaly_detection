import sys
sys.path.append('./utils')
import os
import glob
import tiktoken

import pandas as pd
import re

from openai import OpenAI

def get_labeled_df(label_file, packet_file):
    column_names = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 
        'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 
        'local_resp', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes', 
        'resp_pkts', 'resp_ip_bytes', 'tunnel_parents_label_detailed-label'
    ]

    print(label_file, packet_file)

    # Load data and preprocess
    df = pd.read_csv(
        label_file, sep='\t', header=None, names=column_names,
        comment='#', na_values=['-', '(empty)'], skipinitialspace=True, engine='python'
    )
    df[['tunnel_parents', 'label', 'detailed-label']] = df['tunnel_parents_label_detailed-label'].str.split(expand=True, n=2)
    df.drop(columns='tunnel_parents_label_detailed-label', inplace=True)
    df['ts'] = pd.to_datetime(df['ts'], unit='s')  # Convert timestamp from UNIX time to readable datetime

    # Filter and manipulate data
    selected_flows = df[['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'label']].drop_duplicates()
    tcp_flow = selected_flows[selected_flows['proto'] == 'tcp'].drop(columns='proto')
    tcp_flow.rename(columns={'id.orig_h':'ip.src', 'id.orig_p':'tcp.srcport', 
                             'id.resp_h':'ip.dst', 'id.resp_p': 'tcp.dstport'}, inplace=True)

    # Create a dataframe of swapped source and destination
    swapped_df = tcp_flow.rename(columns={'ip.src': 'ip.dst', 'ip.dst': 'ip.src', 
                                          'tcp.srcport': 'tcp.dstport', 'tcp.dstport': 'tcp.srcport'}, inplace=False)

    # Combine flows from both directions
    tcp_bothway = pd.concat([tcp_flow, swapped_df], ignore_index=True)

    df_packets = pd.read_csv(packet_file)
    merged_df = df_packets.merge(tcp_bothway, how='left', on=['ip.src', 'tcp.srcport', 'ip.dst', 'tcp.dstport'])
    return merged_df


def tokenizer_payload(hex_string, token_length = 4):
    # token_length = 4 
    # new token length according to the max tokens for the embedding; which is original 6   
    regex_pattern = '.{1,' + str(token_length) + '}'
    return ' '.join(re.findall(regex_pattern, hex_string))

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embeddings_from_payload(df, embedding_model = "text-embedding-ada-002"):
    df.dropna(subset = ['tcp.payload'], inplace=True)
    df['tokenizer_content'] = df['tcp.payload'].apply(tokenizer_payload)

    max_tokens = 8000
    
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.tokenizer_content.apply(lambda x: len(encoding.encode(x)))
    top_n = 10
    df = df[df.n_tokens <= max_tokens].head(top_n)
    # df = df[df.n_tokens <= max_tokens]

    df_embedding = pd.DataFrame()
    df_embedding["X"] = df.tokenizer_content.apply(lambda x: get_embedding(x, model=embedding_model))
    df_embedding["y"] = df['label'].apply(lambda x: 0 if x == "Benign" else 1)
    return df_embedding

if __name__ == "__main__":

    # Define the base directory for the IoT datasets
    base_directory = "./iot23/"
    label_dir = "bro"
    label_file = "conn.log.labeled"

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    embedding_encoding = "cl100k_base"
    
    # Set the folder path
    
    capture_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    capture_dirs = [
        # 'CTU-IoT-Malware-Capture-34-1'
        'CTU-IoT-Malware-Capture-35-1'
        # 'CTU-IoT-Malware-Capture-43-1'
        ]

    for capture_dir in capture_dirs:
        print(f"Capture directory {capture_dir}\n")

        pcap_path = None
        label_path = os.path.join(base_directory, capture_dir, "bro", "conn.log.labeled")
    
        # Look for the pcap file in the directory
        for file in os.listdir(os.path.join(base_directory, capture_dir)):
            if file.endswith(".csv"):
                pcap_path = os.path.join(base_directory, capture_dir, file)
                break
        if pcap_path and os.path.exists(label_path):
            print(f"Pcap file name {pcap_path}\n")
            print(f"Label file {label_path}, and start get label file\n")

            df = get_labeled_df(label_path, pcap_path)
            transformed_df = get_embeddings_from_payload(df)
            print(f"The label file path is {label_path}\n", f"The pcap file path is {pcap_path}")
            # Save the transformed data
            save_path = os.path.join(base_directory, capture_dir, "embeddings.h5")
            transformed_df.to_hdf(save_path, key='df', mode='a', complevel=5)
            print(f"Embeddings saved to {save_path}\n")
        else:
            print(f"Required files not found in {capture_dir}")
