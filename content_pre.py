# content_preprocess
# 12 files
import os
import csv


def getContent(pcap_file):
		filename, _ = pcap_file.split(".")
		csv_name = filename + ".csv"
		command_tmp = "tshark -r " + filename + ".pcap -Y \"tcp \
					and frame.number<= 50000\" -T fields -e frame.number -e \
					frame.time -e ip.src -e ip.dst -e tcp.seq -e \
					tcp.payload -E header=y -E separator=, -E quote=d\
					-E occurrence=f > " + save_path + csv_name
		os.system(command_tmp)

folder_path = './iot2023/'
save_path = "./iot_2023_embeddings/"

for pcap_file in glob.glob(os.path.join(folder_path, '*.csv')):
	getContent(pcap_file)