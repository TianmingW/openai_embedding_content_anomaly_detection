import os
import csv
import glob


def getContent(pcap_file):
		tmp = pcap_file.split("/")
		filename = tmp[2]
		tmp = filename.split(".")
		filename = tmp[0]
		csv_name = filename + ".csv"
		command_tmp = "tshark -r " + pcap_file + " -Y \"tcp \
					and frame.number<= 50000\" -T fields -e frame.number -e \
					frame.time -e ip.src -e ip.dst -e tcp.seq -e \
					tcp.payload -E header=y -E separator=, -E quote=d\
					-E occurrence=f > " + save_path + csv_name
		print(command_tmp)
		os.system(command_tmp)

if __name__ == '__main__':
	folder_path = './iot2023/'
	save_path = "./iot_2023_csv/"

	for pcap_file in glob.glob(os.path.join(folder_path, '*.pcap')):
		getContent(pcap_file)
		print("finished" + pcap_file)