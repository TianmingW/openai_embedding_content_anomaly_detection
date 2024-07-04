import os
import csv
import glob
import argparse

def getContent(src, dst):
		tmp = src.split("/")
		filename = tmp[-1]
		tmp = filename.split(".")
		filename = tmp[0]
		csv_name = filename + "_c.csv"
		command_tmp = "tshark -r " + src + " -Y \"tcp and tcp.payload\" \
					-T fields -e frame.number -e \
					frame.time -e ip.src -e tcp.srcport -e ip.dst -e tcp.dstport -e tcp.seq -e \
					tcp.payload -E header=y -E separator=, -E quote=d\
					-E occurrence=f > " + dst + csv_name
		print(command_tmp)
		os.system(command_tmp)

def main(args):
	if args.file:
		getContent(args.file, os.path.dirname(args.file)+'/')
		print("Finished processing:", args.file)
	elif args.src and args.dst:
		# If a source directory and destination directory are specified, process all files
		for pcap_file in glob.glob(os.path.join(args.src, '*.pcap')):
			getContent(pcap_file)
			print("Finished processing:", pcap_file)
	else:
		# Error handling if neither appropriate condition is met
		raise ValueError("Invalid arguments: specify either --file or both --src and --dst.")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Input pre-process file or folder name.")
	parser.add_argument("--src", type=str, help="Source folder containing PCAP files")
	parser.add_argument("--dst", type=str, help="Destination folder for processed files")
	parser.add_argument("--file", type=str, help="Single file path for processing")

	args = parser.parse_args()

	folder_path = args.src
	save_path = args.dst

	try:
		main(args)
	except Exception as e:
		print(f"Error: {e}")