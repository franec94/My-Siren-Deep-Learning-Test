from utils.huffman import HuffmanCoding

#input file path
path = "in.txt"

h = HuffmanCoding(path)

output_path = h.compress()
h.decompress(output_path)