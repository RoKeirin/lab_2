import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, value=None, freq=0, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def flatten(data):
    if isinstance(data, list) and any(isinstance(i, list) for i in data):
        return [item for sublist in data for item in sublist]
    return data

def build_huffman_table(data_list):
    data_list = flatten(data_list)
    freq = defaultdict(int)
    for item in data_list:
        freq[item] += 1

    heap = [HuffmanNode(val, frq) for val, frq in freq.items()]
    if not heap:
        return {}

    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        new = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, new)

    root = heap[0]
    codes = {}

    def traverse(node, code=""):
        if node.value is not None:
            codes[node.value] = code
        else:
            traverse(node.left, code + "0")
            traverse(node.right, code + "1")

    traverse(root)
    return codes

def encode_with_huffman(data_list, huffman_table):
    data_list = flatten(data_list)
    return ''.join(huffman_table[item] for item in data_list)

def decode_huffman(encoded_str, huffman_table):
    reverse_table = {v: k for k, v in huffman_table.items()}
    result = []
    code = ""
    for bit in encoded_str:
        code += bit
        if code in reverse_table:
            result.append(reverse_table[code])
            code = ""
    return result
