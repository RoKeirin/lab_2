import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from huff import build_huffman_table, encode_with_huffman, decode_huffman
def flatten_rle(rle_list):
    return [pair for block in rle_list for pair in block]
def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    Cb = -0.1687 * img[..., 0] - 0.3313 * img[..., 1] + 0.5 * img[..., 2] + 128
    Cr = 0.5 * img[..., 0] - 0.4187 * img[..., 1] - 0.0813 * img[..., 2] + 128
    return np.stack((Y, Cb, Cr), axis=-1)
def ycbcr_to_rgb(img_ycbcr):
    Y, Cb, Cr = img_ycbcr[..., 0], img_ycbcr[..., 1], img_ycbcr[..., 2]
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    rgb = np.stack((R, G, B), axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def downsample(channel):
    H, W = channel.shape
    # Обрезаем до чётного размера, если нужно
    H = H - (H % 2)
    W = W - (W % 2)
    channel = channel[:H, :W]

    # Усредняем 2×2 блоки
    return (channel[0::2, 0::2] +
            channel[1::2, 0::2] +
            channel[0::2, 1::2] +
            channel[1::2, 1::2]) /4
def upsample(channel, shape): return np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)[:shape[0], :shape[1]]

def split_into_blocks(channel, block_size=8):
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    blocks = []
    for i in range(0, channel.shape[0], block_size):
        for j in range(0, channel.shape[1], block_size):
            blocks.append(channel[i:i+block_size, j:j+block_size])
    return np.array(blocks), channel.shape

def merge_blocks(blocks, shape, block_size=8):
    h, w = shape
    img = np.zeros((h, w))
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            img[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return img
def create_dct_matrix(N):
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            D[k, n] = alpha * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return D
def dct2(block, D): return D @ block @ D.T
def idct2(block, D): return D.T @ block @ D
def get_quant_matrix(N, quality, chrominance=False):
    Q_chrom_base = np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]
    ])
    Q_base_luminance = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])
    Q_base = Q_chrom_base if chrominance else Q_base_luminance
    if quality <= 0: quality = 1
    if N != 8: return np.ones((N, N)) * (quality if quality > 0 else 1)
    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    return np.clip((Q_base * scale + 50) // 100, 1, 255)
def quantize(block, Q): return np.round(block / Q)
def dequantize(block, Q): return block * Q
def zigzag(block):
    n = block.shape[0]
    result = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(s+1):
                j = s - i
                if i < n and j < n:
                    result.append(block[i, j])
        else:
            for j in range(s+1):
                i = s - j
                if i < n and j < n:
                    result.append(block[i, j])
    return np.array(result)
def inverse_zigzag(vector, n):
    block = np.zeros((n, n))
    idx = 0
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(s+1):
                j = s - i
                if i < n and j < n:
                    block[i, j] = vector[idx]
                    idx += 1
        else:
            for j in range(s+1):
                i = s - j
                if i < n and j < n:
                    block[i, j] = vector[idx]
                    idx += 1
    return block
def differential_encode_dc(dc_coeffs):
    encoded = []
    prev = 0
    for dc in dc_coeffs:
        diff = dc - prev
        encoded.append(diff)
        prev = dc
    return encoded
def differential_decode_dc(diffs):
    decoded = []
    current = 0
    for diff in diffs:
        current += diff
        decoded.append(current)
    return decoded
def category(value):

    abs_val = abs(int(value))
    if abs_val == 0:
        return 0
    return abs_val.bit_length()
def variable_length_encode(values):
    return [(category(v), v) for v in values]
def variable_length_decode(encoded):
    """Обратное переменное декодирование"""
    return [val for cat, val in encoded]
def rle_encode(ac_coeffs):
    """Кодирует список AC коэффициентов с помощью RLE"""
    result = []
    zero_count = 0
    for val in ac_coeffs:
        if val == 0:
            zero_count += 1
        else:
            while zero_count > 15:
                result.append((15, 0))  # спецификация JPEG
                zero_count -= 16
            result.append((zero_count, val))
            zero_count = 0
    result.append((0, 0))  # EOB (End Of Block)
    return result
def rle_decode(rle_data):
    ac_coeffs = []
    for zeros, value in rle_data:
        if (zeros, value) == (0, 0):
            while len(ac_coeffs) < 63:
                ac_coeffs.append(0)
            break
        ac_coeffs.extend([0] * zeros)
        ac_coeffs.append(value)
    while len(ac_coeffs) < 63:
        ac_coeffs.append(0)
    return ac_coeffs

# def compress_image1(img, quality, block_size):
#     img_ycbcr = rgb_to_ycbcr(img)
#     Y, Cb, Cr = img_ycbcr[..., 0], img_ycbcr[..., 1], img_ycbcr[..., 2]
#     if Cb.size == 0 or Cr.size == 0:
#         raise ValueError("Cb или Cr канал пуст после downsampling. Возможно, изображение слишком маленькое.")
#
#     Cb_down = downsample(Cb)
#     Cr_down = downsample(Cr)
#
#     blocks_Y, shape_Y = split_into_blocks(Y, block_size)
#     blocks_Cb, shape_Cb = split_into_blocks(Cb_down, block_size)
#     blocks_Cr, shape_Cr = split_into_blocks(Cr_down, block_size)
#
#     D = create_dct_matrix(block_size)
#     dct_blocks_Y = np.array([dct2(block, D) for block in blocks_Y])
#     dct_blocks_Cb = np.array([dct2(block, D) for block in blocks_Cb])
#     dct_blocks_Cr = np.array([dct2(block, D) for block in blocks_Cr])
#
#     Q_Y = get_quant_matrix(block_size, quality, chrominance=False)
#     Q_C = get_quant_matrix(block_size, quality, chrominance=True)
#
#     quant_blocks_Y = np.array([quantize(block, Q_Y) for block in dct_blocks_Y])
#     quant_blocks_Cb = np.array([quantize(block, Q_C) for block in dct_blocks_Cb])
#     quant_blocks_Cr = np.array([quantize(block, Q_C) for block in dct_blocks_Cr])
#
#     zigzag_Y = np.array([zigzag(block) for block in quant_blocks_Y])
#     zigzag_Cb = np.array([zigzag(block) for block in quant_blocks_Cb])
#     zigzag_Cr = np.array([zigzag(block) for block in quant_blocks_Cr])
#
#     dc_Y = zigzag_Y[:, 0]
#     dc_Cb = zigzag_Cb[:, 0]
#     dc_Cr = zigzag_Cr[:, 0]
#
#     dc_diffs_Y = differential_encode_dc(dc_Y)
#     dc_diffs_Cb = differential_encode_dc(dc_Cb)
#     dc_diffs_Cr = differential_encode_dc(dc_Cr)
#
#     ac_Y = zigzag_Y[:, 1:]
#     ac_Cb = zigzag_Cb[:, 1:]
#     ac_Cr = zigzag_Cr[:, 1:]
#
#     rle_ac_Y = [rle_encode(ac) for ac in ac_Y]
#     rle_ac_Cb = [rle_encode(ac) for ac in ac_Cb]
#     rle_ac_Cr = [rle_encode(ac) for ac in ac_Cr]
#
#     flat_rle_ac_Y = flatten_rle(rle_ac_Y)
#     flat_rle_ac_Cb = flatten_rle(rle_ac_Cb)
#     flat_rle_ac_Cr = flatten_rle(rle_ac_Cr)
#     # --- Построение и применение таблиц Хаффмана ---
#     huff_dc_Y = build_huffman_table(dc_diffs_Y)
#     huff_dc_Cb = build_huffman_table(dc_diffs_Cb)
#     huff_dc_Cr = build_huffman_table(dc_diffs_Cr)
#
#     huff_ac_Y = build_huffman_table(flat_rle_ac_Y)
#     huff_ac_Cb = build_huffman_table(flat_rle_ac_Cb)
#     huff_ac_Cr = build_huffman_table(flat_rle_ac_Cr)
#
#     encoded_dc_Y = encode_with_huffman(dc_diffs_Y, huff_dc_Y)
#     encoded_dc_Cb = encode_with_huffman(dc_diffs_Cb, huff_dc_Cb)
#     encoded_dc_Cr = encode_with_huffman(dc_diffs_Cr, huff_dc_Cr)
#
#     encoded_ac_Y = encode_with_huffman(flat_rle_ac_Y, huff_ac_Y)
#     encoded_ac_Cb = encode_with_huffman(flat_rle_ac_Cb, huff_ac_Cb)
#     encoded_ac_Cr = encode_with_huffman(flat_rle_ac_Cr, huff_ac_Cr)
#
#     return {
#         "shape_Y": shape_Y,
#         "shape_Cb": shape_Cb,
#         "shape_Cr": shape_Cr,
#         "Q_Y": Q_Y,
#         "Q_C": Q_C,
#         "block_size": block_size,
#
#         "encoded_dc_Y": encoded_dc_Y,
#         "encoded_dc_Cb": encoded_dc_Cb,
#         "encoded_dc_Cr": encoded_dc_Cr,
#
#         "encoded_ac_Y": encoded_ac_Y,
#         "encoded_ac_Cb": encoded_ac_Cb,
#         "encoded_ac_Cr": encoded_ac_Cr,
#
#         "dc_codes_Y": huff_dc_Y,
#         "dc_codes_Cb": huff_dc_Cb,
#         "dc_codes_Cr": huff_dc_Cr,
#
#         "ac_codes_Y": huff_ac_Y,
#         "ac_codes_Cb": huff_ac_Cb,
#         "ac_codes_Cr": huff_ac_Cr
#     }
def compress_image(img, quality, block_size):
    img_ycbcr = rgb_to_ycbcr(img)
    Y, Cb, Cr = img_ycbcr[..., 0], img_ycbcr[..., 1], img_ycbcr[..., 2]

    # Определение grayscale по почти неизменным Cb/Cr
    if np.allclose(Cb, Cb[0, 0]) and np.allclose(Cr, Cr[0, 0]):
        print("Grayscale image detected.")
        compress_chroma = False
    else:
        compress_chroma = True

    blocks_Y, shape_Y = split_into_blocks(Y, block_size)
    D = create_dct_matrix(block_size)
    dct_blocks_Y = np.array([dct2(block, D) for block in blocks_Y])
    Q_Y = get_quant_matrix(block_size, quality, chrominance=False)
    quant_blocks_Y = np.array([quantize(block, Q_Y) for block in dct_blocks_Y])
    zigzag_Y = np.array([zigzag(block) for block in quant_blocks_Y])
    dc_Y = zigzag_Y[:, 0]
    ac_Y = zigzag_Y[:, 1:]
    dc_diffs_Y = differential_encode_dc(dc_Y)
    rle_ac_Y = [rle_encode(ac) for ac in ac_Y]
    flat_rle_ac_Y = flatten_rle(rle_ac_Y)

    huff_dc_Y = build_huffman_table(dc_diffs_Y)
    huff_ac_Y = build_huffman_table(flat_rle_ac_Y)
    encoded_dc_Y = encode_with_huffman(dc_diffs_Y, huff_dc_Y)
    encoded_ac_Y = encode_with_huffman(flat_rle_ac_Y, huff_ac_Y)

    if compress_chroma:
        Cb_down = downsample(Cb)
        Cr_down = downsample(Cr)
        blocks_Cb, shape_Cb = split_into_blocks(Cb_down, block_size)
        blocks_Cr, shape_Cr = split_into_blocks(Cr_down, block_size)
        dct_blocks_Cb = np.array([dct2(block, D) for block in blocks_Cb])
        dct_blocks_Cr = np.array([dct2(block, D) for block in blocks_Cr])
        Q_C = get_quant_matrix(block_size, quality, chrominance=True)
        quant_blocks_Cb = np.array([quantize(block, Q_C) for block in dct_blocks_Cb])
        quant_blocks_Cr = np.array([quantize(block, Q_C) for block in dct_blocks_Cr])
        zigzag_Cb = np.array([zigzag(block) for block in quant_blocks_Cb])
        zigzag_Cr = np.array([zigzag(block) for block in quant_blocks_Cr])
        dc_Cb = zigzag_Cb[:, 0]
        dc_Cr = zigzag_Cr[:, 0]
        ac_Cb = zigzag_Cb[:, 1:]
        ac_Cr = zigzag_Cr[:, 1:]
        dc_diffs_Cb = differential_encode_dc(dc_Cb)
        dc_diffs_Cr = differential_encode_dc(dc_Cr)
        rle_ac_Cb = [rle_encode(ac) for ac in ac_Cb]
        rle_ac_Cr = [rle_encode(ac) for ac in ac_Cr]
        flat_rle_ac_Cb = flatten_rle(rle_ac_Cb)
        flat_rle_ac_Cr = flatten_rle(rle_ac_Cr)
        huff_dc_Cb = build_huffman_table(dc_diffs_Cb)
        huff_dc_Cr = build_huffman_table(dc_diffs_Cr)
        huff_ac_Cb = build_huffman_table(flat_rle_ac_Cb)
        huff_ac_Cr = build_huffman_table(flat_rle_ac_Cr)
        encoded_dc_Cb = encode_with_huffman(dc_diffs_Cb, huff_dc_Cb)
        encoded_dc_Cr = encode_with_huffman(dc_diffs_Cr, huff_dc_Cr)
        encoded_ac_Cb = encode_with_huffman(flat_rle_ac_Cb, huff_ac_Cb)
        encoded_ac_Cr = encode_with_huffman(flat_rle_ac_Cr, huff_ac_Cr)
    else:
        shape_Cb = shape_Cr = (0, 0)
        Q_C = np.ones((block_size, block_size))
        encoded_dc_Cb = encoded_dc_Cr = []
        encoded_ac_Cb = encoded_ac_Cr = []
        huff_dc_Cb = huff_dc_Cr = {}
        huff_ac_Cb = huff_ac_Cr = {}

    return {
        "shape_Y": shape_Y,
        "shape_Cb": shape_Cb,
        "shape_Cr": shape_Cr,
        "Q_Y": Q_Y,
        "Q_C": Q_C,
        "block_size": block_size,
        "encoded_dc_Y": encoded_dc_Y,
        "encoded_dc_Cb": encoded_dc_Cb,
        "encoded_dc_Cr": encoded_dc_Cr,
        "encoded_ac_Y": encoded_ac_Y,
        "encoded_ac_Cb": encoded_ac_Cb,
        "encoded_ac_Cr": encoded_ac_Cr,
        "dc_codes_Y": huff_dc_Y,
        "dc_codes_Cb": huff_dc_Cb,
        "dc_codes_Cr": huff_dc_Cr,
        "ac_codes_Y": huff_ac_Y,
        "ac_codes_Cb": huff_ac_Cb,
        "ac_codes_Cr": huff_ac_Cr
    }

# def decompress_image1(compressed):
#     shape_Y = compressed["shape_Y"]
#     shape_Cb = compressed["shape_Cb"]
#     shape_Cr = compressed["shape_Cr"]
#     Q_Y = compressed["Q_Y"]
#     Q_C = compressed["Q_C"]
#     block_size = compressed["block_size"]
#     D = create_dct_matrix(block_size)
#
#     decode = lambda bits, table: decode_huffman(bits, table)
#
#     dc_diffs_Y = decode(compressed["encoded_dc_Y"], compressed["dc_codes_Y"])
#     dc_diffs_Cb = decode(compressed["encoded_dc_Cb"], compressed["dc_codes_Cb"])
#     dc_diffs_Cr = decode(compressed["encoded_dc_Cr"], compressed["dc_codes_Cr"])
#
#     dc_Y = differential_decode_dc(dc_diffs_Y)
#     dc_Cb = differential_decode_dc(dc_diffs_Cb)
#     dc_Cr = differential_decode_dc(dc_diffs_Cr)
#
#     def restore_ac_blocks(flat_data):
#         blocks = []
#         block = []
#         for pair in flat_data:
#             block.append(pair)
#             if pair == (0, 0):  # EOB
#                 blocks.append(rle_decode(block))
#                 block = []
#         return blocks
#
#     decoded_ac_Y = decode(compressed["encoded_ac_Y"], compressed["ac_codes_Y"])
#     decoded_ac_Cb = decode(compressed["encoded_ac_Cb"], compressed["ac_codes_Cb"])
#     decoded_ac_Cr = decode(compressed["encoded_ac_Cr"], compressed["ac_codes_Cr"])
#
#     ac_Y = restore_ac_blocks(decoded_ac_Y)
#     ac_Cb = restore_ac_blocks(decoded_ac_Cb)
#     ac_Cr = restore_ac_blocks(decoded_ac_Cr)
#
#     zz_Y = np.array([[dc] + ac for dc, ac in zip(dc_Y, ac_Y)])
#     zz_Cb = np.array([[dc] + ac for dc, ac in zip(dc_Cb, ac_Cb)])
#     zz_Cr = np.array([[dc] + ac for dc, ac in zip(dc_Cr, ac_Cr)])
#
#     blocks_Y = np.array([inverse_zigzag(v, block_size) for v in zz_Y])
#     blocks_Cb = np.array([inverse_zigzag(v, block_size) for v in zz_Cb])
#     blocks_Cr = np.array([inverse_zigzag(v, block_size) for v in zz_Cr])
#
#     deq_Y = np.array([dequantize(b, Q_Y) for b in blocks_Y])
#     deq_Cb = np.array([dequantize(b, Q_C) for b in blocks_Cb])
#     deq_Cr = np.array([dequantize(b, Q_C) for b in blocks_Cr])
#
#     idct_Y = np.array([idct2(b, D) for b in deq_Y])
#     idct_Cb = np.array([idct2(b, D) for b in deq_Cb])
#     idct_Cr = np.array([idct2(b, D) for b in deq_Cr])
#
#     Y = merge_blocks(idct_Y, shape_Y, block_size)
#     Cb = merge_blocks(idct_Cb, shape_Cb, block_size)
#     Cr = merge_blocks(idct_Cr, shape_Cr, block_size)
#
#     Cb_up = upsample(Cb, Y.shape)
#     Cr_up = upsample(Cr, Y.shape)
#
#     ycbcr = np.stack((Y, Cb_up, Cr_up), axis=-1)
#     return ycbcr_to_rgb(ycbcr)
def decompress_image(compressed):
    shape_Y = compressed["shape_Y"]
    shape_Cb = compressed["shape_Cb"]
    shape_Cr = compressed["shape_Cr"]
    Q_Y = compressed["Q_Y"]
    Q_C = compressed["Q_C"]
    block_size = compressed["block_size"]
    D = create_dct_matrix(block_size)

    compress_chroma = shape_Cb != (0, 0)

    decode = lambda bits, table: decode_huffman(bits, table)

    dc_diffs_Y = decode(compressed["encoded_dc_Y"], compressed["dc_codes_Y"])
    dc_Y = differential_decode_dc(dc_diffs_Y)
    decoded_ac_Y = decode(compressed["encoded_ac_Y"], compressed["ac_codes_Y"])

    def restore_ac_blocks(flat_data):
        blocks = []
        block = []
        for pair in flat_data:
            block.append(pair)
            if pair == (0, 0):
                blocks.append(rle_decode(block))
                block = []
        return blocks

    ac_Y = restore_ac_blocks(decoded_ac_Y)
    zz_Y = np.array([[dc] + ac for dc, ac in zip(dc_Y, ac_Y)])
    blocks_Y = np.array([inverse_zigzag(v, block_size) for v in zz_Y])
    deq_Y = np.array([dequantize(b, Q_Y) for b in blocks_Y])
    idct_Y = np.array([idct2(b, D) for b in deq_Y])
    Y = merge_blocks(idct_Y, shape_Y, block_size)

    if compress_chroma:
        dc_diffs_Cb = decode(compressed["encoded_dc_Cb"], compressed["dc_codes_Cb"])
        dc_diffs_Cr = decode(compressed["encoded_dc_Cr"], compressed["dc_codes_Cr"])
        dc_Cb = differential_decode_dc(dc_diffs_Cb)
        dc_Cr = differential_decode_dc(dc_diffs_Cr)
        decoded_ac_Cb = decode(compressed["encoded_ac_Cb"], compressed["ac_codes_Cb"])
        decoded_ac_Cr = decode(compressed["encoded_ac_Cr"], compressed["ac_codes_Cr"])
        ac_Cb = restore_ac_blocks(decoded_ac_Cb)
        ac_Cr = restore_ac_blocks(decoded_ac_Cr)
        zz_Cb = np.array([[dc] + ac for dc, ac in zip(dc_Cb, ac_Cb)])
        zz_Cr = np.array([[dc] + ac for dc, ac in zip(dc_Cr, ac_Cr)])
        blocks_Cb = np.array([inverse_zigzag(v, block_size) for v in zz_Cb])
        blocks_Cr = np.array([inverse_zigzag(v, block_size) for v in zz_Cr])
        deq_Cb = np.array([dequantize(b, Q_C) for b in blocks_Cb])
        deq_Cr = np.array([dequantize(b, Q_C) for b in blocks_Cr])
        idct_Cb = np.array([idct2(b, D) for b in deq_Cb])
        idct_Cr = np.array([idct2(b, D) for b in deq_Cr])
        Cb = merge_blocks(idct_Cb, shape_Cb, block_size)
        Cr = merge_blocks(idct_Cr, shape_Cr, block_size)
        Cb_up = upsample(Cb, Y.shape)
        Cr_up = upsample(Cr, Y.shape)
    else:
        Cb_up = np.full(Y.shape, 128)
        Cr_up = np.full(Y.shape, 128)

    ycbcr = np.stack((Y, Cb_up, Cr_up), axis=-1)
    return ycbcr_to_rgb(ycbcr)

import pickle
def save_compressed_data(data, filename):
    with open(filename, "wb") as fa:
        pickle.dump(data, fa)

def load_compressed_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_image(img, filename):
    Image.fromarray(img.astype(np.uint8)).save(filename)


def show_images(original_img, decompressed_img):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(o_img.astype('uint8'))
    plt.title('Оригинальное изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(decompressed_img.astype('uint8'))
    plt.title(f'После декомпрессии (коэффициент = {qq})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


file_name_number = 1
while file_name_number>0:
    file_name_number = int(input(f'\nВЫХОД - 0;\nИзображение для сжатия:\nLenna original - 1\nLenna grayscale - 2\n'
                                 f'Lenna bw with dithering - 3\nLenna bw without dithering - 4\nlion original - 5\nlion grayscale - 6\nlion bw with dithering - 7\nlion bw without dithering - 8\nВВОД: '))
    if file_name_number < 0 or file_name_number > 8:
        print('\nВведите другой номер')
        continue
    else:
        if file_name_number == 0: break
        if file_name_number == 1: path_name = 'Lenna'
        if file_name_number == 2: path_name = 'Lenna_gray'
        if file_name_number == 3: path_name = 'Lenna_bw'
        if file_name_number == 4: path_name = 'Lenna_bw_no_dither'
        if file_name_number == 5: path_name = 'lion'
        if file_name_number == 6: path_name = 'lion_gray'
        if file_name_number == 7: path_name = 'lion_bw'
        if file_name_number == 8: path_name = 'lion_bw_no_dither'
        img = Image.open(f'{path_name}.png')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        o_img = np.array(img)
    bb = 8
    qq = 0
    while qq != 111 :
        qq = int(input(f'\n   Выход - 111\n   Уровень качества: '))
        if qq == 111: break
        if qq < 0 or qq > 100:
            print('   Ошибка. Введите другое значение')
            continue
        else:
            compressed = compress_image(o_img, qq, bb)
            save_compressed_data(compressed, 'image.bin')
            compressed_loaded = load_compressed_data('image.bin')
            restored_img = decompress_image(compressed_loaded)

            save_image(restored_img, f'C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\{path_name}_restored_q={qq}.png')
            save_image(restored_img, f'{path_name}_restored_q={qq}.png')

            show_images(o_img, restored_img)
print('Работа завершена')

import os
import matplotlib.pyplot as plt

def plot_compression_vs_quality(image_path, AAA, block_size=8, qualities=range(0, 101, 5)):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_np = np.array(img)

    file_sizes = []

    for q in qualities:

        compressed = compress_image(img_np, q, block_size)
        save_compressed_data(compressed, 'image.bin')
        file_size = os.path.getsize('image.bin') // 1024
        file_sizes.append(file_size)
        print(f'{q}: {file_size}')
    print(file_sizes)
    plt.figure(figsize=(10, 5))
    plt.plot(qualities, file_sizes, marker='o')
    plt.title(f'Размер файла в зависимости от качества сжатия изображения {AAA}')
    plt.xlabel('Качество сжатия')
    plt.ylabel('Размер файла (Кб)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\dependency_graph_{AAA}.png")
    plt.show()
a1 = 'Lenna'
a2 = 'Lenna_gray'
a3 = 'Lenna_bw'
a4 = 'Lenna_bw_no_dither'
a5 = 'lion'
a6 = 'lion_gray'
a7 = 'lion_bw'
a8 = 'lion_bw_no_dither'
aaa = int(input('для графика: '))
if aaa == 1: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\Lenna_0.png',a1)
if aaa == 2: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\Lenna_gray.png',a2)
if aaa == 3: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\Lenna_bw.png',a3)
if aaa == 4: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\Lenna_bw_no_dither.png',a4)
if aaa == 5: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\lion.png',a1)
if aaa == 6: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\lion_gray.png',a2)
if aaa == 7: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\lion_bw.png',a3)
if aaa == 8: plot_compression_vs_quality('C:\\ЛЭТИ 2 курс\\АиСД 4 сем\\лаб2\\lion_bw_no_dither.png',a4), 1626, 1795, 1988, 2158, 2331, 2538, 2806, 3129, 3508, 4126, 4972, 6479, 9808, 22392]