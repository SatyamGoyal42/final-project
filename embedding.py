import numpy as np
import cv2
import itertools
import random

#block division into 2x2 blocks
def divide_blocks(image, block_size=(2, 2)):
    M, N = image.shape
    blocks = []
    for i in range(0, M, block_size[0]):
        for j in range(0, N, block_size[1]):
            block = image[i:i+block_size[0], j:j+block_size[1]]
            if block.shape == block_size:
                blocks.append((block, (i, j)))
    return blocks

#block regulation using a difference value "d"
def regulate(block, d):
    V = block.flatten()
    R = np.array([0, d, 2*d, 3*d])
    A = V + R
    return A

#check if the block is monotonic in increasing order
def is_monotonic(A):
    return np.all(np.diff(A) > 0) or np.all(np.diff(A) < 0)

#location map function
def create_lmap(blocks, d):
    location_map = []
    regulated_blocks = {}
    for idx, (block, pos) in enumerate(blocks):
        A = regulate(block, d)
        if is_monotonic(A):
            regulated_blocks[pos] = A.reshape((2,2))
        else:
            location_map.append(pos)
    return regulated_blocks, location_map

#embedding secret data into the blocks
def embed(blocks, secret_bits):
    permuted_blocks = {}
    bit_index = 0
    for pos, A in blocks.items():
        A_sorted = sorted(A.flatten())
        permutations = list(itertools.permutations(A_sorted))
        num_permutations = len(permutations)
        bit_capacity = int(np.log2(num_permutations))
        if bit_index + bit_capacity <= len(secret_bits):
            data_segment = secret_bits[bit_index:bit_index + bit_capacity]
            index = int(''.join(map(str, data_segment)), 2)
            new_perm = np.array(permutations[index]).reshape((2,2))
            permuted_blocks[pos] = new_perm
            bit_index += bit_capacity
        else:
            permuted_blocks[pos] = A.reshape((2,2))
    return permuted_blocks

#genrating permutation table
def permutation_table(image, alpha=7, beta=13):
    M, N = image.shape
    hash_values = np.zeros((M//2, N//2))
    for i in range(0, M, 2):
        for j in range(0, N, 2):
            block = image[i:i+2, j:j+2]
            mean_val = np.mean(block)
            h = (alpha * mean_val + beta) % 255
            hash_values[i//2, j//2] = h
    return hash_values
#mainnfunction where all the functions are called
def process_image(image_path, d, secret_bits):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blocks = divide_blocks(image)
    regulated_blocks, location_map = create_lmap(blocks, d)
    permuted_blocks = embed(regulated_blocks, secret_bits)
    perm_table = permutation_table(image)
    return permuted_blocks, location_map, perm_table

