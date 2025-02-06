import cv2
import numpy as np
import itertools

def divide_into_blocks(image, M, N):
    h,w,c = image.shape
    blocks = []
    for i in range(0, h, M):
        for j in range(0, w, N):
            block = image[i:i+M, j:j+N]
            blocks.append(block)
    return blocks, h, w

def reconstruct_image(blocks, h, w, M, N):
    reconstructed = np.zeros((h, w, 3), dtype=np.uint8)
    index = 0
    for i in range(0, h, M):
        for j in range(0, w, N):
            reconstructed[i:i+M, j:j+N] = blocks[index]
            index += 1
    return reconstructed
def scramble_pixels(block, key):
    np.random.seed(key)
    flat_block = block.reshape(-1, 3)
    indices = np.arange(len(flat_block))
    np.random.shuffle(indices)
    scrambled_block = flat_block[indices].reshape(block.shape)
    return scrambled_block, indices

def scramble_blocks(blocks, key):
    np.random.seed(key)
    indices = np.arange(len(blocks))
    np.random.shuffle(indices)
    scrambled = [blocks[i] for i in indices]
    scrambled = [scramble_pixels(block, key)[0] for block in scrambled]
    return scrambled, indices

def descramble_pixels(scrambled_block, indices):
    flat_scrambled = scrambled_block.reshape(-1, 3)
    original_order = np.zeros_like(flat_scrambled)
    for i, idx in enumerate(indices):
        original_order[idx] = flat_scrambled[i]
    return original_order.reshape(scrambled_block.shape)

def descramble_blocks(scrambled_blocks, indices):
    original_order = np.zeros(len(scrambled_blocks), dtype=object)
    for i, idx in enumerate(indices):
        original_order[idx] = scrambled_blocks[i]
    return list(original_order)

def process_image(image_path, M, N, key, save_scrambled=True):
    image = cv2.imread(image_path)
    blocks, h, w = divide_into_blocks(image, M, N)
    scrambled_blocks, indices = scramble_blocks(blocks, key)
    scrambled_image = reconstruct_image(scrambled_blocks, h, w, M, N)
    if save_scrambled:
        cv2.imwrite("scrambled_image.png", scrambled_image)
    return scrambled_image, indices,h,w

# Example Usage
M, N = 64,64 # Block size
key = 1234  # Scrambling key
scrambled_img, scramble_indices ,h,w= process_image('images/Baboon_Gray.png', M, N, key)
cv2.imshow('Scrambled Image', scrambled_img)
print(h,w)
cv2.waitKey(0)

