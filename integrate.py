import cv2
import numpy as np
import itertools

def divide_into_blocks(image, M, N):
    h, w, c = image.shape
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

def block_regulation(block, d):
    grayscale_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    M, N = grayscale_block.shape
    regulated_block = np.copy(grayscale_block)
    
    for i in range(0, M, 2):
        for j in range(0, N, 2):
            if i+1 < M and j+1 < N:
                V = [grayscale_block[i, j], grayscale_block[i, j+1],
                     grayscale_block[i+1, j], grayscale_block[i+1, j+1]]
                R = [0, d, 2*d, 3*d]
                A = [V[k] + R[k] for k in range(4)]
                if not (0 <= min(A) and max(A) <= 255):
                    continue  # Skip if out of grayscale range
                A_sorted = sorted(A)
                regulated_block[i, j], regulated_block[i, j+1], \
                regulated_block[i+1, j], regulated_block[i+1, j+1] = A_sorted
    
    return cv2.cvtColor(regulated_block, cv2.COLOR_GRAY2BGR)

def process_image(image_path, M, N, key, d, save_scrambled=True):
    image = cv2.imread(image_path)
    blocks, h, w = divide_into_blocks(image, M, N)
    
    # Apply block regulation
    regulated_blocks = [block_regulation(block, d) for block in blocks]
    
    scrambled_blocks, indices = scramble_blocks(regulated_blocks, key)
    scrambled_image = reconstruct_image(scrambled_blocks, h, w, M, N)
    
    if save_scrambled:
        cv2.imwrite("scrambled_image.png", scrambled_image)
    
    return scrambled_image, indices, h, w

# Example Usage
M, N = 64, 64  # Block size
d = 5  # Common difference for block regulation
key = 1234  # Scrambling key
scrambled_img, scramble_indices, h, w = process_image('images/Baboon_Gray.png', M, N, key, d)
cv2.imshow('Scrambled Image', scrambled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
