import numpy as np
import math
import itertools
import cv2
import time
# import os
from skimage.metrics import structural_similarity as ssim

def load_image(image_path):
    """Load image as grayscale numpy array"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img

def save_image(image, output_path):
    """Save image to file"""
    cv2.imwrite(output_path, image)

def divide_into_blocks(matrix, M=2, N=2):
    """Divide matrix into blocks with padding if needed"""
    h, w = matrix.shape
    # Calculate padding needed
    pad_h = (M - h % M) % M
    pad_w = (N - w % N) % N
   
    if pad_h > 0 or pad_w > 0:
        matrix = np.pad(matrix, ((0, pad_h), (0, pad_w)), mode='constant')
    
    blocks = []
    new_h, new_w = matrix.shape
    for i in range(0, new_h, M):
        for j in range(0, new_w, N):
            block = matrix[i:i+M, j:j+N]
            blocks.append(block)
    return blocks, new_h, new_w, (pad_h, pad_w)

def reconstruct_matrix(blocks, h, w, pad_size, M=2, N=2):
    """Reconstruct matrix removing any padding"""
    r = np.zeros((h, w), dtype=np.uint8)
    index = 0
    for i in range(0, h, M):
        for j in range(0, w, N):
            r[i:i+M, j:j+N] = blocks[index]
            index += 1
    # Remove padding
    pad_h, pad_w = pad_size
    if pad_h > 0 or pad_w > 0:
        r = r[:-pad_h, :-pad_w] if pad_h else r[:, :-pad_w]
    return r

def scramble_pixels(block, key):
    np.random.seed(key)
    flat_block = block.reshape(-1)
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

def unscramble_pixels(block, indices):
    flat_block = block.reshape(-1)
    unscrambled = np.zeros_like(flat_block)
    unscrambled[indices] = flat_block
    return unscrambled.reshape(block.shape)

def unscramble_blocks(blocks, block_indices, pixel_indices_key):
    blocks = [unscramble_pixels(block, np.random.RandomState(pixel_indices_key).permutation(block.size)) 
             for block in blocks]
    original_blocks = [None] * len(blocks)
    for new_pos, original_pos in enumerate(block_indices):
        original_blocks[original_pos] = blocks[new_pos]
    return original_blocks

def encrypt(matrix, encryption_key):
    blocks, h, w, pad_size = divide_into_blocks(matrix)
    scrambled_blocks, indices = scramble_blocks(blocks, encryption_key)
    # encrypted_matrix = reconstruct_matrix(scrambled_blocks, h, w, pad_size)
    return scrambled_blocks, indices, pad_size,h,w

def decrypt(encrypted_blocks, encryption_key, block_indices, pad_size,h,w):
    # blocks, h, w, _ = divide_into_blocks(encrypted_matrix)
    original_blocks = unscramble_blocks(encrypted_blocks, block_indices, encryption_key)
    decrypted_matrix = reconstruct_matrix(original_blocks, h, w, pad_size)
    return decrypted_matrix

def regulate(block, d=10):
    V = block.flatten()
    R = np.array([0, d, 2*d, 3*d])
    A = V + R
    return A.reshape(block.shape)

def apply_regularization(encrypted_blocks, d=10):
    # blocks, h, w, pad_size = divide_into_blocks(encrypted_matrix)
    regulated_blocks = [regulate(block, d) for block in encrypted_blocks]
    # regulated_matrix = reconstruct_matrix(regulated_blocks, h, w, pad_size)
    return regulated_blocks

def create_location_map(regulated_blocks, M=2, N=2, d=10,w = 0):
    # blocks, h, w, pad_size = divide_into_blocks(regulated_matrix, M, N)
    location_map = []
    anti_location_map = []
    new_regulated_blocks = []
    
    for i, block in enumerate(regulated_blocks):
        block_flat = block.flatten()
        row, col = (i // (w // N)) * M, (i % (w // N)) * N
        sorted_block = np.sort(block_flat)
        flag = True
        for j in range(len(block_flat)):
            if sorted_block[j] != block_flat[j] or block_flat[j] >= 255:
                flag = False
                break
        if not flag:
            location_map.append((row, col))
            original_block = block_flat - np.array([0, d, 2*d, 3*d])
            new_regulated_blocks.append(original_block.reshape(block.shape))
        else:
            anti_location_map.append((row, col))
            new_regulated_blocks.append(block)
    # new_regulated_matrix = reconstruct_matrix(new_regulated_blocks, h, w, pad_size, M, N)
    return location_map, anti_location_map, new_regulated_blocks

def embed_data(secret_bits, location_map, regulated_blocks, M=2, N=2,w=0):
    # blocks, h, w, pad_size = divide_into_blocks(regulated_matrix, M, N)
    stego_blocks = []
    bit_index = 0
    total_embedded = 0
    final = 0
    
    location_positions = set((row, col) for row, col in location_map)
    
    for i, block in enumerate(regulated_blocks):
        row = (i // (w // N)) * M
        col = (i % (w // N)) * N
        
        if (row, col) not in location_positions:
            A_flat = block.flatten()
            #A_sorted = sorted(A_flat)
            unique_perms = list(set(itertools.permutations(A_flat)))
            num_perms = len(unique_perms)
            
            if num_perms > 1:
                bit_capacity = int(np.log2(num_perms))
                final += bit_capacity
                if bit_index + bit_capacity > len(secret_bits):
                    remaining_bits = len(secret_bits) - bit_index
                    data_segment = secret_bits[bit_index:] + [0] * (bit_capacity - remaining_bits)
                else:
                    data_segment = secret_bits[bit_index:bit_index + bit_capacity]
                
                if data_segment:
                    index = int(''.join(map(str, data_segment)), 2) % num_perms
                    new_block = np.array(unique_perms[index]).reshape((M, N))
                    stego_blocks.append(new_block)
                    bit_index += bit_capacity
                    total_embedded += bit_capacity
                else:
                    stego_blocks.append(block)
            else:
                stego_blocks.append(block)
        else:
            stego_blocks.append(block)
    
    # stego_matrix = reconstruct_matrix(stego_blocks, h, w, pad_size, M, N)
    return stego_blocks, total_embedded, final

def extract_data(location_map, stego_blocks, M=2, N=2,w=0):
    # blocks, h, w, pad_size = divide_into_blocks(stego_matrix, M, N)
    extracted_bits = []
    
    location_positions = set((row, col) for row, col in location_map)
    
    for i, block in enumerate(stego_blocks):
        row = (i // (w // N)) * M
        col = (i % (w // N)) * N
        
        if (row, col) not in location_positions:
            A_flat = block.flatten()
            A_sorted = sorted(A_flat)
            unique_perms = list(set(itertools.permutations(A_sorted)))
            num_perms = len(unique_perms)
            
            if num_perms > 1:
                bit_capacity = int(np.log2(num_perms))
                current_perm = tuple(A_flat)
                try:
                    index = unique_perms.index(current_perm)
                    binary_str = bin(index)[2:].zfill(bit_capacity)
                    extracted_bits.extend([int(bit) for bit in binary_str])
                except ValueError:
                    continue
    
    return extracted_bits

def recover_original_matrix(stego_blocks, location_map, d=10, M=2, N=2,w=0):
    # blocks, h, w, pad_size = divide_into_blocks(stego_matrix, M, N)
    recovered_blocks = []
    
    location_positions = set((row, col) for row, col in location_map)
    
    for i, block in enumerate(stego_blocks):
        row = (i // (w // N)) * M
        col = (i % (w // N)) * N
        
        if (row, col) not in location_positions:
            sorted_block = np.sort(block.flatten()).reshape(block.shape)
            deregulated = sorted_block - np.array([0, d, 2*d, 3*d]).reshape(block.shape)
            recovered_blocks.append(deregulated)
        else:
            recovered_blocks.append(block)
    
    # return reconstruct_matrix(recovered_blocks, h, w, pad_size, M, N)
    return recovered_blocks

def calculate_psnr(original, decrypted):
    mse = np.mean((original - decrypted) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ec(image, total_ec):
    total_pixels = image.shape[0] * image.shape[1]
    num_embedded_bits = total_ec
    ec = num_embedded_bits / total_pixels
    return ec, num_embedded_bits

def process_image(image_path, output_dir, secret_bits=[0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1], encryption_key=42, d=3):
    """Complete image processing pipeline"""
    # Load image
    original_img = load_image(image_path)
    
    # Encryption
    encrypted_blocks, encryption_indices, pad_size,h,w = encrypt(original_img, encryption_key)
    
    # Regularization
    regulated_blocks = apply_regularization(encrypted_blocks, d)
    
    # Location map creation
    location_map, _, new_regulated_blocks = create_location_map(regulated_blocks, d=d,w=w)
    
    # Data embedding
    stego_blocks, TE, final= embed_data(secret_bits, location_map, new_regulated_blocks,w=w)
    
    # Data extraction
    extracted_bits = extract_data(location_map, stego_blocks ,w=w)
    
    # Recovery
    recovered_blocks = recover_original_matrix(stego_blocks, location_map, d,w=w)
    
    # Decryption
    decrypted_img = decrypt(recovered_blocks, encryption_key, encryption_indices, pad_size,h,w)
    
    # Save results
    # os.makedirs(output_dir, exist_ok=True)
    # save_image(encrypted_img, f"{output_dir}/encrypted.png")
    # save_image(regulated_img, f"{output_dir}/regulated.png")
    # save_image(stego_img, f"{output_dir}/stego.png")
    # save_image(recovered_img, f"{output_dir}/recovered.png")
    # save_image(decrypted_img, f"{output_dir}/decrypted.png")
    
    # Calculate metrics
    # display_bits = extracted_bits[:5] if len(extracted_bits) >= 5 else extracted_bits
    psnr = calculate_psnr(original_img, decrypted_img)
    ssim_val = ssim(original_img, decrypted_img, data_range=255)
    # ssim_valtemp = ssim(recovered_img, encrypted_img, data_range=255)
    ec, total_embedded = calculate_ec(original_img, TE)
    
    #print(f"Extracted bits: {display_bits}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    # print(f"SSIMtemp: {ssim_valtemp:.4f}")
    print(f"TE: {TE:.4f}")
    print(f"EC(bpp): {ec:.4f}")
    print("Message successfully extracted:", secret_bits == extracted_bits[:len(secret_bits)])
    return decrypted_img

if __name__ == "__main__":
    input_image = "images/Boat.bmp"
    output_dir = "results"
    start_time = time.time()
    recovered_image = process_image(input_image, output_dir)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.4f} seconds")