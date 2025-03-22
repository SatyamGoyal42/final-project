import numpy as np
import cv2
import itertools
from skimage.metrics import structural_similarity as ssim
import math


# Encryption Functions
def divide_into_blocks(image, M=2, N=2):
    h, w = image.shape
    blocks = []
    for i in range(0, h, M):
        for j in range(0, w, N):
            block = image[i:i+M, j:j+N]
            blocks.append(block)
    return blocks, h, w

def reconstruct_image(blocks, h, w, M=2, N=2):
    r = np.zeros((h, w), dtype=np.uint8)
    index = 0
    for i in range(0, h, M):
        for j in range(0, w, N):
            r[i:i+M, j:j+N] = blocks[index]
            index += 1
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

# Embedding Functions
def divide_blocks_for_embedding(image):
    M, N = image.shape
    blocks = []
    for i in range(0, M, 2):
        for j in range(0, N, 2):
            block = image[i:i+2, j:j+2]
            if block.shape == (2, 2):
                blocks.append((block, (i, j)))
    return blocks

def regulate(block, d):
    V = block.flatten()
    R = np.array([0, d, 2*d, 3*d])
    A = V + R
    return A

def is_monotonic(A):
    return np.all(np.diff(A) > 0) or np.all(np.diff(A) < 0)

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

# Integrated System Function
def encrypt_and_embed(image_path, encryption_key, block_size_encryption=(2, 2), d=3, secret_bits=None):
    
    if secret_bits is None:
        secret_bits = []
        
    # Step 1: Read and encrypt the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blocks, h, w = divide_into_blocks(image)
    scrambled_blocks, indices = scramble_blocks(blocks, encryption_key)
    encrypted_image = reconstruct_image(scrambled_blocks, h, w) #this is the encrypted image
    
    # Step 2: Embed data in encrypted image
    embedding_blocks = divide_blocks_for_embedding(encrypted_image)
    regulated_blocks, location_map = create_lmap(embedding_blocks, d)
    final_blocks = embed(regulated_blocks, secret_bits)
    
    # Step 3: Reconstruct final image with embedded data
    result_image = encrypted_image.copy()
    for pos, block in final_blocks.items():
        i, j = pos
        result_image[i:i+2, j:j+2] = block #this is the final image with embedded data
    
    return result_image, indices, location_map

# Decryption and Extraction Functions
def unscramble_pixels(block, indices):
    flat_block = block.reshape(-1)
    unscrambled = np.zeros_like(flat_block)
    unscrambled[indices] = flat_block
    return unscrambled.reshape(block.shape)

def unscramble_blocks(blocks, block_indices, pixel_indices_key):
    # Unscramble pixels within each block
    blocks = [unscramble_pixels(block, np.random.RandomState(pixel_indices_key).permutation(block.size)) 
             for block in blocks]
    # Unscramble block positions
    original_blocks = [None] * len(blocks)
    for new_pos, original_pos in enumerate(block_indices):
        original_blocks[original_pos] = blocks[new_pos]
    return original_blocks

def extract_message(embedded_image, d, location_map):
    
    extracted_bits = []
    blocks = divide_blocks_for_embedding(embedded_image)
    
    for block, pos in blocks:
        if pos not in location_map:  # Only process blocks that were used for embedding
            A = block.flatten()
            A_sorted = sorted(A)
            permutations = list(itertools.permutations(A_sorted))
            current_perm = tuple(A)
            
            try:
                index = permutations.index(current_perm)
                bit_capacity = int(np.log2(len(permutations)))
                binary = format(index, f'0{bit_capacity}b')
                extracted_bits.extend([int(b) for b in binary])
            except ValueError:
                continue
    
    return extracted_bits

def decrypt_and_extract(embedded_image, encryption_key, block_size_encryption=(2, 2), 
                       block_indices=None, d=3, location_map=None):
   
    if location_map is None:
        location_map = []
    if block_indices is None:
        block_indices = []
        
    # First extract the message before decryption
    extracted_message = extract_message(embedded_image, d, location_map)
    
    # Then decrypt the image
    h, w = embedded_image.shape
    blocks, _, _ = divide_into_blocks(embedded_image)
    
    # Unscramble the blocks and pixels
    original_blocks = unscramble_blocks(blocks, block_indices, encryption_key)
    
    # Reconstruct the original image
    decrypted_image = reconstruct_image(original_blocks, h, w)
    
    return decrypted_image, extracted_message

# PSNR and SSIM and EC
def calculate_psnr(original, decrypted):
    mse = np.mean((original - decrypted) ** 2)
    if mse == 0:
        return float('inf')  # No difference
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(original, decrypted):
    score, _ = ssim(original, decrypted, full=True)
    return score

def calculate_ec(image, secret_bits):
    total_pixels = image.shape[0] * image.shape[1]
    num_embedded_bits = len(secret_bits)
    ec = num_embedded_bits / total_pixels
    return ec, num_embedded_bits

def example():

    image_path = "images/Baboon_Gray.png"
    encryption_key = 42
    d = 3
    secret_bits = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]*10000
    
    # Encrypt and embed
    result_image, encryption_indices, location_map = encrypt_and_embed(
        image_path, 
        encryption_key, 
        d=d, 
        secret_bits=secret_bits
    )
    
    # Save the encrypted and embedded image
    output_path = "final.png"
    cv2.imwrite(output_path, result_image)
    
    print("Encryption and Embedding:")
    print(f"Original image: {image_path}")
    print(f"Result saved as: {output_path}")
    # print("Encryption indices:", encryption_indices)
    # print("Location map:", location_map)
    print(f"Number of blocks in location map: {len(location_map)}")
    
    # Decrypt and extract
    decrypted_image, extracted_message = decrypt_and_extract(
        result_image,
        encryption_key,
        block_indices=encryption_indices,
        d=d,
        location_map=location_map
    )
    
    # Save the decrypted image
    decrypted_path = "decrypted.png"
    cv2.imwrite(decrypted_path, decrypted_image)
    
    print("\nDecryption and Extraction:")
    print(f"Decrypted image saved as: {decrypted_path}")
    #print("Original message:", secret_bits)
    #print("Extracted message:", extracted_message[:len(secret_bits)])
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Verification
    print("\nVerification:")
    print("Message successfully extracted:", secret_bits == extracted_message[:len(secret_bits)])

    # Calculate EC,PSNR and SSIM
    ec,l = calculate_ec(result_image, secret_bits)
    psnr_value = calculate_psnr(original_image, decrypted_image)
    ssim_value = calculate_ssim(original_image, decrypted_image)

    print(f"EC: {l}")
    print(f"EC(bpp): {ec:.4f} bpp")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    # decryption_successful = np.array_equal(original_image, decrypted_image)
   # print("Image successfully decrypted:", decryption_successful)

if __name__ == "__main__":
    example() 