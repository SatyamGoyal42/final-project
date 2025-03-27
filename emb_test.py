import numpy as np
import cv2
import itertools
from skimage.metrics import structural_similarity as ssim
import math

def reconstruct_image(blocks, h, w, M=2, N=2):
    r = np.zeros((h, w), dtype=np.uint8)
    index = 0
    for i in range(0, h, M):
        for j in range(0, w, N):
            r[i:i+M, j:j+N] = blocks[index]
            index += 1
    return r
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
    is_mono = np.all(np.diff(A) > 0) or np.all(np.diff(A) < 0)
    in_range = np.all((A >= 0) & (A < 255))
    
    return is_mono and in_range

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

def embed_final(image_path, d=3, secret_bits=None):
    
    if secret_bits is None:
        secret_bits = []
            
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Embed data 
    embedding_blocks = divide_blocks_for_embedding(image)
    regulated_blocks, location_map = create_lmap(embedding_blocks, d)
    final_blocks = embed(regulated_blocks, secret_bits)
    
    # Reconstruct final image with embedded data
    embedded_image = image.copy()
    for pos, block in final_blocks.items():
        i, j = pos
        embedded_image[i:i+2, j:j+2] = block #this is the final image with embedded data
    
    return embedded_image, location_map

def de_embed(embedded_image, d, location_map):

    blocks = divide_blocks_for_embedding(embedded_image)
    de_embedded_blocks = {}

    for block, pos in blocks:
        if pos not in location_map:  # Process only embedded blocks
            A = block.flatten()
            sorted_block = np.sort(A)
            
            # ✅ Subtract the arithmetic progression
            R = np.array([0, d, 2 * d, 3 * d])
            de_embedded_block = sorted_block - R
            
            # Ensure pixel values are clamped between 0-255
            de_embedded_block = np.clip(de_embedded_block, 0, 255).reshape((2, 2))
            
            de_embedded_blocks[pos] = de_embedded_block
        else:
            # Keep non-embedded blocks as they are
            de_embedded_blocks[pos] = block

    # Create a new image with de-embedded blocks
    de_embedded_image = embedded_image.copy()
    
    for pos, block in de_embedded_blocks.items():
        i, j = pos
        de_embedded_image[i:i+2, j:j+2] = block

    return de_embedded_image

def extract_message(embedded_image, location_map):
    
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

def extract(embedded_image, d=3, location_map=None):
   
    if location_map is None:
        location_map = []
        
    # First extract the message before decryption
    extracted_message = extract_message(embedded_image, location_map)
    de_embedded_image = de_embed(embedded_image, d, location_map)
    

    return  extracted_message , de_embedded_image

def calculate_psnr(original, decrypted):
    mse = np.mean((original - decrypted) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_ssim(original, decrypted):
    score, _ = ssim(original, decrypted, full=True)
    return score

def example():

    image_path = "images/Baboon_Gray.png"
    d = 3
    secret_bits = []
    
    # Encrypt and embed
    embedded_image, location_map = embed_final(
        image_path,  
        d=d, 
        secret_bits=secret_bits
    )
    
    # Save the encrypted and embedded image
    output_path = "emb_test.png"
    cv2.imwrite(output_path, embedded_image)
        
    # Decrypt and extract
    print("running1")
    extracted_message,decrypted_image = extract(
        embedded_image,
        d=d,
        location_map=location_map
    )
    
    # Save the decrypted image
    print("running2")
    decrypted_path = "de_emd_test.png"
    cv2.imwrite(decrypted_path, decrypted_image)
    
    
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Verification
    print("\nVerification:")
    print("Message successfully extracted:", secret_bits == extracted_message[:len(secret_bits)])

    # Calculate EC,PSNR and SSIM
    psnr_value = calculate_psnr(original_image, decrypted_image)
    ssim_value = calculate_ssim(original_image, decrypted_image)

    
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    decryption_successful = np.array_equal(original_image, decrypted_image)
    print("Image successfully decrypted:", decryption_successful)

if __name__ == "__main__":
    example() 