import numpy as np
import cv2
import itertools
from skimage.metrics import structural_similarity as ssim
import math

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


def encrypt(image_path, encryption_key):
            
    # Read and encrypt the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blocks, h, w = divide_into_blocks(image)
    scrambled_blocks, indices = scramble_blocks(blocks, encryption_key)
    encrypted_image = reconstruct_image(scrambled_blocks, h, w) #this is the encrypted image
            
    return encrypted_image, indices

def decrypt(embedded_image, encryption_key,block_indices=None):
   
    if block_indices is None:
        block_indices = []
        
   
    h, w = embedded_image.shape
    blocks, _, _ = divide_into_blocks(embedded_image)
    
    # Unscramble the blocks and pixels
    original_blocks = unscramble_blocks(blocks, block_indices, encryption_key)
    
    # Reconstruct the original image
    decrypted_image = reconstruct_image(original_blocks, h, w)
    
    return decrypted_image

def calculate_ssim(original, decrypted):
    score, _ = ssim(original, decrypted, full=True)
    return score

def calculate_psnr(original, decrypted):
    mse = np.mean((original - decrypted) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def example():

    image_path = "images/Baboon_Gray.png"
    encryption_key = 42
    result_image, encryption_indices = encrypt(
        image_path, 
        encryption_key, 
    )

    output_path = "encryption_test.png"
    cv2.imwrite(output_path, result_image)
    
    
    decrypted_image = decrypt(
        result_image,
        encryption_key,
        block_indices=encryption_indices,
    )
    
    # Save the decrypted image
    decrypted_path = "decrypted_test.png"
    cv2.imwrite(decrypted_path, decrypted_image)
    
    
    ogimg = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Verification
    print("\nVerification:")

    # Calculate EC,PSNR and SSIM
    psnr_value = calculate_psnr(ogimg, decrypted_image)
    ssim_value = calculate_ssim(ogimg, decrypted_image)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    dsm = np.array_equal(ogimg, decrypted_image)
    print("Image successfully decrypted:", dsm)

if __name__ == "__main__":
    example() 