import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_complex_data(real_file, imag_file, shape=(250, 4000), order='C'):
    """
    Load complex data from separate real and imaginary files with correct orientation
    
    Parameters:
        real_file: path to the file containing real part
        imag_file: path to the file containing imaginary part
        shape: tuple with the correct shape of the output array (default: 250x4000)
        order: 'F' for Fortran (column-major) or 'C' for C (row-major) order
    
    Returns:
        complex_data: complex numpy array
    """
    real_part = np.fromfile(real_file, dtype=np.float64).reshape(shape, order=order)
    imag_part = np.fromfile(imag_file, dtype=np.float64).reshape(shape, order=order)
    return real_part + 1j * imag_part

def create_sar_image(complex_data, output_png, scale_factor=15, display=True):
    """
    Create and save a SAR image as PNG
    
    Parameters:
        complex_data: complex numpy array
        output_png: path to save the PNG file
        scale_factor: factor to multiply the mean for the colorscale max (default: 15)
        display: whether to display the image using matplotlib
    """
    # Calculate magnitude (absolute value)
    magnitude = np.abs(complex_data)
    
    # Get image dimensions
    height, width = magnitude.shape
    print(f"Image dimensions: {height}x{width}")
    
    # Scale the image (similar to MATLAB's [0, 15*mean(abs(SlcMatr(:)))])
    max_val = scale_factor * np.mean(magnitude)
    
    # Normalize to 0-255 range for 8-bit image
    normalized = np.clip(magnitude * (255.0 / max_val), 0, 255).astype(np.uint8)
    
    # Create PIL image
    img = Image.fromarray(normalized)
    
    # Save as PNG
    img.save(output_png)
    print(f"Image saved to {output_png}")
    
    # Optionally display the image
    if display:
        plt.figure(figsize=(14, 10))
        plt.imshow(normalized, cmap='gray')
        plt.colorbar(label='Amplitude')
        plt.title('Focused SAR Image (Correct Orientation)')
        plt.axis('image')
        plt.tight_layout()
        plt.savefig("SAR_Image_display.png")
        plt.show()
    
    
    return img

def load_from_complex_file(complex_file, shape=(250, 4000), order='C'):
    """
    Load complex data from a single file
    """
    data = np.fromfile(complex_file, dtype=np.float32)  # Changed to float32
    
    n_elements = np.prod(shape)
    
    # Split data in half
    real_part = data[:n_elements].reshape(shape, order=order)
    imag_part = data[n_elements:].reshape(shape, order=order)
    
    # Create complex array
    complex_data = real_part + 1j * imag_part
    #complex_data = np.fliplr(complex_data)
    return complex_data


if __name__ == "__main__":
    # Correct shape of the SAR image (250x4000)
    shape = (250, 4000)  # DimAz, DimRg - correct orientation
    
    # File paths
    real_file = "SlcMatr_real.dat"
    imag_file = "SlcMatr_imag.dat"
    complex_file = "SlcGpu_RTX4090_CuPy_InterleavedF32.dat"
    #complex_file = "SlcPx_idl_matched_complex.dat"
    output_png = "SAR_Image_Correct_v2.png"
    
    # Check which files exist
    print("Select an option:")
    print("1: Load from separate real and imaginary files")
    print("2: Load from a single complex file")
    print("3: Load from 'slc_ch0_res0.300000_4000x250.dat'")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        if os.path.exists(real_file) and os.path.exists(imag_file):
            print("Loading from separate real and imaginary files...")
            complex_data = load_complex_data(real_file, imag_file, shape)
        else:
            print("Error: Real or imaginary file does not exist!")
            exit(1)
    elif choice == "2":
        if os.path.exists(complex_file):
            print("Loading from complex file...")
            complex_data = load_from_complex_file(complex_file, shape)
        else:
            print("Error: Complex file does not exist!")
            exit(1)
    elif choice == "3":
        alt_file = "vera_matrice.dat"
        if os.path.exists(alt_file):
            print(f"Loading from '{alt_file}'...")
            complex_data = load_from_complex_file(alt_file, shape)
        else:
            print(f"Error: File '{alt_file}' does not exist!")
            exit(1)
    else:
        print("Invalid choice! Please enter 1, 2, or 3.")
        exit(1)
        
    # Create and save image
    img = create_sar_image(complex_data, output_png)
    
    # Try different orientations if needed (uncomment to use)
    # Sometimes the data might need to be transposed or flipped
    
    # Option 1: Transpose
    #img_transposed = create_sar_image(complex_data.T, "SAR_Image_Transposed.png")
    
    # Option 2: Flip vertically
    #img_flipped_v = create_sar_image(np.flipud(complex_data), "SAR_Image_FlippedV.png")
    
    # Option 3: Flip horizontally
    #img_flipped_h = create_sar_image(np.fliplr(complex_data), "SAR_Image_FlippedH.png")
    
    print("\nProcess completed. Please check the generated image.")
    print("If the image still doesn't look right, uncomment the alternative orientation")
    print("options in the script to generate different views.")