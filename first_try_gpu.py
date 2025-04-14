import numpy as np
import cupy as cp
import time
import os
from numba import cuda, float64, float32, int32, uint8, complex64, complex128, void
import math

# ###########################################################################
# # CUDA Kernel for Processing Multiple Points in Parallel
# ###########################################################################
@cuda.jit
def process_points_cuda(xxg1_flat, yyg1_flat, zzg1_flat,  # Input coordinates (float64)
                        mask_flat, index_flat, Sint_flat,  # Input parameters (int/byte)
                        wgsx, wgsy, wgsz,                  # Trajectory arrays (float64)
                        Raw_Data,                          # Raw data (complex64)
                        near_range, dra, lambda_val,       # Parameters (float64)
                        Nra, Naz,                          # Dimensions (int)
                        ind_line_ra_min, index_min,        # Offsets (int)
                        focused_sample):                    # Output array (complex128)
    """
    CUDA kernel to process multiple pixels in parallel.
    Each thread processes one pixel in the output grid.
    """
    # Get thread ID
    ii = cuda.grid(1)
    
    # Check if thread ID is within bounds
    if ii >= xxg1_flat.shape[0]:
        return
        
    # Precision Constants (defined directly in the kernel)
    C_F64 = 2.9979246e8
    ALFA_F64 = 3.335987619777693e11
    PI_F64 = 3.141592741012573242
    BETA_F64 = 4.0 * PI_F64 * ALFA_F64 / (C_F64 * C_F64)

    # --- Main Processing Logic ---
    
    # Skip if masked
    if mask_flat[ii] != 1:
        focused_sample[ii] = complex(0.0, 0.0)
        return
        
    # Get sint value for this point
    sint = Sint_flat[ii]
    # Check for invalid value
    if sint == -9999:
        focused_sample[ii] = complex(0.0, 0.0)
        return
        
    # Calculate jj size (synthetic aperture indices)
    jj_size = int(math.ceil(sint / 2.0)) * 2 + 1
    
    # Initialize accumulator for back projection sum
    back_projected_sum_real = 0.0
    back_projected_sum_imag = 0.0
    n_valid_contributors = 0
    
    # Aperture processing loop
    for j in range(jj_size):
        # Calculate index in azimuth (absolute)
        index1 = index_flat[ii] + j - int(math.ceil(abs(sint) / 2.0))
        
        # Check if index is within valid azimuth range
        if index1 >= Naz - 1 or index1 <= 0:
            continue
            
        # Calculate relative position
        xcomp = xxg1_flat[ii] - wgsx[index1]
        ycomp = yyg1_flat[ii] - wgsy[index1]
        zcomp = zzg1_flat[ii] - wgsz[index1]
        
        # Calculate distance squared and distance
        dist_patch2 = xcomp*xcomp + ycomp*ycomp + zcomp*zcomp
        dist_patch = math.sqrt(dist_patch2)
        
        # Calculate range index
        ind_line_ra = int(round((dist_patch - near_range) / dra))
        
        # Check if range index is valid
        if ind_line_ra >= Nra - 1 or ind_line_ra < ind_line_ra_min:
            continue
            
        # Calculate phase term
        phase_term = (-4.0 * PI_F64 / lambda_val * dist_patch) + (BETA_F64 * dist_patch2)
        
        # Complex conjugate filter (using separate real and imaginary parts)
        cos_phase = math.cos(-phase_term)
        sin_phase = math.sin(-phase_term)
        
        # Get raw data value (complex64)
        raw_real = Raw_Data[ind_line_ra, index1].real
        raw_imag = Raw_Data[ind_line_ra, index1].imag
        
        # Compute contribution (complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i)
        term_real = dist_patch2 * (raw_real * cos_phase - raw_imag * sin_phase)
        term_imag = dist_patch2 * (raw_real * sin_phase + raw_imag * cos_phase)
        
        # Accumulate
        back_projected_sum_real += term_real
        back_projected_sum_imag += term_imag
        n_valid_contributors += 1
    
    # Set output
    if n_valid_contributors > 0:
        focused_sample[ii] = complex(back_projected_sum_real / n_valid_contributors, 
                                    back_projected_sum_imag / n_valid_contributors)
    else:
        focused_sample[ii] = complex(0.0, 0.0)


# ###########################################################################
# # GPU-Accelerated Focusing Function
# ###########################################################################
def FocalizzatoreBpGPU(xxg1, yyg1, zzg1, mask, index, wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, Sint):
    """
    GPU-accelerated SAR focusing using CUDA.
    """
    print('FOCALIZADOR VERSIÓN GPU (CUDA-Accelerated)')
    t1 = time.time()

    # Obtain dimensions
    if Raw_Data.ndim != 2:
        raise ValueError("Raw_Data must be a 2D array")
    Nra, Naz = Raw_Data.shape

    if xxg1.ndim != 2:
        raise ValueError("xxg1 must be a 2D array")
    dimx, dimy = xxg1.shape
    nx_grid, ny_grid = dimx, dimy
    dim = dimx * dimy  # Total number of pixels

    # Flatten input arrays using Fortran order ('F') as in IDL
    print("Flattening input arrays (order='F')...")
    xxg1_flat = xxg1.astype(np.float64).flatten(order='F')
    yyg1_flat = yyg1.astype(np.float64).flatten(order='F')
    zzg1_flat = zzg1.astype(np.float64).flatten(order='F')
    mask_flat = mask.astype(np.uint8).flatten(order='F')
    index_flat = index.astype(np.int32).flatten(order='F')
    Sint_flat = Sint.astype(np.int32).flatten(order='F')
    
    # Apply mask for invalid Sint values
    no_valid_indices = np.where(Sint_flat == -9999)[0]
    if no_valid_indices.size > 0:
        print(f"Masking {no_valid_indices.size} points due to Sint == -9999.")
        mask_flat[no_valid_indices] = 0

    # Constants
    ind_line_ra_min = 0
    index_min = 0

    # Initialize output array
    focused_sample = np.zeros(dim, dtype=np.complex128)

    # Transfer data to GPU
    print("Transferring data to GPU...")
    d_xxg1_flat = cuda.to_device(xxg1_flat)
    d_yyg1_flat = cuda.to_device(yyg1_flat)
    d_zzg1_flat = cuda.to_device(zzg1_flat)
    d_mask_flat = cuda.to_device(mask_flat)
    d_index_flat = cuda.to_device(index_flat)
    d_Sint_flat = cuda.to_device(Sint_flat)
    d_wgsx = cuda.to_device(wgsx)
    d_wgsy = cuda.to_device(wgsy)
    d_wgsz = cuda.to_device(wgsz)
    d_Raw_Data = cuda.to_device(Raw_Data)
    d_focused_sample = cuda.to_device(focused_sample)
    
    # CUDA kernel launch configuration
    print("Configuring CUDA kernel...")
    threads_per_block = 256  # Optimal for most NVIDIA GPUs
    blocks_per_grid = (dim + threads_per_block - 1) // threads_per_block
    
    print(f"CUDA Configuration: {blocks_per_grid} blocks with {threads_per_block} threads each")
    print(f"Processing {dim} pixels in parallel on GPU")
    
    # Launch kernel
    print("Launching CUDA kernel...")
    launch_start = time.time()
    process_points_cuda[blocks_per_grid, threads_per_block](
        d_xxg1_flat, d_yyg1_flat, d_zzg1_flat,
        d_mask_flat, d_index_flat, d_Sint_flat,
        d_wgsx, d_wgsy, d_wgsz,
        d_Raw_Data,
        near_range, dra, lambda_val,
        Nra, Naz,
        ind_line_ra_min, index_min,
        d_focused_sample
    )
    
    # Synchronize to ensure kernel completion
    cuda.synchronize()
    launch_end = time.time()
    print(f"CUDA kernel execution time: {launch_end - launch_start:.2f} seconds")
    
    # Copy result back to host
    print("Transferring results back to CPU...")
    focused_sample = d_focused_sample.copy_to_host()
    
    # Reshape output to original 2D dimensions
    print("Reforming output array to 2D (order='F')...")
    focused_sample_2d = np.reshape(focused_sample, (dimx, dimy), order='F')
    
    # Calculate elapsed time
    t2 = time.time()
    print(f"Total GPU processing time: {t2 - t1:.2f} seconds")
    
    return focused_sample_2d


# ###########################################################################
# # Alternative Implementation Using CuPy (Higher-Level GPU Programming)
# ###########################################################################
def FocalizzatoreBpCuPy(xxg1, yyg1, zzg1, mask, index, wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, Sint):
    """
    GPU-accelerated SAR focusing using CuPy for larger batch processing.
    This approach uses more memory but may be faster for certain hardware.
    """
    print('FOCALIZADOR VERSIÓN GPU (CuPy-Accelerated)')
    t1 = time.time()
    
    # Convert inputs to CuPy arrays
    xxg1_cp = cp.asarray(xxg1)
    yyg1_cp = cp.asarray(yyg1)
    zzg1_cp = cp.asarray(zzg1)
    mask_cp = cp.asarray(mask)
    index_cp = cp.asarray(index)
    Sint_cp = cp.asarray(Sint)
    wgsx_cp = cp.asarray(wgsx)
    wgsy_cp = cp.asarray(wgsy)
    wgsz_cp = cp.asarray(wgsz)
    Raw_Data_cp = cp.asarray(Raw_Data)
    
    # Dimensions
    dimx, dimy = xxg1.shape
    Nra, Naz = Raw_Data.shape
    
    # Constants
    C = cp.float64(2.9979246e8)
    ALFA = cp.float64(3.335987619777693e11)
    PI = cp.float64(3.141592741012573242)
    BETA = 4.0 * PI * ALFA / (C * C)
    
    # Apply mask for invalid Sint values
    mask_cp = cp.where(Sint_cp == -9999, 0, mask_cp)
    
    # Initialize result array
    focused_sample = cp.zeros((dimx, dimy), dtype=cp.complex128)
    
    # Process in batches to avoid excessive memory usage
    batch_size = 1000  # Adjust based on your GPU memory
    
    for batch_start in range(0, dimx*dimy, batch_size):
        batch_end = min(batch_start + batch_size, dimx*dimy)
        print(f"\rProcessing batch {batch_start//batch_size + 1}/{(dimx*dimy + batch_size - 1)//batch_size}", end="")
        
        # Get indices for this batch
        indices = cp.arange(batch_start, batch_end)
        x_indices = indices // dimy
        y_indices = indices % dimy
        
        # Extract values for this batch
        xxg1_batch = xxg1_cp[x_indices, y_indices]
        yyg1_batch = yyg1_cp[x_indices, y_indices]
        zzg1_batch = zzg1_cp[x_indices, y_indices]
        mask_batch = mask_cp[x_indices, y_indices]
        index_batch = index_cp[x_indices, y_indices]
        Sint_batch = Sint_cp[x_indices, y_indices]
        
        # Skip masked pixels
        valid_pixels = cp.where(mask_batch == 1)[0]
        if len(valid_pixels) == 0:
            continue
            
        # Process only valid pixels
        for pixel_idx in valid_pixels:
            sint = Sint_batch[pixel_idx]
            if sint == -9999:
                continue
                
            # Calculate synthetic aperture indices
            jj_size = int(cp.ceil(sint / 2.0)) * 2 + 1
            jj = cp.arange(jj_size)
            
            # Calculate azimuth indices
            index1 = index_batch[pixel_idx] + jj - int(cp.ceil(cp.abs(sint) / 2.0))
            
            # Find valid indices
            valid_az = cp.where((index1 < Naz - 1) & (index1 > 0))[0]
            if len(valid_az) == 0:
                continue
                
            # Get valid indices
            index1a = index1[valid_az]
            
            # Geometric calculations
            xcomp = xxg1_batch[pixel_idx] - wgsx_cp[index1a]
            ycomp = yyg1_batch[pixel_idx] - wgsy_cp[index1a]
            zcomp = zzg1_batch[pixel_idx] - wgsz_cp[index1a]
            
            # Distance calculations
            dist_patch2 = xcomp*xcomp + ycomp*ycomp + zcomp*zcomp
            dist_patch = cp.sqrt(dist_patch2)
            
            # Range indices
            ind_line_ra = cp.round((dist_patch - near_range) / dra).astype(cp.int32)
            
            # Find valid range indices
            valid_range = cp.where((ind_line_ra < Nra - 1) & (ind_line_ra >= 0))[0]
            if len(valid_range) == 0:
                continue
                
            # Select valid values
            dist_patch_in = dist_patch[valid_range]
            dist_patch2_in = dist_patch2[valid_range]
            
            # Phase calculations
            phase_term = (-4.0 * PI / lambda_val * dist_patch_in) + (BETA * dist_patch2_in)
            conj_filter = cp.exp(cp.complex128(-1j) * phase_term)
            
            # Get raw data indices
            r_indices = ind_line_ra[valid_range]
            a_indices = index1a[valid_range]
            
            # Get raw data values
            raw_values = Raw_Data_cp[r_indices, a_indices]
            
            # Calculate backprojected sum
            contributions = dist_patch2_in * raw_values * conj_filter
            if len(contributions) > 0:
                focused_sample[x_indices[pixel_idx], y_indices[pixel_idx]] = cp.sum(contributions) / len(contributions)
    
    print("\nCuPy processing completed.")
    
    # Transfer result back to CPU
    result = cp.asnumpy(focused_sample)
    
    t2 = time.time()
    print(f"Total CuPy processing time: {t2 - t1:.2f} seconds")
    
    return result


# ###########################################################################
# # Main Execution Script
# ###########################################################################
if __name__ == "__main__":
    start_time_total = time.time()

    print("="*60)
    print(" SAR Back-projection Focusing (GPU-Accelerated)")
    print(" RTX 4090 Optimized Version ")
    print("="*60)

    # Get CUDA device information
    print("\nGPU Information:")
    device = cuda.get_current_device()
    print(f" Device Name: {device.name}")
    print(f" Compute Capability: {device.compute_capability}")
    print(f" Total Memory: {device.total_memory / (1024**3):.2f} GB")
    print(f" Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
    print(f" Max Block Dimensions: {device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z}")
    print(f" Max Grid Dimensions: {device.MAX_GRID_DIM_X}, {device.MAX_GRID_DIM_Y}, {device.MAX_GRID_DIM_Z}")
    print("-"*60)

    # Dimensions as defined in the test
    DimRg = 4000
    DimAz = 250
    Nra_raw = 4762 # Range dimension of RawData
    Naz_raw = 40001 # Azimuth dimension of RawData

    print(f"Output Grid Dimensions: Range={DimRg}, Azimuth={DimAz}")
    print(f"Raw Data Dimensions: Range={Nra_raw}, Azimuth={Naz_raw}")

    # Initialize arrays with correct NumPy types
    print("\nInitializing NumPy arrays...")
    try:
        xxg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        yyg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        zzg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        Index = np.zeros((DimRg, DimAz), dtype=np.int32)
        Mask = np.zeros((DimRg, DimAz), dtype=np.uint8)
        Sint = np.zeros((DimRg, DimAz), dtype=np.int32)
        RawData = np.zeros((Nra_raw, Naz_raw), dtype=np.complex64)
        wgsx1 = np.zeros(Naz_raw, dtype=np.float64)
        wgsy1 = np.zeros(Naz_raw, dtype=np.float64)
        wgsz1 = np.zeros(Naz_raw, dtype=np.float64)
        NearRange = np.float64(0.0)
        DeltaRange = np.float64(0.0)
        Lambda = np.float64(0.0)
        print("Arrays initialized.")
    except MemoryError:
        print("\nError: Not enough memory to initialize arrays.")
        print(f"Attempting to allocate approx {((DimRg*DimAz*3*8) + (DimRg*DimAz*4*2) + (Nra_raw*Naz_raw*8) + (Naz_raw*3*8)) / (1024**3):.2f} GB")
        exit()

    print("\nPHASE 1: DATA LOADING")
    data_dir = "." # Assume files in current directory, or specify path
    all_files_found = True
    required_files = [
        'Dem_x_Dbl4000x250.dat', 'Dem_y_Dbl4000x250.dat', 'Dem_z_Dbl4000x250.dat',
        'Traiett_x_Dbl40001.dat', 'Traiett_y_Dbl40001.dat', 'Traiett_z_Dbl40001.dat',
        'Index_Long_4000x250.dat', 'Mask_Byte_4000x250.dat', 'Sint_Long_4000x250.dat',
        'RawData_Cmplx_4762x40001.dat',
        'NearRangeDbl_1_elemento.dat', 'DeltaRangeDbl_1_elemento.dat', 'LambdaDbl_1_elemento.dat'
    ]

    # Verify file existence
    for fname in required_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Error: Required data file not found: {fpath}")
            all_files_found = False
    if not all_files_found:
        print("Please ensure all .dat files are in the correct directory.")
        exit()
    else:
        print("All required data files found.")

    # Load data from binary files (IDL readu style)
    try:
        print("Loading DEM X...")
        xxg1 = np.fromfile(os.path.join(data_dir, 'Dem_x_Dbl4000x250.dat'), dtype=np.float64).reshape((DimRg, DimAz), order='F')
        print("Loading DEM Y...")
        yyg1 = np.fromfile(os.path.join(data_dir, 'Dem_y_Dbl4000x250.dat'), dtype=np.float64).reshape((DimRg, DimAz), order='F')
        print("Loading DEM Z...")
        zzg1 = np.fromfile(os.path.join(data_dir, 'Dem_z_Dbl4000x250.dat'), dtype=np.float64).reshape((DimRg, DimAz), order='F')

        print("Loading Trajectory X...")
        wgsx1 = np.fromfile(os.path.join(data_dir, 'Traiett_x_Dbl40001.dat'), dtype=np.float64)
        print("Loading Trajectory Y...")
        wgsy1 = np.fromfile(os.path.join(data_dir, 'Traiett_y_Dbl40001.dat'), dtype=np.float64)
        print("Loading Trajectory Z...")
        wgsz1 = np.fromfile(os.path.join(data_dir, 'Traiett_z_Dbl40001.dat'), dtype=np.float64)

        print("Loading Index...")
        Index = np.fromfile(os.path.join(data_dir, 'Index_Long_4000x250.dat'), dtype=np.int32).reshape((DimRg, DimAz), order='F')
        print("Loading Mask...")
        Mask = np.fromfile(os.path.join(data_dir, 'Mask_Byte_4000x250.dat'), dtype=np.uint8).reshape((DimRg, DimAz), order='F')
        print("Loading Sint...")
        Sint = np.fromfile(os.path.join(data_dir, 'Sint_Long_4000x250.dat'), dtype=np.int32).reshape((DimRg, DimAz), order='F')

        print("Loading RawData (assuming interleaved float32 pairs)...")
        raw_data_path = os.path.join(data_dir, 'RawData_Cmplx_4762x40001.dat')
        # Read as float32
        data_f32 = np.fromfile(raw_data_path, dtype=np.float32)
        expected_len = 2 * Nra_raw * Naz_raw
        if data_f32.size == expected_len:
            # Reconstruct complex64 from interleaved pairs (real1, imag1, real2, imag2, ...)
            real_part = data_f32[0::2].reshape((Nra_raw, Naz_raw), order='F')
            imag_part = data_f32[1::2].reshape((Nra_raw, Naz_raw), order='F')
            RawData = real_part + 1j * imag_part # Result is complex64
            print(f"RawData loaded successfully. Shape: {RawData.shape}, Dtype: {RawData.dtype}")
        else:
            # Try split format (all reals, then all imaginaries)
            print("Interleaved format size mismatch. Trying split format (all reals then all imaginaries)...")
            if data_f32.size == expected_len: # Reuse expected_len because it's the same total number of float32
                half_len = expected_len // 2
                real_part = data_f32[:half_len].reshape((Nra_raw, Naz_raw), order='F')
                imag_part = data_f32[half_len:].reshape((Nra_raw, Naz_raw), order='F')
                RawData = real_part + 1j * imag_part # Result is complex64
                print(f"RawData loaded successfully using split format. Shape: {RawData.shape}, Dtype: {RawData.dtype}")
            else:
                raise IOError(f"RawData file size ({data_f32.size} floats) does not match expected size for interleaved or split format ({expected_len} floats).")

        print("Loading NearRange...")
        NearRange = np.fromfile(os.path.join(data_dir, 'NearRangeDbl_1_elemento.dat'), dtype=np.float64)[0]
        print("Loading DeltaRange...")
        DeltaRange = np.fromfile(os.path.join(data_dir, 'DeltaRangeDbl_1_elemento.dat'), dtype=np.float64)[0]
        print("Loading Lambda...")
        Lambda = np.fromfile(os.path.join(data_dir, 'LambdaDbl_1_elemento.dat'), dtype=np.float64)[0]

        print("\nAll data loaded successfully.")
        print(f" NearRange = {NearRange:.6f}")
        print(f" DeltaRange = {DeltaRange:.6f}")
        print(f" Lambda = {Lambda:.6f}")

    except Exception as e:
        print(f"\nError loading data: {e}")
        exit()

    print("\nPHASE 2: STARTING GPU FOCUSING PROCESS")
    # Choose the implementation to use
    use_cupy = False  # Set to True to use CuPy implementation instead
    
    if use_cupy:
        SlcGpu = FocalizzatoreBpCuPy(xxg1, yyg1, zzg1, Mask, Index, wgsx1, wgsy1, wgsz1, RawData, NearRange, DeltaRange, Lambda, Sint)
    else:
        SlcGpu = FocalizzatoreBpGPU(xxg1, yyg1, zzg1, Mask, Index, wgsx1, wgsy1, wgsz1, RawData, NearRange, DeltaRange, Lambda, Sint)

    print("\nPHASE 3: SAVING RESULTS")
    try:
        output_filename_base = 'SlcGpu_RTX4090'

        # Save in NumPy native format (recommended for Python reuse)
        output_filename_npy = output_filename_base + '.npy'
        print(f"Saving in NumPy native format (.npy) to: {output_filename_npy}...")
        np.save(output_filename_npy, SlcGpu)
        print("NumPy .npy format saved.")

        # Save in split float32 format (real_part, imag_part)
        output_filename_split = output_filename_base + '_SplitF32.dat'
        print(f"Saving in split float32 format to: {output_filename_split}...")
        real_part_f32 = SlcGpu.real.astype(np.float32)
        imag_part_f32 = SlcGpu.imag.astype(np.float32)
        with open(output_filename_split, 'wb') as f:
            # Write all real part first, then all imaginary part
            real_part_f32.flatten(order='F').tofile(f)
            imag_part_f32.flatten(order='F').tofile(f)
        print("Split float32 format saved.")

    except Exception as e:
        print(f"\nError saving results: {e}")

    # Print total time
    elapsed_total = time.time() - start_time_total
    print(f"\nTotal execution time: {elapsed_total:.2f} seconds ({elapsed_total/60.0:.2f} minutes)")
    print("="*60)

    # Optional: Display amplitude image
    try:
        import matplotlib.pyplot as plt
        print("\nGenerating amplitude image preview...")
        plt.figure(figsize=(10, 8))

        # Calculate amplitude
        amplitude = np.abs(SlcGpu)
        valid_amplitude = amplitude[np.isfinite(amplitude)]
        if valid_amplitude.size == 0:
            print("Warning: Amplitude array contains no valid finite values. Cannot display.")
        else:
            mean_amp = np.mean(valid_amplitude)
            vmax_val = 15 * mean_amp if mean_amp > 1e-9 else 1.0
            vmin_val = np.percentile(valid_amplitude[valid_amplitude > 1e-9] if np.any(valid_amplitude > 1e-9) else valid_amplitude, 1)

            plt.imshow(amplitude, cmap='gray', vmin=vmin_val, vmax=vmax_val, aspect='auto')
            plt.colorbar(label='Amplitude')
            plt.title(f'SAR Image Amplitude (GPU Output)\nvmax ≈ 15 * mean = {vmax_val:.2e}')
            plt.xlabel("Azimuth Samples")
            plt.ylabel("Range Samples")
            plt.tight_layout()
            img_filename = 'SlcGpu_RTX4090_Amplitude.png'
            plt.savefig(img_filename, dpi=150)
            print(f"Image preview saved to {img_filename}")
    except ImportError:
        print("\nMatplotlib not installed. Skipping image preview generation.")
    except Exception as e:
        print(f"\nCould not create image preview due to an error: {e}")