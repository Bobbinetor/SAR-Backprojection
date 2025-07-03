#!/usr/bin/env python3

import numpy as np
import cupy as cp
import time
import os
import h5py
import pickle
import json
from datetime import datetime, timedelta
import psutil
import gc

# ###########################################################################
# # Memory Utility Functions 
# ###########################################################################
def get_memory_info():
    """Return memory information for both system RAM and GPU."""
    # System RAM
    ram = psutil.virtual_memory()
    ram_total = ram.total / (1024**3)  # GB
    ram_used = ram.used / (1024**3)    # GB
    ram_free = ram.available / (1024**3)  # GB
    
    # GPU memory
    try:
        gpu = cp.cuda.Device()
        gpu_total = gpu.mem_info[1] / (1024**3)  # GB
        gpu_free = gpu.mem_info[0] / (1024**3)   # GB
        gpu_used = gpu_total - gpu_free
    except Exception as e:
        gpu_total = gpu_free = gpu_used = 0
        print(f"Error getting GPU info: {e}")
    
    return {
        "ram_total": ram_total,
        "ram_used": ram_used,
        "ram_free": ram_free,
        "gpu_total": gpu_total,
        "gpu_used": gpu_used,
        "gpu_free": gpu_free
    }

def print_memory_status(prefix=""):
    """Print current memory usage."""
    mem = get_memory_info()
    print(f"{prefix} Memory Status:")
    print(f"  RAM: {mem['ram_used']:.1f}/{mem['ram_total']:.1f} GB ({mem['ram_used']/mem['ram_total']*100:.1f}%)")
    print(f"  GPU: {mem['gpu_used']:.1f}/{mem['gpu_total']:.1f} GB ({mem['gpu_used']/mem['gpu_total']*100:.1f}%)")

def free_memory():
    """Aggressively free memory."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

# ###########################################################################
# # Improved CuPy Kernel for SAR Processing
# ###########################################################################
process_point_kernel = cp.RawKernel(r'''
extern "C" __global__
void process_point_kernel(
    const double* xxg1_flat, const double* yyg1_flat, const double* zzg1_flat,
    const unsigned char* mask_flat, const int* index_flat, const int* Sint_flat, // index_flat is relative to chunk start
    const double* wgsx, const double* wgsy, const double* wgsz, // These are chunked trajectory data
    const float* Raw_Data_real, const float* Raw_Data_imag, // These are chunked raw data
    const double near_range, const double dra, const double lambda_val,
    const int Nra, // Full range dimension (Nra_raw)
    const int chunk_az_width, // Width of the current raw data/trajectory chunk (raw_az_size)
    const int global_total_az_width, // Total azimuth samples in the full dataset (Naz_raw)
    const int chunk_global_az_offset, // Global starting azimuth index of the current chunk (min_az_index)
    double* focused_real, double* focused_imag,
    const int total_points)
{
    // Get thread ID
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Check if thread ID is within bounds
    if (ii >= total_points)
        return;
        
    const double C_F64 = 2.9979246e8;
    const double ALFA_F64 = 3.335987619777693e11;
    const double PI_F64 = 3.141592741012573242;
    const double BETA_F64 = 4.0 * PI_F64 * ALFA_F64 / (C_F64 * C_F64);

    focused_real[ii] = 0.0;
    focused_imag[ii] = 0.0;

    if (mask_flat[ii] != 1)
        return;
        
    int sint = Sint_flat[ii];
    if (sint == -9999)
        return;
        
    int jj_size = int(ceil(sint / 2.0)) * 2 + 1;
    
    double back_projected_sum_real = 0.0;
    double back_projected_sum_imag = 0.0;
    int n_valid_contributors = 0;
    
    for (int j = 0; j < jj_size; j++) {
        // index_flat[ii] is the center of aperture, relative to chunk_global_az_offset
        // Calculate index1_relative_to_chunk: position within the current chunk's azimuth dimension
        int index1_relative_to_chunk = index_flat[ii] + j - int(ceil(abs((double)sint) / 2.0));
        
        // Calculate the corresponding global azimuth index
        int index1_global = chunk_global_az_offset + index1_relative_to_chunk;
        
        // Boundary check using GLOBAL azimuth index and GLOBAL total width
        if (index1_global >= global_total_az_width - 1 || index1_global <= 0)
            continue;
            
        // Additional check: ensure the relative index is within the bounds of the current chunk
        // This should hold true if chunking logic correctly covers the aperture.
        if (index1_relative_to_chunk < 0 || index1_relative_to_chunk >= chunk_az_width)
            continue;

        double xcomp = xxg1_flat[ii] - wgsx[index1_relative_to_chunk];
        double ycomp = yyg1_flat[ii] - wgsy[index1_relative_to_chunk];
        double zcomp = zzg1_flat[ii] - wgsz[index1_relative_to_chunk];

        double dist_patch2 = xcomp*xcomp + ycomp*ycomp + zcomp*zcomp;
        double dist_patch = sqrt(dist_patch2);
        
        int ind_line_ra = int(round((dist_patch - near_range) / dra));
        // Skip if range index is out of bounds
        if (ind_line_ra < 0 || ind_line_ra >= Nra)
            continue;

        // Correct quadratic phase term (factor 1/2)
        double phase_term = (-4.0 * PI_F64 / lambda_val * dist_patch) + (BETA_F64 * dist_patch2);
        
        double cos_phase = cos(-phase_term);
        double sin_phase = sin(-phase_term);
        
        // Access raw data using relative index and chunk_az_width
        float raw_real = Raw_Data_real[ind_line_ra * chunk_az_width + index1_relative_to_chunk];
        float raw_imag = Raw_Data_imag[ind_line_ra * chunk_az_width + index1_relative_to_chunk];
        
        double term_real = dist_patch2 * (raw_real * cos_phase - raw_imag * sin_phase);
        double term_imag = dist_patch2 * (raw_real * sin_phase + raw_imag * cos_phase);
        
        back_projected_sum_real += term_real;
        back_projected_sum_imag += term_imag;
        n_valid_contributors += 1;
    }
    
    if (n_valid_contributors > 0) {
        focused_real[ii] = back_projected_sum_real / n_valid_contributors;
        focused_imag[ii] = back_projected_sum_imag / n_valid_contributors;
    }
}
''', 'process_point_kernel')

# ###########################################################################
# # Super-optimized SAR Focuser for Massive Datasets
# ###########################################################################
class SARSuperOptimizedFocuser:
    def __init__(self, output_file='SlcCuPy_SuperOptimized.h5', checkpoint_base_name='SlcCuPy_SuperOptimized_checkpoint'):
        self.output_file = output_file
        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_base_name}.json")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.config = {
            'batch_size_azimuth': 2000,   # Number of azimuth columns to process in each batch
            'batch_size_points': 1000000, # Max points to process in GPU in one batch
            'raw_chunk_size': 500,        # Number of range lines to load at once
            'max_gpu_memory': 24.0,       # Target max GPU memory usage in GB (conservative for 24GB cards)
            'threads_per_block': 256,     # CUDA threads per block
            'azimuth_overlap': 100,       # Overlap between azimuth batches
        }
        
        # Runtime state
        self.processed_batches = 0
        self.start_time = None
        self.checkpoint_data = {}
        self.raw_data_mmap = None
        
        # Initialize GPU
        self._init_gpu()
    
    def _init_gpu(self):
        """Setup GPU and print information."""
        try:
            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            device_name = props['name'].decode('utf-8')  # Correctly get device name
            print("\nGPU Information:")
            print(f" Device Name: {device_name}")
            print(f" Compute Capability: {props['major']}.{props['minor']}")  # Correctly get compute capability
            print(f" Total Memory: {device.mem_info[1] / (1024**3):.2f} GB")
            print(f" Free Memory: {device.mem_info[0] / (1024**3):.2f} GB")
            
            # Auto-adjust config based on GPU capabilities
            gpu_mem = device.mem_info[1] / (1024**3)
            if gpu_mem > 20:  # Adjust for high-end GPUs like RTX 4090
                self.config['max_gpu_memory'] = gpu_mem * 0.8  # Use up to 80% of available GPU memory
                self.config['batch_size_points'] = 2000000  # Increase batch size for high-end GPUs
            
            print(f"Configured to use up to {self.config['max_gpu_memory']:.1f} GB of GPU memory")
            print("-"*60)
        except Exception as e:
            print(f"Error initializing GPU: {e}")
    
    def _validate_raw_data_format_and_get_dims(self, data_dir, filename, expected_Nra, expected_Naz):
        """
        Validates RawData file size for interleaved complex64 format and returns actual dimensions.
        Does NOT load the entire file into RAM.
        Returns: actual_Nra, actual_Naz
        """
        print(f"Validating RawData format and dimensions for {filename} in directory {data_dir}...")
        raw_data_path = os.path.join(data_dir, filename)

        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

        file_size_bytes = os.path.getsize(raw_data_path)
        bytes_per_complex64 = np.dtype(np.complex64).itemsize # Should be 8
        
        total_complex_elements_from_size = file_size_bytes // bytes_per_complex64
        
        print(f"RawData file size: {file_size_bytes} bytes. Implies {total_complex_elements_from_size} complex64 elements.")

        if file_size_bytes % bytes_per_complex64 != 0:
            raise ValueError(f"RawData file size ({file_size_bytes}) is not a multiple of complex64 size ({bytes_per_complex64} bytes). "
                             f"Cannot safely memmap as complex64. The file might not be purely interleaved complex64 data or might be truncated.")

        actual_Nra = expected_Nra
        actual_Naz = expected_Naz

        if total_complex_elements_from_size != expected_Nra * expected_Naz:
            print(f"Warning: Total complex elements from file size ({total_complex_elements_from_size}) "
                  f"does not match expected dimensions {expected_Nra}x{expected_Naz} ({expected_Nra * expected_Naz}). Reconciling...")
            if expected_Nra > 0 and total_complex_elements_from_size % expected_Nra == 0:
                actual_Naz = total_complex_elements_from_size // expected_Nra
                print(f"Adjusting Naz based on Nra={expected_Nra}: New dimensions {actual_Nra}x{actual_Naz}")
            elif expected_Naz > 0 and total_complex_elements_from_size % expected_Naz == 0:
                actual_Nra = total_complex_elements_from_size // expected_Naz
                print(f"Adjusting Nra based on Naz={expected_Naz}: New dimensions {actual_Nra}x{actual_Naz}")
            else:
                raise ValueError(f"Could not reconcile file size ({total_complex_elements_from_size} complex elements) "
                                 f"with expected dimensions {expected_Nra}x{expected_Naz} to form a valid 2D array for memmap.")
        
        print(f"Validation successful. Using dimensions for memmap: {actual_Nra} x {actual_Naz}")
        return actual_Nra, actual_Naz

    def load_checkpoint(self):
        """Load processing checkpoint if exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoint_data = json.load(f)
                print(f"Loaded checkpoint from {self.checkpoint_file}: {self.checkpoint_data}")
                self.processed_batches = self.checkpoint_data.get('processed_batches', 0)
                elapsed_seconds_before_resume = self.checkpoint_data.get('total_elapsed_time_seconds', 0)
                self.start_time = time.time() - elapsed_seconds_before_resume 
                return True
            except Exception as e:
                print(f"Error loading checkpoint from {self.checkpoint_file}: {e}")
                self.checkpoint_data = {}
                self.start_time = time.time()
        else:
            print(f"No checkpoint file found at {self.checkpoint_file}. Starting fresh.")
            self.start_time = time.time()
        return False

    def save_checkpoint(self, current_az_start, current_az_end):
        """Save the current processing state to a checkpoint file."""
        if self.start_time is None:
            self.start_time = time.time() 
            
        current_elapsed_seconds = time.time() - self.start_time
        
        checkpoint_content = {
            'last_processed_az_start': current_az_start,
            'last_processed_az_end': current_az_end,
            'processed_batches': self.processed_batches,
            'total_elapsed_time_seconds': current_elapsed_seconds,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_content, f, indent=4)
            print(f"Checkpoint saved to {self.checkpoint_file} (Azimuth: {current_az_start}-{current_az_end})")
        except Exception as e:
            print(f"Error saving checkpoint to {self.checkpoint_file}: {e}")

    def process_dataset(self, data_dir='.'):
        """Process the entire large SAR dataset in optimized batches."""        
        resumed = self.load_checkpoint()
        if not resumed and self.start_time is None:
             self.start_time = time.time()
        total_start_time_for_this_run = time.time()

        print("="*80)
        print(" SUPER-OPTIMIZED SAR BACK-PROJECTION PROCESSOR")
        print(" Designed for Very Large Datasets")
        print("="*80)
        
        # Get file paths
        dem_x_path = os.path.join(data_dir, 'Dem_x_Dbl_4000x32000.dat')
        dem_y_path = os.path.join(data_dir, 'Dem_y_Dbl_4000x32000.dat')
        dem_z_path = os.path.join(data_dir, 'Dem_z_Dbl_4000x32000.dat')
        index_path = os.path.join(data_dir, 'Index_Long_4000x32000.dat')
        mask_path = os.path.join(data_dir, 'Mask_Byte_4000x32000.dat')
        sint_path = os.path.join(data_dir, 'Sint_Long_4000x32000.dat')
        
        # Load scalar parameters
        print("\nLoading processing parameters...")
        NearRange = np.fromfile(os.path.join(data_dir, 'NearRangeDbl_1_elemento.dat'), dtype=np.float64)[0]
        DeltaRange = np.fromfile(os.path.join(data_dir, 'DeltaRangeDbl_1_elemento.dat'), dtype=np.float64)[0]
        Lambda = np.fromfile(os.path.join(data_dir, 'LambdaDbl_1_elemento.dat'), dtype=np.float64)[0]
        
        print(f" NearRange = {NearRange:.6f}")
        print(f" DeltaRange = {DeltaRange:.6f}")
        print(f" Lambda = {Lambda:.6f}")
        
        DimRg = 4000
        DimAz = 32000
        Nra_raw_expected = 4762
        Naz_raw_expected = 403484
        
        print(f"\nDataset dimensions (expected):")
        print(f" DEM grid: {DimRg} × {DimAz} ({DimRg*DimAz} points)")
        print(f" Raw data (expected): {Nra_raw_expected} × {Naz_raw_expected}")
        
        print("\nLoading trajectory data...")
        wgsx = np.fromfile(os.path.join(data_dir, 'Traiett_x_Dbl403485.dat'), dtype=np.float64)
        wgsy = np.fromfile(os.path.join(data_dir, 'Traiett_y_Dbl403485.dat'), dtype=np.float64)
        wgsz = np.fromfile(os.path.join(data_dir, 'Traiett_z_Dbl403485.dat'), dtype=np.float64)
        
        print(f" Trajectory length: {len(wgsx)} points")
        
        print(f"\nPreparing output file: {self.output_file}")
        h5_mode = 'a' if resumed else 'w'
        with h5py.File(self.output_file, h5_mode) as h5f:
            if not resumed:
                slc_dataset = h5f.create_dataset(
                    'slc', 
                    shape=(DimRg, DimAz),
                    dtype=np.complex128,
                    chunks=(DimRg, min(1000, DimAz)),
                    compression='gzip',
                    compression_opts=4
                )
                h5f.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                h5f.attrs['dim_range'] = DimRg
                h5f.attrs['dim_azimuth'] = DimAz
                h5f.attrs['near_range'] = NearRange
                h5f.attrs['delta_range'] = DeltaRange
                h5f.attrs['lambda'] = Lambda
                h5f.attrs['fortran_order'] = True
            else:
                slc_dataset = h5f['slc']
            
            print("\nValidating Raw Data format and dimensions for memmap...")
            try:
                Nra_raw, Naz_raw = self._validate_raw_data_format_and_get_dims(
                    data_dir, 
                    'RawData_Cmplx_4762x403484.dat', 
                    Nra_raw_expected, 
                    Naz_raw_expected
                )
                print(f"Actual RawData dimensions for memmap: {Nra_raw} x {Naz_raw}")
            except Exception as e:
                print(f"Critical error validating raw data for memmap: {e}")
                raise

            print("Creating memory map for RawData...")
            raw_data_path_full = os.path.join(data_dir, 'RawData_Cmplx_4762x403484.dat')
            try:
                self.raw_data_mmap = np.memmap(
                    raw_data_path_full,
                    dtype=np.complex64,
                    mode='r',
                    shape=(Nra_raw, Naz_raw),
                    order='F'
                )
                print(f"Successfully created memory map for RawData with shape {self.raw_data_mmap.shape}")
            except Exception as e:
                print(f"Critical error creating memory map for raw data: {e}")
                raise

            traj_len = len(wgsx)
            if traj_len > Naz_raw:
                print(f" Note: Trajectory ({traj_len} points) is longer than actual raw data azimuth ({Naz_raw} samples).")
                print(f" Truncating trajectory to match raw data azimuth length ({Naz_raw} points).")
                wgsx = wgsx[:Naz_raw]
                wgsy = wgsy[:Naz_raw]
                wgsz = wgsz[:Naz_raw]
            elif traj_len < Naz_raw:
                print(f" Warning: Trajectory ({traj_len} points) is shorter than actual raw data azimuth ({Naz_raw} samples).")
                print(f" Logically truncating raw data azimuth to match trajectory length ({traj_len} points) for processing.")
                Naz_raw = traj_len
            print(f"Final consistent RawData azimuth samples for processing: {Naz_raw}, Trajectory points: {len(wgsx)}")

            batch_az_size = self.config['batch_size_azimuth']
            start_az = 0
            resumed_from_batch_idx = 0

            if resumed and 'last_processed_az_end' in self.checkpoint_data:
                start_az = self.checkpoint_data['last_processed_az_end'] - self.config['azimuth_overlap']
                start_az = max(0, start_az)
                resumed_from_batch_idx = start_az // batch_az_size 
                print(f"Resuming. Effective start azimuth: {start_az}, corresponding to batch index ~{resumed_from_batch_idx}")
            
            num_total_dem_batches = (DimAz + batch_az_size - 1) // batch_az_size
            
            print(f"\nProcessing DEM in {num_total_dem_batches} azimuth batches, each with ~{batch_az_size} columns")
            print("="*80)
            
            for batch_idx in range(num_total_dem_batches):
                current_dem_az_start = batch_idx * batch_az_size
                current_dem_az_end = min(current_dem_az_start + batch_az_size, DimAz)

                if resumed and current_dem_az_end <= start_az:
                    print(f"Skipping already processed DEM batch {batch_idx+1}/{num_total_dem_batches} (az: {current_dem_az_start}-{current_dem_az_end}) as per checkpoint.")
                    continue
                
                batch_run_start_time = time.time()
                print(f"\nDEM Batch {batch_idx+1}/{num_total_dem_batches} - Processing DEM azimuth columns {current_dem_az_start}-{current_dem_az_end}...")
                
                print(" Loading DEM data for this batch (Fortran order)...")
                xxg1_batch = self._load_binary_subset(dem_x_path, DimRg, DimAz, 0, DimRg, current_dem_az_start, current_dem_az_end, np.float64)
                yyg1_batch = self._load_binary_subset(dem_y_path, DimRg, DimAz, 0, DimRg, current_dem_az_start, current_dem_az_end, np.float64)
                zzg1_batch = self._load_binary_subset(dem_z_path, DimRg, DimAz, 0, DimRg, current_dem_az_start, current_dem_az_end, np.float64)
                
                print(" Loading mask and indices for this batch (Fortran order)...")
                index_batch = self._load_binary_subset(index_path, DimRg, DimAz, 0, DimRg, current_dem_az_start, current_dem_az_end, np.int32)
                mask_batch = self._load_binary_subset(mask_path, DimRg, DimAz, 0, DimRg, current_dem_az_start, current_dem_az_end, np.uint8)
                sint_batch = self._load_binary_subset(sint_path, DimRg, DimAz, 0, DimRg, current_dem_az_start, current_dem_az_end, np.int32)
                
                no_valid_indices = np.where(sint_batch == -9999)
                if len(no_valid_indices[0]) > 0:
                    print(f" Masking {len(no_valid_indices[0])} points due to Sint == -9999.")
                    mask_batch[no_valid_indices] = 0
                
                valid_count = np.sum(mask_batch == 1)
                batch_points = xxg1_batch.size
                
                print(f" Batch contains {valid_count} valid points out of {batch_points} total points")
                
                slc_batch_output = np.zeros((DimRg, current_dem_az_end - current_dem_az_start), dtype=np.complex128)
                
                xxg1_flat = xxg1_batch.flatten(order='F')
                yyg1_flat = yyg1_batch.flatten(order='F')
                zzg1_flat = zzg1_batch.flatten(order='F')
                mask_flat = mask_batch.flatten(order='F')
                index_flat = index_batch.flatten(order='F')
                sint_flat = sint_batch.flatten(order='F')
                
                del xxg1_batch, yyg1_batch, zzg1_batch, index_batch, mask_batch, sint_batch
                
                sub_batch_size = self.config['batch_size_points']
                sub_batches = (batch_points + sub_batch_size - 1) // sub_batch_size
                
                print(f" Processing in {sub_batches} point sub-batches, each with up to {sub_batch_size} points")
                
                focused_real = np.zeros(batch_points, dtype=np.float64)
                focused_imag = np.zeros(batch_points, dtype=np.float64)
                
                half_aperture = np.ceil(np.abs(sint_flat) / 2.0).astype(np.int32)
                
                valid_points_mask_for_aperture = (mask_flat == 1) & (sint_flat != -9999)
                
                if np.any(valid_points_mask_for_aperture):
                    min_potential_az_access = np.min(index_flat[valid_points_mask_for_aperture] - half_aperture[valid_points_mask_for_aperture])
                    max_potential_az_access = np.max(index_flat[valid_points_mask_for_aperture] + half_aperture[valid_points_mask_for_aperture])
                else:
                    min_potential_az_access = 0
                    max_potential_az_access = 0

                min_az_index = int(np.floor(min_potential_az_access))
                # Expand window with overlap margin to capture edge contributions
                min_az_index = max(0, min_az_index - self.config['azimuth_overlap'])
                
                max_az_index = int(np.ceil(max_potential_az_access)) + 1
                # Expand window with overlap margin
                max_az_index = min(Naz_raw, max_az_index + self.config['azimuth_overlap'])

                if max_az_index <= min_az_index:
                    max_az_index = min_az_index + 1
                    if max_az_index > Naz_raw:
                        max_az_index = Naz_raw
                        if min_az_index >= Naz_raw and Naz_raw > 0:
                            min_az_index = Naz_raw - 1
                        elif Naz_raw == 0:
                            min_az_index = 0

                print(f" Calculated required raw data global azimuth range for this batch: [{min_az_index}, {max_az_index}) for full raw data.")
                print(f" This corresponds to a chunk of width {max_az_index - min_az_index} samples.")

                raw_data_slice_mmap_complex = self.raw_data_mmap[:, min_az_index:max_az_index]

                raw_data_real_cpu = np.ascontiguousarray(raw_data_slice_mmap_complex.real.astype(np.float32))
                raw_data_imag_cpu = np.ascontiguousarray(raw_data_slice_mmap_complex.imag.astype(np.float32))
                
                print(" Transferring raw data chunk to GPU...")
                d_raw_data_real = cp.asarray(raw_data_real_cpu)
                d_raw_data_imag = cp.asarray(raw_data_imag_cpu)
                
                del raw_data_real_cpu, raw_data_imag_cpu, raw_data_slice_mmap_complex
                print_memory_status(" After loading raw data chunk to GPU:")

                print(f" Preparing and transferring trajectory chunk ({min_az_index}:{max_az_index}) to GPU...")
                wgsx_chunk_cpu = wgsx[min_az_index:max_az_index].copy()
                wgsy_chunk_cpu = wgsy[min_az_index:max_az_index].copy()
                wgsz_chunk_cpu = wgsz[min_az_index:max_az_index].copy()

                d_wgsx_batch = cp.asarray(wgsx_chunk_cpu)
                d_wgsy_batch = cp.asarray(wgsy_chunk_cpu)
                d_wgsz_batch = cp.asarray(wgsz_chunk_cpu)

                del wgsx_chunk_cpu, wgsy_chunk_cpu, wgsz_chunk_cpu
                
                for sb in range(sub_batches):
                    sub_start = sb * sub_batch_size
                    sub_end = min(sub_start + sub_batch_size, batch_points)
                    sub_size = sub_end - sub_start
                    
                    print(f" Processing sub-batch {sb+1}/{sub_batches} (points {sub_start}-{sub_end})...")
                    
                    d_xxg1_sub = cp.asarray(xxg1_flat[sub_start:sub_end])
                    d_yyg1_sub = cp.asarray(yyg1_flat[sub_start:sub_end])
                    d_zzg1_sub = cp.asarray(zzg1_flat[sub_start:sub_end])
                    d_mask_sub = cp.asarray(mask_flat[sub_start:sub_end])
                    d_index_sub = cp.asarray(index_flat[sub_start:sub_end] - min_az_index)
                    d_sint_sub = cp.asarray(sint_flat[sub_start:sub_end])
                    
                    d_focused_real = cp.zeros(sub_size, dtype=np.float64)
                    d_focused_imag = cp.zeros(sub_size, dtype=np.float64)
                    
                    threads_per_block = self.config['threads_per_block']
                    blocks_per_grid = (sub_size + threads_per_block - 1) // threads_per_block
                    
                    process_point_kernel(
                        (blocks_per_grid,), (threads_per_block,),
                        (d_xxg1_sub, d_yyg1_sub, d_zzg1_sub,
                        d_mask_sub, d_index_sub, d_sint_sub,
                        d_wgsx_batch, d_wgsy_batch, d_wgsz_batch,
                        d_raw_data_real, d_raw_data_imag,
                        np.float64(NearRange), np.float64(DeltaRange), np.float64(Lambda),
                        np.int32(Nra_raw),
                        np.int32(max_az_index - min_az_index),
                        np.int32(Naz_raw),
                        np.int32(min_az_index),
                        d_focused_real, d_focused_imag,
                        np.int32(sub_size))
                    )
                    
                    focused_real[sub_start:sub_end] = cp.asnumpy(d_focused_real)
                    focused_imag[sub_start:sub_end] = cp.asnumpy(d_focused_imag)
                    
                    del d_xxg1_sub, d_yyg1_sub, d_zzg1_sub
                    del d_mask_sub, d_index_sub, d_sint_sub
                    del d_focused_real, d_focused_imag
                    cp.get_default_memory_pool().free_all_blocks()
                
                del d_raw_data_real, d_raw_data_imag
                del d_wgsx_batch, d_wgsy_batch, d_wgsz_batch
                cp.get_default_memory_pool().free_all_blocks()
                
                focused_cplx = focused_real + 1j * focused_imag
                slc_batch_output = np.reshape(focused_cplx, (DimRg, current_dem_az_end - current_dem_az_start), order='F')
                
                if not slc_batch_output.flags.f_contiguous:
                    print(" Warning: Converting result to F-contiguous order for consistency")
                    slc_batch_output = np.asfortranarray(slc_batch_output)
                
                print(" Writing results to output file...")
                slc_dataset[:, current_dem_az_start:current_dem_az_end] = slc_batch_output
                h5f.flush()
                
                self.processed_batches += 1
                self.save_checkpoint(current_dem_az_start, current_dem_az_end)
                
                h5f.attrs['processed_azimuth'] = current_dem_az_end
                h5f.attrs['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                h5f.attrs['processed_percentage'] = float(current_dem_az_end) / DimAz * 100
                
                if self.processed_batches % 5 == 0 or (current_dem_az_end / DimAz * 100) % 25 < (batch_az_size / DimAz * 100):
                    print("\nGenerating intermediate preview image...")
                    self.generate_preview()
                
                batch_time_taken = time.time() - batch_run_start_time
                total_elapsed_since_run_start = time.time() - total_start_time_for_this_run
                overall_elapsed_for_eta = time.time() - self.start_time 
                
                print(f"\nDEM Batch {batch_idx+1}/{num_total_dem_batches} completed in {batch_time_taken:.1f} seconds")
                print(f"Progress: {float(current_dem_az_end) / DimAz * 100:.1f}% complete")
                print(f"Estimated remaining time: {self.estimate_remaining_time(current_dem_az_end, DimAz, overall_elapsed_for_eta)}")
                print(f"Total elapsed time for this run: {timedelta(seconds=int(total_elapsed_since_run_start))}")
                print(f"Overall elapsed time (including previous runs if any): {timedelta(seconds=int(overall_elapsed_for_eta))}")
                
                del slc_batch_output, focused_cplx, focused_real, focused_imag
                del xxg1_flat, yyg1_flat, zzg1_flat
                del mask_flat, index_flat, sint_flat
                free_memory()
                print_memory_status(" After batch completion:")
                print("="*80)
            
            if hasattr(self, 'raw_data_mmap'):
                if hasattr(self.raw_data_mmap, '_mmap') and self.raw_data_mmap._mmap is not None:
                    try:
                        self.raw_data_mmap._mmap.close()
                    except Exception as e:
                        print(f"Error closing memmap file handle: {e}")
                del self.raw_data_mmap
                print("Closed and deleted raw_data_mmap.")
            
            cp.get_default_memory_pool().free_all_blocks()
            
            h5f.attrs['completion_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            h5f.attrs['processing_time_seconds'] = time.time() - total_start_time_for_this_run
            
            print("\nProcessing completed successfully!")
            print(f"Total processing time: {timedelta(seconds=int(time.time() - total_start_time_for_this_run))}")
            print(f"Results saved to: {self.output_file}")
    
    def estimate_remaining_time(self, current_progress_units, total_units, elapsed_time_seconds):
        """Estimate remaining processing time."""
        if current_progress_units == 0 or elapsed_time_seconds == 0:
            return "Calculating..."
        
        rate = current_progress_units / elapsed_time_seconds
        remaining_units = total_units - current_progress_units
        if rate == 0:
            return "Infinite (rate is zero)"
        remaining_seconds = remaining_units / rate
        return str(timedelta(seconds=int(remaining_seconds)))

    def generate_preview(self, preview_size=1000):
        """Generate a preview of the processed SAR image."""
        import matplotlib.pyplot as plt
        
        try:
            with h5py.File(self.output_file, 'r') as h5f:
                slc = h5f['slc']
                dim_rg, dim_az = slc.shape
                
                if dim_az > preview_size:
                    if 'processed_azimuth' in h5f.attrs:
                        processed_az = h5f.attrs['processed_azimuth']
                        if processed_az > preview_size:
                            start_az = max(0, min(processed_az, dim_az) // 2 - preview_size // 2)
                            end_az = min(dim_az, start_az + preview_size)
                        else:
                            start_az = 0
                            end_az = min(dim_az, processed_az)
                    else:
                        start_az = max(0, dim_az // 2 - preview_size // 2)
                        end_az = min(dim_az, start_az + preview_size)
                    
                    print(f"Generating preview for columns {start_az}-{end_az}...")
                    subset = slc[:, start_az:end_az]
                else:
                    subset = slc[:, :]
                
                amplitude = np.abs(subset)
                
                invalid_count = np.sum(~np.isfinite(amplitude))
                if invalid_count > 0:
                    print(f"Warning: {invalid_count} invalid (NaN/Inf) values detected in preview region")
                    amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)
                
                valid_amplitude = amplitude[np.isfinite(amplitude)]
                
                if valid_amplitude.size == 0:
                    print("Error: No valid data for preview generation")
                    return
                
                plt.figure(figsize=(12, 10))
                
                mean_amp = np.mean(valid_amplitude)
                median_amp = np.median(valid_amplitude)
                max_amp = np.max(valid_amplitude)
                
                print(f"Amplitude statistics: mean={mean_amp:.2e}, median={median_amp:.2e}, max={max_amp:.2e}")
                
                vmax_val = min(15 * median_amp, max_amp)
                if np.any(valid_amplitude > 0):
                    non_zero = valid_amplitude[valid_amplitude > 0]
                    if len(non_zero) > 0:
                        vmin_val = np.percentile(non_zero, 1)
                    else:
                        vmin_val = 0
                else:
                    vmin_val = 0
                
                plt.imshow(amplitude, cmap='gray', vmin=vmin_val, vmax=vmax_val, aspect='auto')
                plt.colorbar(label='Amplitude')
                plt.title(f'SAR Image Amplitude (Preview)\nColumns {start_az}-{end_az} of {dim_az}')
                plt.xlabel("Azimuth Samples")
                plt.ylabel("Range Samples")
                plt.tight_layout()
                
                preview_filename = os.path.join(self.checkpoint_dir, 'SlcCuPy_SuperOptimized_Preview.png')
                plt.savefig(preview_filename, dpi=150)
                print(f"Preview saved to {preview_filename}")
                
                plt.figure(figsize=(12, 10))
                log_amplitude = np.log10(amplitude + 1e-10)
                plt.imshow(log_amplitude, cmap='viridis', aspect='auto')
                plt.colorbar(label='Log10(Amplitude)')
                plt.title(f'SAR Image Amplitude (Log Scale)\nColumns {start_az}-{end_az} of {dim_az}')
                plt.xlabel("Azimuth Samples")
                plt.ylabel("Range Samples")
                plt.tight_layout()
                
                log_preview_filename = os.path.join(self.checkpoint_dir, 'SlcCuPy_SuperOptimized_LogScale_Preview.png')
                plt.savefig(log_preview_filename, dpi=150)
                print(f"Log-scale preview saved to {log_preview_filename}")
                
                if dim_az <= 10000:
                    print("Generating full-resolution preview...")
                    try:
                        full_amplitude = np.abs(slc[:, :])
                        
                        full_amplitude = np.nan_to_num(full_amplitude, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        plt.figure(figsize=(16, 12))
                        plt.imshow(full_amplitude, cmap='gray', vmin=vmin_val, vmax=vmax_val, aspect='auto')
                        plt.colorbar(label='Amplitude')
                        plt.title('SAR Image Amplitude (Full Resolution)')
                        plt.xlabel("Azimuth Samples")
                        plt.ylabel("Range Samples")
                        plt.tight_layout()
                        
                        full_filename = os.path.join(self.checkpoint_dir, 'SlcCuPy_SuperOptimized_Full.png')
                        plt.savefig(full_filename, dpi=150)
                        print(f"Full resolution image saved to {full_filename}")
                    except Exception as e:
                        print(f"Could not generate full-resolution preview: {e}")
        
        except ImportError:
            print("Matplotlib not installed. Cannot generate preview.")
        except Exception as e:
            print(f"Error generating preview: {e}")
            import traceback
            traceback.print_exc()
    
    def export_to_formats(self, export_split=True, export_interleaved=True, export_numpy=True):
        """Export the HDF5 processed data to other formats."""
        try:
            with h5py.File(self.output_file, 'r') as h5f:
                print("Loading data from HDF5 file...")
                slc = h5f['slc'][:]
                
                if not slc.flags.f_contiguous:
                    print("Converting data to Fortran order (column-major) for export...")
                    slc = np.asfortranarray(slc)
                else:
                    print("Data is already in Fortran order (column-major)")
                
                base_name = 'SlcCuPy_SuperOptimized'
                
                if export_split:
                    output_filename = f"{base_name}_SplitF32.dat"
                    print(f"Exporting to split float32 format: {output_filename}")
                    real_part_f32 = slc.real.astype(np.float32)
                    imag_part_f32 = slc.imag.astype(np.float32)
                    with open(output_filename, 'wb') as f:
                        real_part_f32.flatten(order='F').tofile(f)
                        imag_part_f32.flatten(order='F').tofile(f)
                
                if export_interleaved:
                    output_filename = f"{base_name}_InterleavedF32.dat"
                    print(f"Exporting to interleaved float32 format: {output_filename}")
                    real_flat_f32 = slc.real.astype(np.float32).flatten(order='F')
                    imag_flat_f32 = slc.imag.astype(np.float32).flatten(order='F')
                    interleaved_f32 = np.empty(real_flat_f32.size * 2, dtype=np.float32)
                    interleaved_f32[0::2] = real_flat_f32
                    interleaved_f32[1::2] = imag_flat_f32
                    with open(output_filename, 'wb') as f:
                        interleaved_f32.tofile(f)
                
                if export_numpy:
                    output_filename = f"{base_name}.npy"
                    print(f"Exporting to NumPy format: {output_filename}")
                    np.save(output_filename, slc)
                    meta_filename = f"{base_name}_metadata.json"
                    with open(meta_filename, 'w') as f:
                        json.dump({
                            'shape': slc.shape,
                            'fortran_order': True,
                            'dtype': 'complex128'
                        }, f, indent=2)
                
                print("Export completed.")
        
        except Exception as e:
            print(f"Error during export: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_binary_subset(self, filepath, dim_rg, dim_az, rg_start, rg_end, az_start, az_end, dtype):
        """Load a subset of a binary file for a specific region, preserving Fortran order."""
        element_size = np.dtype(dtype).itemsize
        
        az_end_safe = min(az_end, dim_az)
        if az_end_safe < az_start:
             az_end_safe = az_start

        with open(filepath, 'rb') as f:
            try:
                full_data = np.memmap(
                    f, 
                    dtype=dtype, 
                    mode='r', 
                    shape=(dim_rg, dim_az),
                    order='F'
                )
            except ValueError as e:
                print(f"Error creating memmap for {filepath} with shape {(dim_rg, dim_az)}: {e}")
                raise
            
            actual_rg_end = min(rg_end, dim_rg)
            actual_az_end = min(az_end_safe, dim_az)

            if rg_start >= actual_rg_end or az_start >= actual_az_end:
                print(f"Warning: Attempting to load an empty or invalid slice from {filepath}. Rg: {rg_start}-{actual_rg_end}, Az: {az_start}-{actual_az_end}")
                return np.array([], dtype=dtype).reshape((actual_rg_end - rg_start, actual_az_end - az_start), order='F')

            subset = full_data[rg_start:actual_rg_end, az_start:actual_az_end].copy(order='F')
            
        return subset

# ###########################################################################
# # Main Execution Script
# ###########################################################################
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" SUPER OPTIMIZED SAR BACK-PROJECTION FOCUSER FOR LARGE DATASETS")
    print(" Designed for RTX 4090 GPU and 64GB RAM systems")
    print("="*80)
    
    print("\nSystem Information:")
    mem_info = get_memory_info()
    print(f" System RAM: {mem_info['ram_total']:.1f} GB")
    print(f" GPU Memory: {mem_info['gpu_total']:.1f} GB")
    
    focuser = SARSuperOptimizedFocuser()
    
    try:
        focuser.process_dataset()
        focuser.generate_preview()
        focuser.export_to_formats()
        
        print("\nProcessing completed successfully!")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()