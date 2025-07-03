import numpy as np
import cupy as cp
import time
import os

# ###########################################################################
# # Kernel CuPy per processare tutti i punti in parallelo
# ###########################################################################
process_point_kernel = cp.RawKernel(r'''
extern "C" __global__
void process_point_kernel(
    const double* xxg1_flat, const double* yyg1_flat, const double* zzg1_flat,
    const unsigned char* mask_flat, const int* index_flat, const int* Sint_flat,
    const double* wgsx, const double* wgsy, const double* wgsz,
    const float* Raw_Data_real, const float* Raw_Data_imag,
    const double near_range, const double dra, const double lambda_val,
    const int Nra, const int Naz,
    double* focused_real, double* focused_imag,
    const int total_points)
{
    // Get thread ID
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Check if thread ID is within bounds
    if (ii >= total_points)
        return;
        
    // Precision Constants (definite direttamente nel kernel)
    const double C_F64 = 2.9979246e8;
    const double ALFA_F64 = 3.335987619777693e11;
    const double PI_F64 = 3.141592741012573242;
    const double BETA_F64 = 4.0 * PI_F64 * ALFA_F64 / (C_F64 * C_F64);

    // Inizializza output a zero
    focused_real[ii] = 0.0;
    focused_imag[ii] = 0.0;

    // Salta se mascherato
    if (mask_flat[ii] != 1)
        return;
        
    // Ottieni valore sint per questo punto
    int sint = Sint_flat[ii];
    // Controlla valore non valido
    if (sint == -9999)
        return;
        
    // Calcola jj_size (indici apertura sintetica)
    int jj_size = int(ceil(sint / 2.0)) * 2 + 1;
    
    // Inizializza accumulatore per la somma di back projection
    double back_projected_sum_real = 0.0;
    double back_projected_sum_imag = 0.0;
    int n_valid_contributors = 0;
    
    // Loop sull'apertura
    for (int j = 0; j < jj_size; j++) {
        // Calcola indice in azimuth (assoluto)
        int index1 = index_flat[ii] + j - int(ceil(abs((double)sint) / 2.0));
        
        // Controlla se l'indice è nel range azimuth valido
        if (index1 >= Naz - 1 || index1 <= 0)
            continue;
            
        // Calcola posizione relativa
        double xcomp = xxg1_flat[ii] - wgsx[index1];
        double ycomp = yyg1_flat[ii] - wgsy[index1];
        double zcomp = zzg1_flat[ii] - wgsz[index1];
        
        // Calcola distanza al quadrato e distanza
        double dist_patch2 = xcomp*xcomp + ycomp*ycomp + zcomp*zcomp;
        double dist_patch = sqrt(dist_patch2);
        
        // Calcola indice di range
        int ind_line_ra = int(round((dist_patch - near_range) / dra));
        
        // Controlla se l'indice di range è valido
        if (ind_line_ra >= Nra - 1 || ind_line_ra < 0)
            continue;
            
        // Calcola termine di fase
        double phase_term = (-4.0 * PI_F64 / lambda_val * dist_patch) + (BETA_F64 * dist_patch2);
        
        // Filtro complesso coniugato (usando coseno e seno)
        double cos_phase = cos(-phase_term);
        double sin_phase = sin(-phase_term);
        
        // Accedi ai dati raw usando indirizzamento row-major per CuPy (C-style)
        float raw_real = Raw_Data_real[ind_line_ra * Naz + index1];
        float raw_imag = Raw_Data_imag[ind_line_ra * Naz + index1];
        
        // Calcola contributo (moltiplicazione complessa)
        double term_real = dist_patch2 * (raw_real * cos_phase - raw_imag * sin_phase);
        double term_imag = dist_patch2 * (raw_real * sin_phase + raw_imag * cos_phase);
        
        // Accumula
        back_projected_sum_real += term_real;
        back_projected_sum_imag += term_imag;
        n_valid_contributors += 1;
    }
    
    // Imposta output
    if (n_valid_contributors > 0) {
        focused_real[ii] = back_projected_sum_real / n_valid_contributors;
        focused_imag[ii] = back_projected_sum_imag / n_valid_contributors;
    }
}
''', 'process_point_kernel')

# ###########################################################################
# # Funzione Principale di Focalizzazione CuPy
# ###########################################################################
def FocalizzatoreBpCuPy(xxg1, yyg1, zzg1, mask, index, wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, Sint):
    """
    GPU-accelerated SAR focusing versione CuPy che replica esattamente l'algoritmo CPU.
    """
    print('FOCALIZADOR VERSIÓN CuPy (GPU-Accelerated)')
    t1 = time.time()

    # Ottieni dimensioni
    if Raw_Data.ndim != 2:
        raise ValueError("Raw_Data must be a 2D array")
    Nra, Naz = Raw_Data.shape

    if xxg1.ndim != 2:
        raise ValueError("xxg1 must be a 2D array")
    dimx, dimy = xxg1.shape
    dim = dimx * dimy  # Numero totale di pixel

    # Appiattisci array input usando ordine Fortran ('F') come nella versione CPU
    print("Flattening input arrays (order='F')...")
    xxg1_flat = xxg1.astype(np.float64).flatten(order='F')
    yyg1_flat = yyg1.astype(np.float64).flatten(order='F')
    zzg1_flat = zzg1.astype(np.float64).flatten(order='F')
    mask_flat = mask.astype(np.uint8).flatten(order='F')
    index_flat = index.astype(np.int32).flatten(order='F')
    Sint_flat = Sint.astype(np.int32).flatten(order='F')

    # Applica maschera per valori Sint non validi
    no_valid_indices = np.where(Sint_flat == -9999)[0]
    if no_valid_indices.size > 0:
        print(f"Masking {no_valid_indices.size} points due to Sint == -9999.")
        mask_flat[no_valid_indices] = 0

    # Dividi RawData in componenti reali e immaginarie per CuPy
    # Mantieni l'ordine di storage di NumPy ma converti in array C-contiguous
    Raw_Data_real = Raw_Data.real.astype(np.float32).copy(order='C')
    Raw_Data_imag = Raw_Data.imag.astype(np.float32).copy(order='C')

    # Inizializza array di output (complesso)
    focused_real = np.zeros(dim, dtype=np.float64)
    focused_imag = np.zeros(dim, dtype=np.float64)

    # Trasferisci dati alla GPU
    print("Transferring data to GPU...")
    d_xxg1_flat = cp.asarray(xxg1_flat)
    d_yyg1_flat = cp.asarray(yyg1_flat)
    d_zzg1_flat = cp.asarray(zzg1_flat)
    d_mask_flat = cp.asarray(mask_flat)
    d_index_flat = cp.asarray(index_flat)
    d_Sint_flat = cp.asarray(Sint_flat)
    d_wgsx = cp.asarray(wgsx)
    d_wgsy = cp.asarray(wgsy)
    d_wgsz = cp.asarray(wgsz)
    d_Raw_Data_real = cp.asarray(Raw_Data_real)
    d_Raw_Data_imag = cp.asarray(Raw_Data_imag)
    d_focused_real = cp.asarray(focused_real)
    d_focused_imag = cp.asarray(focused_imag)

    # Configura grid e blocchi CUDA
    print("Configuring CUDA kernel...")
    threads_per_block = 256
    blocks_per_grid = (dim + threads_per_block - 1) // threads_per_block
    
    print(f"CUDA Configuration: {blocks_per_grid} blocks with {threads_per_block} threads each")
    print(f"Processing {dim} pixels in parallel on GPU")
    
    # Lancia kernel
    print("Launching CUDA kernel...")
    kernel_launch_start = time.time()
    
    process_point_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (d_xxg1_flat, d_yyg1_flat, d_zzg1_flat,
         d_mask_flat, d_index_flat, d_Sint_flat,
         d_wgsx, d_wgsy, d_wgsz,
         d_Raw_Data_real, d_Raw_Data_imag,
         np.float64(near_range), np.float64(dra), np.float64(lambda_val),
         np.int32(Nra), np.int32(Naz),
         d_focused_real, d_focused_imag,
         np.int32(dim))
    )
    
    # Sincronizza per assicurare il completamento del kernel
    cp.cuda.stream.get_current_stream().synchronize()
    kernel_end = time.time()
    print(f"CUDA kernel execution time: {kernel_end - kernel_launch_start:.2f} seconds")
    
    # Copia risultati dalla GPU
    print("Transferring results back to CPU...")
    focused_real = cp.asnumpy(d_focused_real)
    focused_imag = cp.asnumpy(d_focused_imag)
    
    # Combina componenti in array complesso
    focused_sample = focused_real + 1j * focused_imag
    
    # Reshape nell'array 2D originale
    print("Reforming output array to 2D (order='F')...")
    focused_sample_2d = np.reshape(focused_sample, (dimx, dimy), order='F')
    
    # Calcola tempo trascorso
    t2 = time.time()
    print(f"Total GPU processing time: {t2 - t1:.2f} seconds")
    
    # Libera memoria GPU
    del d_xxg1_flat, d_yyg1_flat, d_zzg1_flat
    del d_mask_flat, d_index_flat, d_Sint_flat
    del d_wgsx, d_wgsy, d_wgsz
    del d_Raw_Data_real, d_Raw_Data_imag
    del d_focused_real, d_focused_imag
    cp.get_default_memory_pool().free_all_blocks()
    
    return focused_sample_2d


# ###########################################################################
# # Versione con Batching per Dataset molto grandi
# ###########################################################################
def FocalizzatoreBpCuPy_Batched(xxg1, yyg1, zzg1, mask, index, wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, Sint, batch_size=500000):
    """
    GPU-accelerated SAR focusing con CuPy e batching per dataset molto grandi.
    """
    print('FOCALIZADOR VERSIÓN CuPy CON BATCHING (GPU-Accelerated)')
    t1 = time.time()

    # Ottieni dimensioni
    if Raw_Data.ndim != 2:
        raise ValueError("Raw_Data must be a 2D array")
    Nra, Naz = Raw_Data.shape

    if xxg1.ndim != 2:
        raise ValueError("xxg1 must be a 2D array")
    dimx, dimy = xxg1.shape
    dim = dimx * dimy  # Numero totale di pixel

    # Appiattisci array input usando ordine Fortran ('F')
    print("Flattening input arrays (order='F')...")
    xxg1_flat = xxg1.astype(np.float64).flatten(order='F')
    yyg1_flat = yyg1.astype(np.float64).flatten(order='F')
    zzg1_flat = zzg1.astype(np.float64).flatten(order='F')
    mask_flat = mask.astype(np.uint8).flatten(order='F')
    index_flat = index.astype(np.int32).flatten(order='F')
    Sint_flat = Sint.astype(np.int32).flatten(order='F')

    # Applica maschera per valori Sint non validi
    no_valid_indices = np.where(Sint_flat == -9999)[0]
    if no_valid_indices.size > 0:
        print(f"Masking {no_valid_indices.size} points due to Sint == -9999.")
        mask_flat[no_valid_indices] = 0

    # Dividi RawData in componenti reali e immaginarie per CuPy
    Raw_Data_real = Raw_Data.real.astype(np.float32).copy(order='C')
    Raw_Data_imag = Raw_Data.imag.astype(np.float32).copy(order='C')

    # Inizializza array di output
    focused_real = np.zeros(dim, dtype=np.float64)
    focused_imag = np.zeros(dim, dtype=np.float64)

    # Trasferisci dati condivisi alla GPU
    print("Transferring shared data to GPU...")
    d_wgsx = cp.asarray(wgsx)
    d_wgsy = cp.asarray(wgsy)
    d_wgsz = cp.asarray(wgsz)
    d_Raw_Data_real = cp.asarray(Raw_Data_real)
    d_Raw_Data_imag = cp.asarray(Raw_Data_imag)
    
    # Calcola il numero di batch
    num_batches = (dim + batch_size - 1) // batch_size
    
    print(f"Processing in {num_batches} batches, each with up to {batch_size} pixels")
    
    # Configura blocchi CUDA
    threads_per_block = 256
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, dim)
        batch_dim = end_idx - start_idx
        
        print(f"Processing batch {batch_idx+1}/{num_batches} (pixels {start_idx}-{end_idx-1})")
        
        # Estrai dati del batch
        batch_xxg1 = xxg1_flat[start_idx:end_idx]
        batch_yyg1 = yyg1_flat[start_idx:end_idx]
        batch_zzg1 = zzg1_flat[start_idx:end_idx]
        batch_mask = mask_flat[start_idx:end_idx]
        batch_index = index_flat[start_idx:end_idx]
        batch_Sint = Sint_flat[start_idx:end_idx]
        
        # Inizializza output del batch
        batch_focused_real = np.zeros(batch_dim, dtype=np.float64)
        batch_focused_imag = np.zeros(batch_dim, dtype=np.float64)
        
        # Trasferisci dati del batch alla GPU
        d_batch_xxg1 = cp.asarray(batch_xxg1)
        d_batch_yyg1 = cp.asarray(batch_yyg1)
        d_batch_zzg1 = cp.asarray(batch_zzg1)
        d_batch_mask = cp.asarray(batch_mask)
        d_batch_index = cp.asarray(batch_index)
        d_batch_Sint = cp.asarray(batch_Sint)
        d_batch_focused_real = cp.asarray(batch_focused_real)
        d_batch_focused_imag = cp.asarray(batch_focused_imag)
        
        # Configura grid per questo batch
        blocks_per_grid = (batch_dim + threads_per_block - 1) // threads_per_block
        
        # Lancia kernel per questo batch
        process_point_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (d_batch_xxg1, d_batch_yyg1, d_batch_zzg1,
             d_batch_mask, d_batch_index, d_batch_Sint,
             d_wgsx, d_wgsy, d_wgsz,
             d_Raw_Data_real, d_Raw_Data_imag,
             np.float64(near_range), np.float64(dra), np.float64(lambda_val),
             np.int32(Nra), np.int32(Naz),
             d_batch_focused_real, d_batch_focused_imag,
             np.int32(batch_dim))
        )
        
        # Sincronizza
        cp.cuda.stream.get_current_stream().synchronize()
        
        # Copia risultati
        batch_focused_real = cp.asnumpy(d_batch_focused_real)
        batch_focused_imag = cp.asnumpy(d_batch_focused_imag)
        
        # Memorizza nell'array completo
        focused_real[start_idx:end_idx] = batch_focused_real
        focused_imag[start_idx:end_idx] = batch_focused_imag
        
        # Libera memoria GPU del batch
        del d_batch_xxg1, d_batch_yyg1, d_batch_zzg1
        del d_batch_mask, d_batch_index, d_batch_Sint
        del d_batch_focused_real, d_batch_focused_imag
    
    # Libera memoria GPU rimanente
    del d_wgsx, d_wgsy, d_wgsz
    del d_Raw_Data_real, d_Raw_Data_imag
    cp.get_default_memory_pool().free_all_blocks()
    
    # Combina componenti reali e immaginarie
    focused_sample = focused_real + 1j * focused_imag
    
    # Reshape nell'array 2D originale
    print("Reforming output array to 2D (order='F')...")
    focused_sample_2d = np.reshape(focused_sample, (dimx, dimy), order='F')
    
    # Calcola tempo trascorso
    t2 = time.time()
    print(f"Total GPU processing time: {t2 - t1:.2f} seconds")
    
    return focused_sample_2d


# ###########################################################################
# # Funzione per caricare i dati RAW gestendo i casi problematici
# ###########################################################################
def load_raw_data(data_dir, filename, expected_Nra, expected_Naz):
    """
    Carica i dati RawData gestendo anche i casi problematici come un numero dispari di float.
    Ritorna: array RawData, Nra effettivo, Naz effettivo
    """
    print(f"Loading RawData from {filename}...")
    raw_data_path = os.path.join(data_dir, filename)
    
    # Calcola i float totali in base alla dimensione del file
    file_size_bytes = os.path.getsize(raw_data_path)
    bytes_per_float = 4  # float32 occupa 4 byte
    total_floats = file_size_bytes // bytes_per_float
    
    print(f"RawData file contains {total_floats} float32 values")
    
    # Verifica se il numero è dispari
    if total_floats % 2 != 0:
        print(f"Warning: Il file RawData ha un numero dispari di float ({total_floats}).")
        print("Ignoro l'ultimo float per rendere il numero pari...")
        
        # Carica tutti i dati tranne l'ultimo float
        data_f32 = np.fromfile(raw_data_path, dtype=np.float32)[:-1]
        print(f"Loaded {data_f32.size} floats after ignoring last value")
    else:
        # Numero pari di float, carica normalmente
        data_f32 = np.fromfile(raw_data_path, dtype=np.float32)
    
    # Calcola il numero di valori complessi e le dimensioni
    total_complex = data_f32.size // 2
    
    # Determina le dimensioni migliori per reshaping
    # Prima cerca di usare Nra atteso
    actual_Nra = expected_Nra
    
    # Calcola Naz in base a Nra
    if total_complex % actual_Nra == 0:
        actual_Naz = total_complex // actual_Nra
        print(f"Using dimensions: {actual_Nra} x {actual_Naz}")
    else:
        print(f"Warning: I dati non possono essere divisi equamente usando Nra={actual_Nra}")
        print("Cercando dimensioni alternative...")
        
        # Cerca fattori vicini alle dimensioni attese
        import math
        factor = int(math.sqrt(total_complex))
        # Trova un fattore che divide perfettamente
        while total_complex % factor != 0 and factor > 1:
            factor -= 1
        
        if factor > 1:
            actual_Nra = factor
            actual_Naz = total_complex // factor
            print(f"Using closest valid dimensions: {actual_Nra} x {actual_Naz}")
        else:
            # Ultimo tentativo: usa le dimensioni originali e taglia i dati
            actual_Nra = expected_Nra
            actual_Naz = total_complex // actual_Nra
            if total_complex % actual_Nra != 0:
                print(f"Warning: Truncating data to fit {actual_Nra} x {actual_Naz}")
                data_f32 = data_f32[:2 * actual_Nra * actual_Naz]
    
    # Prova formato interleaved
    try:
        print(f"Trying interleaved format with dimensions {actual_Nra} x {actual_Naz}...")
        real_part = data_f32[0::2].reshape((actual_Nra, actual_Naz), order='F')
        imag_part = data_f32[1::2].reshape((actual_Nra, actual_Naz), order='F')
        RawData = real_part + 1j * imag_part
        print("Successfully loaded with interleaved format")
    except Exception as e:
        print(f"Interleaved format failed: {e}")
        
        # Prova formato split
        try:
            print(f"Trying split format with dimensions {actual_Nra} x {actual_Naz}...")
            half_len = data_f32.size // 2
            real_part = data_f32[:half_len].reshape((actual_Nra, actual_Naz), order='F')
            imag_part = data_f32[half_len:].reshape((actual_Nra, actual_Naz), order='F')
            RawData = real_part + 1j * imag_part
            print("Successfully loaded with split format")
        except Exception as e2:
            raise IOError(f"All load attempts failed. Last error: {e2}")
    
    print(f"RawData loaded with dimensions: {RawData.shape}")
    return RawData, RawData.shape[0], RawData.shape[1]


# ###########################################################################
# # Script Principale di Esecuzione
# ###########################################################################
if __name__ == "__main__":
    start_time_total = time.time()

    print("="*60)
    print(" SAR Back-projection Focusing (CuPy GPU-Accelerated)")
    print(" Identical to CPU algorithm but GPU-accelerated")
    print("="*60)

    # Stampa informazioni sulla GPU
    try:
        device = cp.cuda.Device(0)
        print("\nGPU Information:")
        print(f" Device Name: {device.name}")
        print(f" Compute Capability: {device.compute_capability}")
        print(f" Total Memory: {device.mem_info[1] / (1024**3):.2f} GB")
        print(f" Free Memory: {device.mem_info[0] / (1024**3):.2f} GB")
        print("-"*60)
    except Exception as e:
        print(f"Error getting GPU info: {e}")

    # Dimensioni come definite nel test originale
    DimRg = 4000
    DimAz = 250
    Nra_raw = 4762  # Range dimension of RawData
    Naz_raw = 40001  # Default azimuth dimension - will be adjusted based on actual data

    print(f"Output Grid Dimensions: Range={DimRg}, Azimuth={DimAz}")
    print(f"Initial Raw Data Dimensions: Range={Nra_raw}, Azimuth={Naz_raw}")

    # Inizializzazione degli array
    print("\nInitializing NumPy arrays...")
    try:
        xxg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        yyg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        zzg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        Index = np.zeros((DimRg, DimAz), dtype=np.int32)
        Mask = np.zeros((DimRg, DimAz), dtype=np.uint8)
        Sint = np.zeros((DimRg, DimAz), dtype=np.int32)
        # RawData sarà inizializzato dopo il caricamento
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
    data_dir = "."  # Assume files in current directory, or specify path
    all_files_found = True
    required_files = [
        'Dem_x_Dbl4000x250.dat', 'Dem_y_Dbl4000x250.dat', 'Dem_z_Dbl4000x250.dat',
        'Traiett_x_Dbl40001.dat', 'Traiett_y_Dbl40001.dat', 'Traiett_z_Dbl40001.dat',
        'Index_Long_4000x250.dat', 'Mask_Byte_4000x250.dat', 'Sint_Long_4000x250.dat',
        'RawData_Cmplx_4762x40001.dat',
        'NearRangeDbl_1_elemento.dat', 'DeltaRangeDbl_1_elemento.dat', 'LambdaDbl_1_elemento.dat'
    ]

    # Verifica esistenza file
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

    # Carica dati da file binari
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

        # Usa il loader personalizzato per RawData
        RawData, Nra_raw, Naz_raw = load_raw_data(data_dir, 'RawData_Cmplx_4762x40001.dat', 4762, 40001)

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

        # Verifica dimensioni traiettoria vs RawData
        traj_len = min(len(wgsx1), len(wgsy1), len(wgsz1))
        print(f"Trajectory length: {traj_len}, RawData Naz: {Naz_raw}")
        
        # Regola se necessario
        if traj_len < Naz_raw:
            print(f"Warning: Trajectory is shorter than RawData azimuth dimension.")
            print(f"Truncating RawData azimuth dimension to {traj_len}")
            RawData = RawData[:, :traj_len]
            Naz_raw = traj_len
        elif traj_len > Naz_raw:
            print(f"Warning: Trajectory is longer than RawData azimuth dimension.")
            print(f"Will use trajectory points up to RawData length only.")
            # Troncamento traiettoria per corrispondere a RawData
            wgsx1 = wgsx1[:Naz_raw] 
            wgsy1 = wgsy1[:Naz_raw]
            wgsz1 = wgsz1[:Naz_raw]

    except FileNotFoundError as e:
        print(f"\nError loading data: File not found - {e}")
        exit()
    except IOError as e:
        print(f"\nError loading data: IO Error - {e}")
        exit()
    except ValueError as e:
        print(f"\nError loading data: Value Error - {e}")
        exit()
    except Exception as e:
        print(f"\nAn unexpected error occurred during data loading: {e}")
        exit()

    print("\nPHASE 2: STARTING GPU FOCUSING PROCESS")
    # Scegli l'implementazione
    use_batched = True  # Usa batching per dataset grandi
    
    if use_batched:
        SlcCuPy = FocalizzatoreBpCuPy_Batched(xxg1, yyg1, zzg1, Mask, Index, wgsx1, wgsy1, wgsz1, RawData, NearRange, DeltaRange, Lambda, Sint)
    else:
        SlcCuPy = FocalizzatoreBpCuPy(xxg1, yyg1, zzg1, Mask, Index, wgsx1, wgsy1, wgsz1, RawData, NearRange, DeltaRange, Lambda, Sint)

    print("\nPHASE 3: SAVING RESULTS")
    try:
        output_filename_base = 'SlcCuPy_GPU'

        # 1. Salva in formato split float32 (real_part, imag_part)
        output_filename_split = output_filename_base + '_SplitF32.dat'
        print(f"Saving in split float32 format to: {output_filename_split}...")
        real_part_f32 = SlcCuPy.real.astype(np.float32)
        imag_part_f32 = SlcCuPy.imag.astype(np.float32)
        with open(output_filename_split, 'wb') as f:
            # Scrivi prima parte reale, poi parte immaginaria
            real_part_f32.flatten(order='F').tofile(f)
            imag_part_f32.flatten(order='F').tofile(f)
        print("Split float32 format saved.")

        # 2. Salva in formato interleaved float32
        output_filename_interleaved = output_filename_base + '_InterleavedF32.dat'
        print(f"Saving in interleaved float32 format to: {output_filename_interleaved}...")
        # Assicura formato float32 prima dell'interleaving
        real_flat_f32 = SlcCuPy.real.astype(np.float32).flatten(order='F')
        imag_flat_f32 = SlcCuPy.imag.astype(np.float32).flatten(order='F')
        interleaved_f32 = np.empty(real_flat_f32.size * 2, dtype=np.float32)
        interleaved_f32[0::2] = real_flat_f32
        interleaved_f32[1::2] = imag_flat_f32
        with open(output_filename_interleaved, 'wb') as f:
            interleaved_f32.tofile(f)
        print("Interleaved float32 format saved.")

        # 3. Salva in formato NumPy nativo
        output_filename_npy = output_filename_base + '.npy'
        print(f"Saving in NumPy native format (.npy) to: {output_filename_npy}...")
        np.save(output_filename_npy, SlcCuPy)
        print("NumPy .npy format saved.")

    except Exception as e:
        print(f"\nError saving results: {e}")

    # Stampa tempo totale
    elapsed_total = time.time() - start_time_total
    print(f"\nTotal execution time: {elapsed_total:.2f} seconds ({elapsed_total/60.0:.2f} minutes)")
    print("="*60)

    # Visualizzazione immagine ampiezza
    try:
        import matplotlib.pyplot as plt
        print("\nGenerating amplitude image preview...")
        plt.figure(figsize=(10, 8))

        # Calcola ampiezza
        amplitude = np.abs(SlcCuPy)
        valid_amplitude = amplitude[np.isfinite(amplitude)]
        if valid_amplitude.size == 0:
            print("Warning: Amplitude array contains no valid finite values. Cannot display.")
        else:
            mean_amp = np.mean(valid_amplitude)
            vmax_val = 15 * mean_amp if mean_amp > 1e-9 else 1.0
            vmin_val = np.percentile(valid_amplitude[valid_amplitude > 1e-9] if np.any(valid_amplitude > 1e-9) else valid_amplitude, 1)

            plt.imshow(amplitude, cmap='gray', vmin=vmin_val, vmax=vmax_val, aspect='auto')
            plt.colorbar(label='Amplitude')
            plt.title(f'SAR Image Amplitude (CuPy GPU Output)\nvmax ≈ 15 * mean = {vmax_val:.2e}')
            plt.xlabel("Azimuth Samples")
            plt.ylabel("Range Samples")
            plt.tight_layout()
            img_filename = 'SlcCuPy_GPU_Amplitude.png'
            plt.savefig(img_filename, dpi=150)
            print(f"Image preview saved to {img_filename}")
    except ImportError:
        print("\nMatplotlib not installed. Skipping image preview generation.")
        print("To install: pip install matplotlib")
    except Exception as e:
        print(f"\nCould not create image preview due to an error: {e}")