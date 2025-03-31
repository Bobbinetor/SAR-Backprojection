import numpy as np
import time
import os
from numba import jit, prange # Importa prange per potenziale parallelizzazione

# ###########################################################################
# # Funzione Numba JIT per processare un singolo punto (pixel)
# ###########################################################################
@jit(nopython=True) # cache=True potrebbe velocizzare le esecuzioni successive
def process_point(xxg1_flat_ii, yyg1_flat_ii, zzg1_flat_ii, # Valori scalari per il punto ii
                  mask_flat_ii, index_flat_ii, Sint_flat_ii, # Valori scalari per il punto ii
                  wgsx, wgsy, wgsz,         # Traiettoria (array 1D float64)
                  Raw_Data,                 # Dati grezzi (array 2D complex64)
                  near_range, dra, lambda_val, # Parametri scalari (float64)
                  Nra, Naz,                 # Dimensioni RawData (int)
                  ind_line_ra_min=0, index_min=0): # Offsets (int, probabilmente 0)
    """
    Processa un singolo punto (pixel) ii della griglia di output.
    Utilizza precisione float64 per calcoli geometrici e di fase.
    Assume Raw_Data sia complex64.
    Restituisce un valore complex128.
    """
    # Costanti fisiche definite con precisione float64 all'interno della funzione JIT
    C_F64 = 2.9979246e8
    ALFA_F64 = 3.335987619777693e11
    PI_F64 = np.pi
    # Beta calcolato in float64
    BETA_F64 = 4.0 * PI_F64 * ALFA_F64 / (C_F64 * C_F64)
    # Lambda in float64 (passato come argomento)

    # --- Inizio Logica IDL ---

    # Salta se mascherato (IDL: if(mask(ii) eq 1l) then begin)
    if mask_flat_ii != 1:
        return np.complex128(0j) # Restituisce il tipo corretto

    # Ottieni valore sint per questo punto (IDL: sint(ii))
    sint = Sint_flat_ii # Questo è int32
    # Controlla anche valore non valido (redundante se già mascherato prima, ma sicuro)
    if sint == -9999:
        return np.complex128(0j)

    # Calcola jj (indici relativi apertura sintetica)
    # IDL: jj = indgen(ceil(sint(ii)/2.)*2l+1l)
    jj_size = int(np.ceil(sint / 2.0)) * 2 + 1
    jj = np.arange(jj_size, dtype=np.int32) # Array di interi

    # Calcola index1 (indici assoluti in azimuth nel Raw_Data)
    # IDL: index1 = index(ii)+jj-ceil(abs(sint(ii))/2.)
    index1 = index_flat_ii + jj - int(np.ceil(np.abs(sint) / 2.0)) # Array di interi

    # Trova indici validi nell'intervallo di azimuth del RawData
    # IDL: area = where((index1 lt NAZ-1) and (index1 gt 0))
    area = np.where((index1 < Naz - 1) & (index1 > 0))[0] # Indici di jj che soddisfano la condizione

    # IDL: if(area(0) ne -1) then begin
    if len(area) > 0:
        # Seleziona gli indici assoluti validi (index1a)
        # IDL: index1a = index1(area)
        index1a = index1[area] # Array di interi

        # Calcola indici per accedere a wgsx/y/z (assumendo index_min=0)
        # IDL: index1a-index_min
        idx_wgs = index1a # Array di interi

        # --- Calcoli geometrici in float64 ---
        # xxg1_flat_ii, ecc. sono float64 scalari
        # wgsx[idx_wgs], ecc. sono array float64
        # NumPy gestisce la sottrazione scalare - array
        xcomp = xxg1_flat_ii - wgsx[idx_wgs]
        ycomp = yyg1_flat_ii - wgsy[idx_wgs]
        zcomp = zzg1_flat_ii - wgsz[idx_wgs]

        # Distanza al quadrato (float64)
        # IDL: dist_patch2 = xcomp*xcomp+ycomp*ycomp+zcomp*zcomp
        dist_patch2 = xcomp*xcomp + ycomp*ycomp + zcomp*zcomp

        # Distanza (float64)
        # IDL: dist_patch = sqrt(dist_patch2)
        # Aggiungi un piccolo epsilon per evitare sqrt(0) se necessario, anche se raro
        # dist_patch = np.sqrt(dist_patch2 + 1e-12)
        dist_patch = np.sqrt(dist_patch2)


        # Calcola indice di range (float64 -> round -> int32)
        # IDL: ind_line_ra = round((dist_patch-near_range)/dra)
        # near_range, dra sono float64
        ind_line_ra_float = (dist_patch - near_range) / dra
        ind_line_ra = np.round(ind_line_ra_float).astype(np.int32) # Array di interi

        # Trova indici validi nell'intervallo di range del RawData
        # IDL: ind_in = where(ind_line_ra lt Nra-1)
        # Considera anche >= 0 per sicurezza
        ind_in = np.where((ind_line_ra < Nra - 1) & (ind_line_ra >= ind_line_ra_min))[0] # Indici di 'area'/'index1a' che soddisfano la condizione

        # IDL: if(ind_in(0) ne -1) then begin
        if len(ind_in) > 0:
            # --- Calcolo Fase (float64) e Filtro (complex128) ---
            # Seleziona solo i valori corrispondenti a ind_in
            dist_patch_in = dist_patch[ind_in]
            dist_patch2_in = dist_patch2[ind_in]

            # Calcolo del termine di fase (array float64)
            phase_term = (-4.0 * PI_F64 / lambda_val * dist_patch_in) + (BETA_F64 * dist_patch2_in)

            # Calcolo del filtro complesso coniugato (array complex128)
            # IDL usa conj(filter), quindi calcoliamo exp(-1j * phase)
            # filter = exp( j * phase_term ) -> conj(filter) = exp( -j * phase_term )
            conj_filter_val = np.exp(np.complex128(-1j) * phase_term)

            # --- Indicizzazione RawData e Somma ---
            # Ottieni gli indici finali per RawData
            # Gli indici in ind_in si riferiscono agli array xcomp, ycomp, ..., ind_line_ra
            final_indices_range = ind_line_ra[ind_in] # Già scalati con ind_line_ra_min implicito se ind_line_ra >= ind_line_ra_min
            final_indices_az = index1a[ind_in]        # Già scalati con index_min implicito se index_min=0

            # Peso di ampiezza (float64)
            final_dist_patch2 = dist_patch2_in

            # Estrai valori da RawData (complex64) e moltiplica
            # È più sicuro usare un loop per l'indicizzazione 2D in Numba
            back_projected_sum = np.complex128(0j)
            n_valid_contributors = 0
            for k in range(len(final_indices_range)):
                r_idx = final_indices_range[k]
                a_idx = final_indices_az[k]
                # Controllo limiti esplicito (anche se where dovrebbe aver già filtrato)
                if (r_idx >= 0 and r_idx < Nra and a_idx >= 0 and a_idx < Naz):
                    # Lettura da RawData (complex64)
                    raw_value = Raw_Data[r_idx, a_idx]
                    # Calcolo contributo (float64 * complex64 * complex128 -> complex128)
                    term = final_dist_patch2[k] * raw_value * conj_filter_val[k]
                    back_projected_sum += term
                    n_valid_contributors += 1

            # IDL: niter = n_elements(ind_in) -> Usiamo il conteggio effettivo dei contributi validi
            # IDL: focused_sample(ii) = dcomplex(total(back_projected))/niter
            if n_valid_contributors > 0:
                # Somma è complex128, divisione per int -> complex128
                return back_projected_sum / n_valid_contributors

    # Se nessuna condizione è soddisfatta o nessun contributo valido, ritorna 0
    return np.complex128(0j)

# ###########################################################################
# # Funzione Principale di Focalizzazione (simile a IDL)
# ###########################################################################
def FocalizzatoreBpPx(xxg1, yyg1, zzg1, mask, index, wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, Sint):
    """
    Focalizzatore SAR puntuale (pixel-per-pixel) - Corretto in precisione.
    """
    print('FOCALIZADOR VERSIÓN PUNTUAL (Precisione Corretta)')
    t1 = time.time()

    # Costanti definite come float64 (non usate direttamente qui ma per coerenza)
    # C_F64 = np.float64(2.9979246e8)
    # ALFA_F64 = np.float64(3.335987619777693e11)
    # BETA_F64 = np.float64(4.0) * np.float64(np.pi) * ALFA_F64 / (C_F64**2)

    # Ottieni dimensioni come in IDL
    # IDL: siz = size(raw_data) & Nra = siz(1) & Naz = siz(2)
    if Raw_Data.ndim != 2:
        raise ValueError("Raw_Data must be a 2D array")
    Nra, Naz = Raw_Data.shape

    # IDL: siz = size(xxg1) & dimx = siz(1) & dimy = siz(2)
    if xxg1.ndim != 2:
        raise ValueError("xxg1 must be a 2D array")
    dimx, dimy = xxg1.shape
    nx_grid, ny_grid = dimx, dimy

    # IDL: dim = siz(4) # Numero totale di elementi
    dim = dimx * dimy # Equivalente a xxg1.size

    # Inizializza array risultato come complex128 (doppia precisione complessa)
    # IDL: focused_sample = complexarr(dim) -> ma poi usa dcomplex() per la somma
    focused_sample = np.zeros(dim, dtype=np.complex128)

    # Appiattisci gli input usando ordine Fortran ('F') come IDL
    # Assicurati che siano del tipo corretto (float64, int32, uint8)
    print("Flattening input arrays (order='F')...")
    xxg1_flat = xxg1.astype(np.float64).flatten(order='F')
    yyg1_flat = yyg1.astype(np.float64).flatten(order='F')
    zzg1_flat = zzg1.astype(np.float64).flatten(order='F')
    mask_flat = mask.astype(np.uint8).flatten(order='F')
    index_flat = index.astype(np.int32).flatten(order='F')
    Sint_flat = Sint.astype(np.int32).flatten(order='F')
    print("Flattening done.")

    # Applica maschera per valori Sint non validi (come in IDL)
    # IDL: no_valid=where(sint eq -9999)
    # IDL: if (no_valid[0] ne -1) then begin mask[no_valid] = 0l; endif
    # Lavora sulla versione flat per coerenza con il loop
    no_valid_indices = np.where(Sint_flat == -9999)[0]
    if no_valid_indices.size > 0:
        print(f"Masking {no_valid_indices.size} points due to Sint == -9999.")
        mask_flat[no_valid_indices] = 0

    # Costanti IDL (probabilmente 0)
    ind_line_ra_min = 0
    index_min = 0

    # Gestione Checkpoint
    checkpoint_file = 'focalizador_px_checkpoint.npz'
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            print(f"Loading checkpoint from {checkpoint_file}...")
            with np.load(checkpoint_file) as checkpoint:
                # Verifica che le dimensioni corrispondano
                if checkpoint['focused_sample'].shape == focused_sample.shape:
                    focused_sample = checkpoint['focused_sample']
                    start_idx = int(checkpoint['next_idx']) # Assicura sia un intero Python
                    print(f"Resuming from index {start_idx}/{dim}")
                else:
                    print("Checkpoint dimensions mismatch. Starting from scratch.")
                    start_idx = 0
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_idx = 0

    # IDL: for ii=0,n_elements(xxg1)-1l do begin
    # Loop principale pixel-per-pixel
    batch_size = 5000  # Aggiorna checkpoint ogni tot pixel
    print_interval = 100 # Ogni quanti pixel stampare progresso

    print("Starting point-by-point focusing loop...")
    last_print_time = time.time()

    # Passa argomenti scalari per il punto ii a process_point
    for ii in range(start_idx, dim):
        focused_sample[ii] = process_point(
            xxg1_flat[ii], yyg1_flat[ii], zzg1_flat[ii], # Scalari float64
            mask_flat[ii], index_flat[ii], Sint_flat[ii], # Scalari int/byte
            wgsx, wgsy, wgsz,          # Array float64
            Raw_Data,                  # Array complex64
            np.float64(near_range), np.float64(dra), np.float64(lambda_val), # Scalari float64
            Nra, Naz,                  # Int
            ind_line_ra_min, index_min # Int
        )

        # Stampa progresso (come IDL, con sovrascrittura)
        if (ii + 1) % print_interval == 0 or ii == dim - 1:
             completion = round(((ii + 1) / dim) * 100.0)
             current_time = time.time()
             elapsed_since_last = current_time - last_print_time
             pixels_since_last = print_interval if (ii+1) % print_interval == 0 else (ii+1)%print_interval
             rate = pixels_since_last / elapsed_since_last if elapsed_since_last > 0 else 0
             print(f"\rProcessed: {ii + 1}/{dim} ({completion}%) - Rate: {rate:.1f} px/s", end="")
             last_print_time = current_time

        # Salva checkpoint
        if (ii + 1) % batch_size == 0 and ii != dim -1 :
            try:
                np.savez(checkpoint_file,
                         focused_sample=focused_sample,
                         next_idx=np.int64(ii + 1)) # Salva indice successivo
                # print(f"\nCheckpoint saved at index {ii + 1}") # Opzionale: conferma salvataggio
            except Exception as e:
                print(f"\nWarning: Failed to save checkpoint at index {ii+1}: {e}")


    print("\nFocusing loop completed.")

    # Riforma l'array risultato nelle dimensioni 2D originali, usando ordine Fortran
    # IDL: ;focused_sample = reform(focused_sample,dimx,dimy)
    print("Reforming output array to 2D (order='F')...")
    focused_sample_2d = np.reshape(focused_sample, (dimx, dimy), order='F')
    print("Reforming done.")

    # Calcola tempo trascorso
    t2 = time.time()
    print(f"\tElapsed time in FocalizzatoreBpPx: {t2 - t1:.2f} seconds")

    # Rimuovi file checkpoint se esiste
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("Checkpoint file successfully removed.")
        except Exception as e:
            print(f"Warning: Unable to remove checkpoint file: {e}")

    return focused_sample_2d

# ###########################################################################
# # Script Principale di Esecuzione (come nel file IDL di test)
# ###########################################################################
if __name__ == "__main__":
    start_time_total = time.time()

    print("="*60)
    print(" SAR Point-based Back-projection Focusing (Python/Numba)")
    print(" Precision Corrected Version ")
    print("="*60)

    # Dimensioni come definite nel test IDL
    DimRg = 4000
    DimAz = 250
    Nra_raw = 4762 # Dimensione Range RawData
    Naz_raw = 40001 # Dimensione Azimuth RawData

    print(f"Output Grid Dimensions: Range={DimRg}, Azimuth={DimAz}")
    print(f"Raw Data Dimensions: Range={Nra_raw}, Azimuth={Naz_raw}")

    # Inizializzazione array con tipi NumPy corrispondenti a IDL
    # Usiamo float64 per double, int32 per long, uint8 per byte, complex64 per complex
    print("\nInitializing NumPy arrays...")
    try:
        xxg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        yyg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        zzg1 = np.zeros((DimRg, DimAz), dtype=np.float64)
        Index = np.zeros((DimRg, DimAz), dtype=np.int32)
        Mask = np.zeros((DimRg, DimAz), dtype=np.uint8)
        Sint = np.zeros((DimRg, DimAz), dtype=np.int32)
        # RawData sarà complex64 dopo il caricamento
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
    data_dir = "." # Assumi file nella directory corrente, altrimenti specifica path
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

    # Carica dati da file binari (stile IDL readu)
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
        # Leggi come float32
        data_f32 = np.fromfile(raw_data_path, dtype=np.float32)
        expected_len = 2 * Nra_raw * Naz_raw
        if data_f32.size == expected_len:
             # Ricostruisci complex64 da coppie interleaved (real1, imag1, real2, imag2, ...)
             # Assicurati che il reshape usi l'ordine corretto ('F')
             real_part = data_f32[0::2].reshape((Nra_raw, Naz_raw), order='F')
             imag_part = data_f32[1::2].reshape((Nra_raw, Naz_raw), order='F')
             RawData = real_part + 1j * imag_part # Risultato è complex64
             print(f"RawData loaded successfully. Shape: {RawData.shape}, Dtype: {RawData.dtype}")
        else:
             # Prova formato split (tutti i reali, poi tutti gli immaginari)
             print("Interleaved format size mismatch. Trying split format (all reals then all imaginaries)...")
             expected_len_split = Nra_raw * Naz_raw
             if data_f32.size == expected_len: # Riusa expected_len perché è lo stesso numero totale di float32
                 half_len = expected_len // 2
                 real_part = data_f32[:half_len].reshape((Nra_raw, Naz_raw), order='F')
                 imag_part = data_f32[half_len:].reshape((Nra_raw, Naz_raw), order='F')
                 RawData = real_part + 1j * imag_part # Risultato è complex64
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

        # Controllo dimensioni traiettoria vs RawData
        if not (len(wgsx1) == len(wgsy1) == len(wgsz1) == Naz_raw):
             print("\nWarning: Trajectory length does not match RawData azimuth dimension!")
             print(f" WGSX len: {len(wgsx1)}, WGSY len: {len(wgsy1)}, WGSZ len: {len(wgsz1)}, RawData Naz: {Naz_raw}")


    except FileNotFoundError as e:
        print(f"\nError loading data: File not found - {e}")
        exit()
    except IOError as e:
        print(f"\nError loading data: IO Error - {e}")
        exit()
    except ValueError as e:
        print(f"\nError loading data: Value Error (often reshape issue) - {e}")
        exit()
    except Exception as e:
        print(f"\nAn unexpected error occurred during data loading: {e}")
        exit()


    print("\nPHASE 2: STARTING FOCUSING PROCESS")
    # Chiama la funzione di focalizzazione corretta
    SlcPx = FocalizzatoreBpPx(xxg1, yyg1, zzg1, Mask, Index, wgsx1, wgsy1, wgsz1, RawData, NearRange, DeltaRange, Lambda, Sint)

    print("\nPHASE 3: SAVING RESULTS")
    try:
        output_filename_base = 'SlcPx_Python_Corrected'

        # 1. Salva in formato split float32 (real_part, imag_part) come nel codice originale
        output_filename_split = output_filename_base + '_SplitF32.dat'
        print(f"Saving in split float32 format to: {output_filename_split}...")
        real_part_f32 = SlcPx.real.astype(np.float32)
        imag_part_f32 = SlcPx.imag.astype(np.float32)
        with open(output_filename_split, 'wb') as f:
            # Scrivi prima tutta la parte reale, poi tutta quella immaginaria
            real_part_f32.flatten(order='F').tofile(f)
            imag_part_f32.flatten(order='F').tofile(f)
        print("Split float32 format saved.")

        # 2. Salva in formato interleaved float32 (per potenziale compatibilità IDL readu)
        output_filename_interleaved = output_filename_base + '_InterleavedF32.dat'
        print(f"Saving in interleaved float32 format to: {output_filename_interleaved}...")
        # Assicurati che siano float32 prima di interleaving
        real_flat_f32 = SlcPx.real.astype(np.float32).flatten(order='F')
        imag_flat_f32 = SlcPx.imag.astype(np.float32).flatten(order='F')
        interleaved_f32 = np.empty(real_flat_f32.size * 2, dtype=np.float32)
        interleaved_f32[0::2] = real_flat_f32
        interleaved_f32[1::2] = imag_flat_f32
        with open(output_filename_interleaved, 'wb') as f:
             interleaved_f32.tofile(f)
        print("Interleaved float32 format saved.")

        # 3. Salva in formato NumPy nativo (consigliato per riutilizzo in Python)
        output_filename_npy = output_filename_base + '.npy'
        print(f"Saving in NumPy native format (.npy) to: {output_filename_npy}...")
        np.save(output_filename_npy, SlcPx) # Salva l'array complex128 direttamente
        print("NumPy .npy format saved.")


    except Exception as e:
        print(f"\nError saving results: {e}")

    # Stampa tempo totale
    elapsed_total = time.time() - start_time_total
    print(f"\nTotal execution time: {elapsed_total:.2f} seconds ({elapsed_total/60.0:.2f} minutes)")
    print("="*60)

    # Opzionale: Visualizza immagine ampiezza
    try:
        import matplotlib.pyplot as plt
        print("\nAttempting to generate amplitude image preview...")
        plt.figure(figsize=(10, 8))

        # Calcola ampiezza e statistiche
        amplitude = np.abs(SlcPx)
        # Gestisci caso di immagine completamente nulla o con NaN/Inf
        valid_amplitude = amplitude[np.isfinite(amplitude)]
        if valid_amplitude.size == 0:
             print("Warning: Amplitude array contains no valid finite values. Cannot display.")
        else:
             mean_amp = np.mean(valid_amplitude)
             # Imposta vmax come in IDL, ma previeni vmax=0
             vmax_val = 15 * mean_amp if mean_amp > 1e-9 else 1.0
             # Usa percentile per vmin per migliore contrasto, escludendo zeri se sono molti
             vmin_val = np.percentile(valid_amplitude[valid_amplitude > 1e-9] if np.any(valid_amplitude > 1e-9) else valid_amplitude, 1)

             plt.imshow(amplitude, cmap='gray', vmin=vmin_val, vmax=vmax_val, aspect='auto') # aspect='auto' per riempire figura
             plt.colorbar(label='Amplitude')
             plt.title(f'SAR Image Amplitude (Python/Numba Output)\nvmax ≈ 15 * mean = {vmax_val:.2e}')
             plt.xlabel("Azimuth Samples")
             plt.ylabel("Range Samples")
             plt.tight_layout()
             img_filename = 'SlcPx_Python_Amplitude.png'
             plt.savefig(img_filename, dpi=150)
             print(f"Image preview saved to {img_filename}")
             # plt.show() # Descommenta per visualizzazione interattiva
    except ImportError:
        print("\nMatplotlib not installed. Skipping image preview generation.")
        print("To install: pip install matplotlib")
    except Exception as e:
        print(f"\nCould not create image preview due to an error: {e}")