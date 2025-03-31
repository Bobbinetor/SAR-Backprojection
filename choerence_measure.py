import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Configurazione ---
FILE_IDL_OUTPUT = 'vera_matrice.dat'
#FILE_PYTHON_OUTPUT = 'SlcPx_Python_Corrected_InterleavedF32.dat'
#FILE_PYTHON_OUTPUT = 'SlcPx_Python_Corrected_SplitF32.dat'
FILE_PYTHON_OUTPUT = 'SlcPx_Python_Corrected.npy' # Usa il file .npy per precisione
#FILE_PYTHON_OUTPUT = 'SlcPx_Python_Corrected_RoundFixed.npy' # Usa il file .dat per flessibilità
OUTPUT_PLOT_FILE = 'comparison_idl_vs_python.png'

# Dimensioni attese dell'immagine (come nello script di focalizzazione)
DIM_RG = 4000 # Dimensione Range (righe)
DIM_AZ = 250  # Dimensione Azimuth (colonne)

# Soglia di magnitudine sotto la quale la fase è considerata inaffidabile
# (espressa come frazione della magnitudine massima IDL)
PHASE_MAGNITUDE_THRESHOLD_FRACTION = 1e-5
# --- Fine Configurazione ---

def load_idl_dat(filename, dim_rg, dim_az, dtype=np.float32):
    """
    Carica dati complessi da un file .dat binario, assumendo coppie
    interleaved (real1, imag1, ...) del tipo specificato (default float32).
    Utilizza ordine Fortran ('F') per il reshape.
    """
    print(f"Loading IDL data from: {filename} (assuming interleaved {dtype.__name__})")
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)

    try:
        # Leggi i dati binari grezzi
        data_raw = np.fromfile(filename, dtype=dtype)

        # Verifica la dimensione
        expected_elements = 2 * dim_rg * dim_az
        if data_raw.size != expected_elements:
            print(f"Error: File size mismatch for {filename}.")
            print(f"Expected {expected_elements} elements ({dtype.__name__}), found {data_raw.size}")
            print("Possible causes: Wrong dimensions, wrong dtype (e.g., float64?), wrong format (split vs interleaved?).")
            sys.exit(1)

        # Estrai parte reale e immaginaria
        real_part = data_raw[0::2]
        imag_part = data_raw[1::2]

        # Ricostruisci l'array complesso
        if dtype == np.float32:
            complex_data = real_part + 1j * imag_part # Risulta complex64
        elif dtype == np.float64:
            complex_data = real_part + 1j * imag_part # Risulta complex128
        else:
            raise ValueError(f"Unsupported dtype for complex reconstruction: {dtype}")

        # Riforma nelle dimensioni corrette con ordine Fortran
        complex_data_2d = complex_data.reshape((dim_rg, dim_az), order='F')
        print(f"IDL data loaded successfully. Shape: {complex_data_2d.shape}, Dtype: {complex_data_2d.dtype}")
        return complex_data_2d

    except Exception as e:
        print(f"An error occurred while loading {filename}: {e}")
        sys.exit(1)

def load_python_data(filename, dim_rg, dim_az, dtype=np.float32):
    """Carica dati complessi da file .npy o .dat, gestendo formati interleaved e split."""
    print(f"Loading Python data from: {filename}")
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)

    name, ext = os.path.splitext(filename)

    try:
        if ext.lower() == '.npy':
            data = np.load(filename)
            print(f"Python data loaded successfully (NumPy .npy). Shape: {data.shape}, Dtype: {data.dtype}")
            return data
        elif ext.lower() == '.dat':
            # Prova prima a caricare come interleaved float32 (come IDL)
            expected_elements_interleaved = 2 * dim_rg * dim_az
            try:
                data_raw = np.fromfile(filename, dtype=dtype)
                if data_raw.size == expected_elements_interleaved:
                    print(f"Assuming .dat format is interleaved {dtype.__name__}.")
                    real_part = data_raw[0::2]
                    imag_part = data_raw[1::2]
                    if dtype == np.float32:
                        complex_data = real_part + 1j * imag_part # Risulta complex64
                    elif dtype == np.float64:
                        complex_data = real_part + 1j * imag_part # Risulta complex128
                    else:
                        raise ValueError(f"Unsupported dtype for complex reconstruction: {dtype}")
                    complex_data_2d = complex_data.reshape((dim_rg, dim_az), order='F')
                    print(f"Python data loaded successfully (interleaved .dat). Shape: {complex_data_2d.shape}, Dtype: {complex_data_2d.dtype}")
                    return complex_data_2d
                else:
                    print(f"Interleaved .dat file size mismatch (expected {expected_elements_interleaved}, found {data_raw.size}). Trying split format.")
                    # Se la dimensione non corrisponde all'interleaved, prova come split float32
                    expected_elements_split = dim_rg * dim_az
                    if data_raw.size == 2 * expected_elements_split:
                        print(f"Assuming .dat format is split {dtype.__name__}.")
                        real_part = data_raw[:expected_elements_split].reshape((dim_rg, dim_az), order='F')
                        imag_part = data_raw[expected_elements_split:].reshape((dim_rg, dim_az), order='F')
                        if dtype == np.float32:
                            complex_data = real_part + 1j * imag_part # Risulta complex64
                        elif dtype == np.float64:
                            complex_data = real_part + 1j * imag_part # Risulta complex128
                        else:
                            raise ValueError(f"Unsupported dtype for complex reconstruction: {dtype}")
                        print(f"Python data loaded successfully (split .dat). Shape: {complex_data.shape}, Dtype: {complex_data.dtype}")
                        return complex_data
                    else:
                        raise ValueError(f"Split .dat file size mismatch (expected {2 * expected_elements_split}, found {data_raw.size}).")
            except Exception as e_interleaved:
                print(f"Error loading as interleaved: {e_interleaved}")
                print("Attempting to load as split format...")
                try:
                    data_raw = np.fromfile(filename, dtype=dtype)
                    expected_elements_split = dim_rg * dim_az
                    if data_raw.size == 2 * expected_elements_split:
                        print(f"Assuming .dat format is split {dtype.__name__}.")
                        real_part = data_raw[:expected_elements_split].reshape((dim_rg, dim_az), order='F')
                        imag_part = data_raw[expected_elements_split:].reshape((dim_rg, dim_az), order='F')
                        if dtype == np.float32:
                            complex_data = real_part + 1j * imag_part # Risulta complex64
                        elif dtype == np.float64:
                            complex_data = real_part + 1j * imag_part # Risulta complex128
                        else:
                            raise ValueError(f"Unsupported dtype for complex reconstruction: {dtype}")
                        print(f"Python data loaded successfully (split .dat). Shape: {complex_data.shape}, Dtype: {complex_data.dtype}")
                        return complex_data
                    else:
                        raise ValueError(f"Split .dat file size mismatch (expected {2 * expected_elements_split}, found {data_raw.size}).")
                except Exception as e_split:
                    raise ValueError(f"Failed to load .dat file as either interleaved or split: Interleaved error: {e_interleaved}, Split error: {e_split}")

        else:
            raise ValueError(f"Unsupported file extension for Python data: {ext}")

    except Exception as e:
        print(f"An error occurred while loading {filename}: {e}")
        sys.exit(1)

def wrap_phase(phase_diff):
    """Riporta la differenza di fase nell'intervallo [-pi, pi]."""
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    print("--- Inizio Comparazione Output SAR ---")

    # 1. Carica i dati
    # Prova a caricare IDL come float32 interleaved, altrimenti commenta/modifica dtype
    data_idl = load_idl_dat(FILE_IDL_OUTPUT, DIM_RG, DIM_AZ, dtype=np.float32)
    # Carica i dati Python, ora la funzione gestisce sia .npy che .dat (interleaved e split)
    data_py = load_python_data(FILE_PYTHON_OUTPUT, DIM_RG, DIM_AZ, dtype=np.float32)

    # 2. Verifica preliminare dimensioni
    if data_idl.shape != data_py.shape:
        print("Error: Le dimensioni delle matrici IDL e Python non corrispondono!")
        print(f"IDL shape: {data_idl.shape}, Python shape: {data_py.shape}")
        sys.exit(1)
    if data_idl.shape != (DIM_RG, DIM_AZ):
        print(f"Warning: Loaded data shape {data_idl.shape} differs from expected ({DIM_RG}, {DIM_AZ})")

    # Converti IDL a complex128 per confronto a precisione maggiore se necessario
    if data_idl.dtype != np.complex128:
        print(f"Converting IDL data from {data_idl.dtype} to complex128 for comparison.")
        data_idl = data_idl.astype(np.complex128)

    print(f"\nComparing data: IDL({data_idl.dtype}) vs Python({data_py.dtype})")

    # 3. Calcola Magnitudine e Fase
    print("Calculating magnitude and phase...")
    mag_idl = np.abs(data_idl)
    mag_py = np.abs(data_py)

    phase_idl = np.angle(data_idl)
    phase_py = np.angle(data_py)

    # 4. Calcola Differenze
    print("Calculating differences...")
    # Differenza Magnitudine (signed e absolute)
    diff_mag = mag_py - mag_idl
    abs_diff_mag = np.abs(diff_mag)

    # Differenza Fase (raw e wrapped in [-pi, pi])
    diff_phase_raw = phase_py - phase_idl
    diff_phase_wrapped = wrap_phase(diff_phase_raw)
    abs_diff_phase_wrapped = np.abs(diff_phase_wrapped)

    # 5. Crea maschera per fase valida (dove magnitudine non è troppo bassa)
    mag_idl_max = np.max(mag_idl) if np.any(mag_idl) else 1.0 # Evita divisione per zero
    phase_threshold = PHASE_MAGNITUDE_THRESHOLD_FRACTION * mag_idl_max
    valid_phase_mask = (mag_idl > phase_threshold) & (mag_py > phase_threshold)
    num_valid_phase_pixels = np.sum(valid_phase_mask)
    print(f"Phase comparison valid for {num_valid_phase_pixels}/{data_idl.size} pixels (Magnitude > {phase_threshold:.2e})")

    # 6. Calcola Statistiche
    print("\n--- Statistiche Differenza Magnitudine ---")
    mean_abs_diff_mag = np.mean(abs_diff_mag)
    std_abs_diff_mag = np.std(abs_diff_mag)
    max_abs_diff_mag = np.max(abs_diff_mag)
    mean_signed_diff_mag = np.mean(diff_mag)
    std_signed_diff_mag = np.std(diff_mag)
    rmse_mag = np.sqrt(np.mean(diff_mag**2)) # Root Mean Square Error

    print(f"Mean Absolute Difference:   {mean_abs_diff_mag:.4e}")
    print(f"Std Dev Abs Difference:     {std_abs_diff_mag:.4e}")
    print(f"Max Absolute Difference:    {max_abs_diff_mag:.4e}")
    print(f"Mean Signed Difference:     {mean_signed_diff_mag:.4e} (Py - IDL)")
    print(f"Std Dev Signed Difference:  {std_signed_diff_mag:.4e}")
    print(f"Root Mean Square Error:     {rmse_mag:.4e}")

    print("\n--- Statistiche Differenza Fase (pixel validi) ---")
    if num_valid_phase_pixels > 0:
        mean_abs_phase_diff = np.mean(abs_diff_phase_wrapped[valid_phase_mask])
        std_abs_phase_diff = np.std(abs_diff_phase_wrapped[valid_phase_mask])
        max_abs_phase_diff = np.max(abs_diff_phase_wrapped[valid_phase_mask])
        mean_signed_phase_diff = np.mean(diff_phase_wrapped[valid_phase_mask])
        std_signed_phase_diff = np.std(diff_phase_wrapped[valid_phase_mask])

        print(f"Mean Absolute Wrapped Diff: {mean_abs_phase_diff:.4f} rad ({np.degrees(mean_abs_phase_diff):.4f} deg)")
        print(f"Std Dev Abs Wrapped Diff:   {std_abs_phase_diff:.4f} rad ({np.degrees(std_abs_phase_diff):.4f} deg)")
        print(f"Max Absolute Wrapped Diff:  {max_abs_phase_diff:.4f} rad ({np.degrees(max_abs_phase_diff):.4f} deg)")
        print(f"Mean Signed Wrapped Diff:   {mean_signed_phase_diff:.4f} rad ({np.degrees(mean_signed_phase_diff):.4f} deg) (Py - IDL)")
        print(f"Std Dev Signed Wrapped Diff:{std_signed_phase_diff:.4f} rad ({np.degrees(std_signed_phase_diff):.4f} deg)")
    else:
        print("Nessun pixel valido per il confronto di fase.")

    # 7. Visualizzazione
    print(f"\nGenerating comparison plot ({OUTPUT_PLOT_FILE})...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        fig.suptitle('Confronto Output IDL vs Python', fontsize=16)
        plt.subplots_adjust(hspace=0.3, wspace=0.1) # Aggiusta spazi

        # Colormap comune per magnitudine
        valid_mags = mag_idl[np.isfinite(mag_idl)]
        if valid_mags.size > 0:
            # Usa percentili per una migliore scala di colori
            vmin_mag = np.percentile(valid_mags[valid_mags > 1e-9] if np.any(valid_mags > 1e-9) else valid_mags, 1)
            vmax_mag = np.percentile(valid_mags, 99)
        else:
             vmin_mag, vmax_mag = 0, 1


        # Riga 1: Magnitudine
        im = axes[0, 0].imshow(mag_idl, cmap='gray', vmin=vmin_mag, vmax=vmax_mag, aspect='auto')
        axes[0, 0].set_title(f'IDL Magnitudine ({data_idl.dtype})')
        axes[0, 0].set_ylabel('Range')
        # fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im = axes[0, 1].imshow(mag_py, cmap='gray', vmin=vmin_mag, vmax=vmax_mag, aspect='auto')
        axes[0, 1].set_title(f'Python Magnitudine ({data_py.dtype})')
        # fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im = axes[0, 2].imshow(abs_diff_mag, cmap='magma', aspect='auto') # 'magma' evidenzia differenze
        axes[0, 2].set_title(f'|Diff. Magnitudine| (Max: {max_abs_diff_mag:.2e})')
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Riga 2: Fase
        # Usa colormap ciclico per fase
        im = axes[1, 0].imshow(phase_idl, cmap='hsv', vmin=-np.pi, vmax=np.pi, aspect='auto')
        axes[1, 0].set_title('IDL Fase')
        axes[1, 0].set_xlabel('Azimuth')
        axes[1, 0].set_ylabel('Range')
        # fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi], format='%.2f')

        im = axes[1, 1].imshow(phase_py, cmap='hsv', vmin=-np.pi, vmax=np.pi, aspect='auto')
        axes[1, 1].set_title('Python Fase')
        axes[1, 1].set_xlabel('Azimuth')
        # fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi], format='%.2f')

        # Mostra differenza di fase assoluta *solo* dove la magnitudine è sufficiente
        phase_diff_display = np.full(abs_diff_phase_wrapped.shape, np.nan) # Inizia con NaN
        if num_valid_phase_pixels > 0:
            phase_diff_display[valid_phase_mask] = abs_diff_phase_wrapped[valid_phase_mask]
            vmax_ph_diff = np.percentile(abs_diff_phase_wrapped[valid_phase_mask], 99.5) # Scala basata su 99.5 percentile
        else:
             vmax_ph_diff = np.pi

        im = axes[1, 2].imshow(phase_diff_display, cmap='viridis', vmin=0, vmax=vmax_ph_diff, aspect='auto') # 'viridis' per diff assoluta
        axes[1, 2].set_title(f'|Diff. Fase Wrapped| (Max Valid: {max_abs_phase_diff:.2f} rad)')
        axes[1, 2].set_xlabel('Azimuth')
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # Rimuovi etichette y per plot centrali e a destra per pulizia
        for ax in axes[0, 1:]: plt.setp(ax.get_yticklabels(), visible=False)
        for ax in axes[1, 1:]: plt.setp(ax.get_yticklabels(), visible=False)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta layout per titolo generale
        plt.savefig(OUTPUT_PLOT_FILE, dpi=150)
        print(f"Comparison plot saved to {OUTPUT_PLOT_FILE}")
        # plt.show() # Descommenta per mostrare il plot interattivamente

    except ImportError:
        print("\nMatplotlib non trovato. Impossibile generare il plot.")
        print("Installa con: pip install matplotlib")
    except Exception as e:
        print(f"\nErrore durante la generazione del plot: {e}")

    print("\n--- Comparazione Completata ---")