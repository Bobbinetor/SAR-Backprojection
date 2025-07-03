"""
Visualizzatore di immagini SAR con supporto per diversi formati di file e filtri per lo speckle.

Filtri per lo speckle implementati:
- Lee: Filtro classico per immagini SAR, bilancia bene riduzione del rumore e conservazione dei dettagli
- Kuan: Simile a Lee ma con un modello di rumore diverso, spesso più efficace in alcune situazioni
- Frost: Considera la distanza spaziale dei pixel, adatto per aree con texture complesse
- Median: Semplice ed efficace per rimuovere lo speckle puntiforme, ma può sfumare i dettagli fini
- Bilateral: Preserva i bordi durante la riduzione del rumore, buono per mantenere la struttura
- Non-local means (nlm): Molto efficace ma computazionalmente intensivo, ottimo per dettagli fini
- Total variation (tv): Rimuove il rumore preservando i bordi netti, buono per immagini con strutture definite

Per controllare quali filtri applicare, modificare i parametri nella funzione main().
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import gc
import tifffile
from scipy import ndimage
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_nl_means, estimate_sigma

def load_npy_file(file_path, subset=False):
    """
    Carica un file .npy completo
    
    Parameters:
        file_path: percorso al file .npy
        subset: se True, estrae un sottoinsieme (ignorato in questa versione)
    
    Returns:
        complex_data: array complesso numpy
    """
    print(f"Caricamento del file .npy: {file_path}")
    complex_data = np.load(file_path)
    print(f"Dimensioni originali: {complex_data.shape}")
    return complex_data

def load_split_binary(file_path, shape, dtype=np.float32, order='F'):
    """
    Carica dati in formato split (prima parte reale, poi parte immaginaria)
    
    Parameters:
        file_path: percorso al file binario
        shape: forma dell'array completo (range, azimuth)
        dtype: tipo di dato (default: float32)
        order: ordine dei dati ('F' per Fortran, 'C' per C)
    
    Returns:
        complex_data: array complesso numpy
    """
    print(f"Caricamento file split binario: {file_path}")
    file_size = os.path.getsize(file_path)
    full_size = np.prod(shape) * 2 * np.dtype(dtype).itemsize
    
    if abs(file_size - full_size) > 1024:
        print(f"ATTENZIONE: La dimensione del file ({file_size} bytes) non corrisponde alle dimensioni specificate ({full_size} bytes)")
        print("Le dimensioni potrebbero essere invertite o errate.")
        
    # Carica tutto il file
    data = np.fromfile(file_path, dtype=dtype)
    
    # Verifica la dimensione
    half_point = len(data) // 2
    
    # Carica la parte reale e immaginaria
    real_part = data[:half_point].reshape(shape, order=order)
    imag_part = data[half_point:half_point*2].reshape(shape, order=order)
    
    # Combina in array complesso
    complex_data = real_part + 1j * imag_part
    
    return complex_data

def load_interleaved_binary(file_path, shape, dtype=np.float32, order='F'):
    """
    Carica dati in formato interleaved (alternando parte reale e immaginaria)
    
    Parameters:
        file_path: percorso al file binario
        shape: forma dell'array completo (range, azimuth)
        dtype: tipo di dato (default: float32)
        order: ordine dei dati ('F' per Fortran, 'C' per C)
    
    Returns:
        complex_data: array complesso numpy
    """
    print(f"Caricamento file interleaved binario: {file_path}")
    
    file_size = os.path.getsize(file_path)
    full_size = np.prod(shape) * 2 * np.dtype(dtype).itemsize
    
    if abs(file_size - full_size) > 1024:
        print(f"ATTENZIONE: La dimensione del file ({file_size} bytes) non corrisponde alle dimensioni specificate ({full_size} bytes)")
        print("Le dimensioni potrebbero essere invertite o errate.")
    
    # Per dataset enormi, utilizziamo un approccio a blocchi
    print("Caricamento in blocchi per dataset di grandi dimensioni...")
    
    # Dimensioni di un blocco: 1000 righe alla volta
    block_size = 1000  
    range_dim, azimuth_dim = shape
    
    # Prepara gli array di output
    real_data = np.zeros(shape, dtype=dtype)
    imag_data = np.zeros(shape, dtype=dtype)
    
    # Leggi i dati in blocchi
    with open(file_path, 'rb') as f:
        for block_start in range(0, range_dim, block_size):
            block_end = min(block_start + block_size, range_dim)
            current_block_size = block_end - block_start
            
            print(f"Caricamento blocco {block_start}-{block_end} di {range_dim}...")
            
            for i in range(block_start, block_end):
                # Calcola l'offset per questa riga
                if order == 'F':
                    # In ordine Fortran, dobbiamo leggere per colonne
                    # Questo è complicato per file interleaved, quindi leggiamo elemento per elemento
                    for j in range(azimuth_dim):
                        offset = (j * range_dim + i) * 2 * np.dtype(dtype).itemsize
                        f.seek(offset)
                        elem = np.fromfile(f, dtype=dtype, count=2)
                        if len(elem) == 2:
                            real_data[i, j] = elem[0]
                            imag_data[i, j] = elem[1]
                else:  # order == 'C'
                    # In ordine C, possiamo leggere una riga intera
                    row_offset = i * azimuth_dim * 2 * np.dtype(dtype).itemsize
                    f.seek(row_offset)
                    
                    # Leggi i dati interleaved per questa riga
                    row_data = np.fromfile(f, dtype=dtype, count=azimuth_dim*2)
                    
                    if len(row_data) == azimuth_dim*2:
                        real_data[i, :] = row_data[0::2]  # Elementi pari (0, 2, 4, ...)
                        imag_data[i, :] = row_data[1::2]  # Elementi dispari (1, 3, 5, ...)
    
    # Combina in array complesso
    complex_data = real_data + 1j * imag_data
    
    return complex_data

def process_and_save_image(complex_data, base_name, output_dir, scale_factor=15, apply_filters=True, filter_types=None):
    """
    Processa i dati complessi e salva l'immagine in vari formati e orientazioni
    
    Parameters:
        complex_data: array complesso numpy
        base_name: nome base per i file di output
        output_dir: directory di output
        scale_factor: fattore di scala per la visualizzazione
        apply_filters: se True, applica i filtri speckle
        filter_types: lista di filtri da applicare, default ['lee', 'median', 'frost', 'bilateral']
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtri speckle da applicare se richiesto
    if filter_types is None:
        filter_types = ['lee', 'median', 'frost', 'bilateral']
    
    # Lista di trasformazioni da applicare
    transformations = [
        ("original", lambda x: x),
        ("transposed", lambda x: x.T),
        ("flipped_v", lambda x: np.flipud(x)),
        ("flipped_h", lambda x: np.fliplr(x)),
        ("flipped_vh", lambda x: np.flipud(np.fliplr(x))),
        ("transposed_flipped_v", lambda x: np.flipud(x.T)),
        ("transposed_flipped_h", lambda x: np.fliplr(x.T)),
        ("transposed_flipped_vh", lambda x: np.flipud(np.fliplr(x.T)))
    ]
    
    for name, transform in transformations:
        try:
            print(f"Elaborazione della variante: {name}")
            
            # Applica la trasformazione
            transformed_data = transform(complex_data)
            
            # Calcola il modulo (valore assoluto)
            magnitude = np.abs(transformed_data)
            
            # Ottieni le dimensioni dell'immagine
            height, width = magnitude.shape
            print(f"Dimensioni immagine: {height}x{width}")
            
            # Scala l'immagine 
            valid_mag = magnitude[np.isfinite(magnitude)]
            if len(valid_mag) == 0:
                print("ERRORE: Nessun valore valido nell'immagine!")
                continue
            
            # --- INIZIO MODIFICA SCALING ---
            # Calcola vmin e vmax in modo più robusto usando percentili
            # Questo aiuta a gestire outlier e migliora il contrasto
            vmin_perc = 1  # Percentile per il minimo (es. 1%)
            vmax_perc = 99 # Percentile per il massimo (es. 99%)
            
            vmin = np.percentile(valid_mag, vmin_perc)
            vmax = np.percentile(valid_mag, vmax_perc)

            # Se vmax è molto vicino o uguale a vmin (es. immagine quasi costante o con pochi valori validi)
            if vmax <= vmin:
                # Fallback a media +/- deviazione standard o media * fattore se i percentili falliscono
                mean_val = np.mean(valid_mag)
                std_val = np.std(valid_mag)
                vmin = max(0, mean_val - 2 * std_val) # Evita vmin negativo
                vmax = mean_val + 2 * std_val
                if vmax <= vmin: # Se ancora problematico
                    vmax = vmin + 1e-5 # Un piccolo range per evitare divisione per zero
            
            print(f"Statistiche Magnitudine: Min={np.min(valid_mag):.2e}, Max={np.max(valid_mag):.2e}")
            print(f"Media={np.mean(valid_mag):.2e}, Mediana={np.median(valid_mag):.2e}")
            print(f"Scaling con: vmin ({vmin_perc}% percentile)={vmin:.2e}, vmax ({vmax_perc}% percentile)={vmax:.2e}")

            # Normalizza usando vmin e vmax calcolati
            # Prima scala i dati nell'intervallo [0, 1]
            normalized_float = (magnitude - vmin) / (vmax - vmin)
            # Quindi clippa e scala a [0, 255]
            normalized = np.clip(normalized_float * 255.0, 0, 255).astype(np.uint8)
            # --- FINE MODIFICA SCALING ---
            
            # Salva come PNG
            output_png = os.path.join(output_dir, f"{base_name}_{name}.png")
            Image.fromarray(normalized).save(output_png)
            print(f"PNG salvato in {output_png}")
            
            # Salva come TIFF
            output_tiff = os.path.join(output_dir, f"{base_name}_{name}.tiff")
            tifffile.imwrite(output_tiff, normalized, compression='lzw')
            print(f"TIFF salvato in {output_tiff}")
            
            # Applica filtri per lo speckle se richiesto
            if apply_filters:
                print("Applicazione filtri per lo speckle...")
                for filter_name in filter_types:
                    try:
                        print(f"Applicazione filtro {filter_name}...")
                        
                        # Applica il filtro mantenendo l'intervallo di valori
                        filtered_mag = apply_speckle_filter(magnitude.astype(np.float32), 
                                                        filter_type=filter_name, 
                                                        window_size=7)  # Finestra 7x7 è un buon default
                        
                        # Normalizza il risultato filtrato usando gli stessi parametri
                        filtered_normalized = np.clip((filtered_mag - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
                        
                        # Salva le versioni filtrate
                        filtered_png = os.path.join(output_dir, f"{base_name}_{name}_{filter_name}.png")
                        Image.fromarray(filtered_normalized).save(filtered_png)
                        print(f"PNG filtrato ({filter_name}) salvato in {filtered_png}")
                        
                        filtered_tiff = os.path.join(output_dir, f"{base_name}_{name}_{filter_name}.tiff")
                        tifffile.imwrite(filtered_tiff, filtered_normalized, compression='lzw')
                        print(f"TIFF filtrato ({filter_name}) salvato in {filtered_tiff}")
                        
                        # Libera memoria
                        del filtered_mag, filtered_normalized
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Errore durante l'applicazione del filtro {filter_name}: {e}")
                        continue
            
            # Libera memoria
            del transformed_data, magnitude, normalized
            gc.collect()
            
        except Exception as e:
            print(f"Errore durante l'elaborazione della variante {name}: {e}")
            continue

def visualize_sar_data(file_path, output_dir, data_type, shape=(4000, 32000), apply_filters=True, filter_types=None):
    """
    Visualizza i dati SAR da vari formati
    
    Parameters:
        file_path: percorso al file dati
        output_dir: directory di output
        data_type: tipo di dati ('npy', 'split', 'interleaved')
        shape: forma dell'array completo (range, azimuth)
        apply_filters: se True, applica i filtri speckle
        filter_types: lista di filtri da applicare, default ['lee', 'median', 'frost', 'bilateral']
    """
    # Nome file di output basato sul tipo di dato
    output_base = os.path.splitext(os.path.basename(file_path))[0]
    
    # Carica i dati in base al tipo
    if data_type == 'npy':
        complex_data = load_npy_file(file_path)
    elif data_type == 'split':
        complex_data = load_split_binary(file_path, shape, order='F')
    elif data_type == 'interleaved':
        complex_data = load_interleaved_binary(file_path, shape, order='F')
    else:
        print(f"Tipo di dati non supportato: {data_type}")
        return
    
    # Verifica che i dati siano validi
    if complex_data is None or complex_data.size == 0:
        print("Errore nel caricamento dei dati. Verifica il percorso del file e le dimensioni.")
        return
    
    # Processa e salva l'immagine in vari formati e orientazioni
    process_and_save_image(complex_data, output_base, output_dir, apply_filters=apply_filters, filter_types=filter_types)
    
    # Libera memoria
    del complex_data
    gc.collect()

def apply_speckle_filter(magnitude, filter_type='lee', window_size=5, **kwargs):
    """
    Applica filtri per ridurre lo speckle tipico delle immagini SAR
    
    Parameters:
        magnitude: array numpy con l'ampiezza dell'immagine SAR
        filter_type: tipo di filtro da applicare ('lee', 'kuan', 'frost', 'median', 'bilateral', 'nlm', 'tv')
        window_size: dimensione della finestra per i filtri locali
        **kwargs: parametri aggiuntivi specifici per ogni filtro
    
    Returns:
        filtered_magnitude: array numpy con l'ampiezza filtrata
    """
    print(f"Applicazione filtro speckle di tipo '{filter_type}' con finestra di {window_size}x{window_size}")
    
    # Controlla che i dati siano float per i calcoli
    if magnitude.dtype != np.float32 and magnitude.dtype != np.float64:
        magnitude = magnitude.astype(np.float32)
    
    # Gestisci eventuali valori non finiti (NaN, Inf)
    if np.any(~np.isfinite(magnitude)):
        print("ATTENZIONE: Valori non finiti trovati nell'immagine. Sostituzione con zeri.")
        magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Implementazione dei vari tipi di filtro
    if filter_type.lower() == 'median':
        # Filtro mediano - semplice ma efficace per lo speckle
        filtered = ndimage.median_filter(magnitude, size=window_size)
    
    elif filter_type.lower() == 'lee':
        # Filtro di Lee - classico per immagini SAR
        # Implementazione del filtro di Lee
        mean = ndimage.uniform_filter(magnitude, size=window_size)
        mean_sqr = ndimage.uniform_filter(magnitude**2, size=window_size)
        var = mean_sqr - mean**2
        
        # Stima della varianza del rumore dalla varianza globale o dal parametro fornito
        noise_var = kwargs.get('noise_var', np.mean(var) * 0.5)
        
        # Calcolo del rapporto segnale-rumore
        weights = var / (var + noise_var)
        weights = np.clip(weights, 0.0, 1.0)  # Assicura che i pesi siano nell'intervallo [0,1]
        
        # Applicazione del filtro
        filtered = mean + weights * (magnitude - mean)
    
    elif filter_type.lower() == 'kuan':
        # Filtro di Kuan - simile a Lee ma con modello di rumore moltiplicativo diverso
        mean = ndimage.uniform_filter(magnitude, size=window_size)
        mean_sqr = ndimage.uniform_filter(magnitude**2, size=window_size)
        var = mean_sqr - mean**2
        
        # Stima varianza del rumore
        noise_var = kwargs.get('noise_var', np.mean(var) * 0.5)
        
        # Calcolo varianza del segnale
        signal_var = var - mean**2 * noise_var
        signal_var = np.maximum(signal_var, 1e-10)  # Evita divisione per zero
        
        # Calcolo dei pesi
        weights = signal_var / (signal_var + noise_var * mean**2)
        weights = np.clip(weights, 0.0, 1.0)
        
        # Applicazione del filtro
        filtered = mean + weights * (magnitude - mean)
    
    elif filter_type.lower() == 'frost':
        # Filtro di Frost - considera la distanza spaziale dei pixel nella finestra
        # Parametro di sensibilità del filtro di Frost
        K = kwargs.get('K', 2.0)
        
        # Creazione di un kernel gaussiano
        x, y = np.meshgrid(np.arange(-window_size//2+1, window_size//2+1),
                           np.arange(-window_size//2+1, window_size//2+1))
        distance = np.sqrt(x**2 + y**2)
        
        # Inizializza l'array filtrato
        filtered = np.zeros_like(magnitude)
        
        # Calcola media e varianza locali
        mean = ndimage.uniform_filter(magnitude, size=window_size)
        mean_sqr = ndimage.uniform_filter(magnitude**2, size=window_size)
        var = mean_sqr - mean**2
        
        # Applica il filtro Frost
        for i in range(window_size//2, magnitude.shape[0]-window_size//2):
            for j in range(window_size//2, magnitude.shape[1]-window_size//2):
                # Estrai la finestra locale
                window = magnitude[i-window_size//2:i+window_size//2+1, 
                                   j-window_size//2:j+window_size//2+1]
                
                # Calcola il coefficiente di variazione locale
                local_mean = mean[i, j]
                local_var = var[i, j]
                
                if local_mean > 0:
                    cv = local_var / (local_mean**2)
                else:
                    cv = 0
                
                # Calcola i pesi in base alla distanza e al coefficiente di variazione
                weights = np.exp(-K * cv * distance)
                weights = weights / np.sum(weights)  # Normalizza i pesi
                
                # Applica i pesi alla finestra
                filtered[i, j] = np.sum(window * weights)
    
    elif filter_type.lower() == 'bilateral':
        # Filtro bilaterale - preserva i bordi
        sigma_color = kwargs.get('sigma_color', 0.1)
        sigma_spatial = kwargs.get('sigma_spatial', 1.0)
        
        # Normalizza i dati per il filtro bilaterale
        normalized = magnitude / np.max(magnitude)
        
        # Applica il filtro bilaterale
        filtered = denoise_bilateral(normalized, 
                                    sigma_color=sigma_color,
                                    sigma_spatial=sigma_spatial,
                                    win_size=window_size)
        
        # Ripristina la scala originale
        filtered = filtered * np.max(magnitude)
    
    elif filter_type.lower() == 'nlm' or filter_type.lower() == 'non-local-means':
        # Filtro Non-Local Means - molto efficace ma computazionalmente intensivo
        # Stima il livello di rumore se non fornito
        sigma = kwargs.get('sigma', estimate_sigma(magnitude, multichannel=False))
        h = kwargs.get('h', 0.8 * sigma)  # Parametro di filtro
        patch_size = kwargs.get('patch_size', 5)  # Dimensione patch
        patch_distance = kwargs.get('patch_distance', 7)  # Distanza massima di ricerca
        
        # Applica il filtro NLM
        filtered = denoise_nl_means(magnitude, 
                                  h=h,
                                  sigma=sigma,
                                  fast_mode=True,
                                  patch_size=patch_size,
                                  patch_distance=patch_distance)
    
    elif filter_type.lower() == 'tv' or filter_type.lower() == 'total-variation':
        # Filtro Total Variation - buono per rimuovere rumore preservando bordi
        weight = kwargs.get('weight', 0.1)
        
        # Normalizza i dati per il filtro TV
        normalized = magnitude / np.max(magnitude)
        
        # Applica il filtro TV
        filtered = denoise_tv_chambolle(normalized, weight=weight)
        
        # Ripristina la scala originale
        filtered = filtered * np.max(magnitude)
    
    else:
        print(f"Tipo di filtro '{filter_type}' non riconosciuto. Usa uno tra: 'lee', 'kuan', 'frost', 'median', 'bilateral', 'nlm', 'tv'")
        filtered = magnitude  # Ritorna l'immagine originale se il filtro non è riconosciuto
    
    return filtered

if __name__ == "__main__":
    import argparse
    
    # Configurazione del parser degli argomenti
    parser = argparse.ArgumentParser(description='Visualizzatore di immagini SAR con filtri per lo speckle')
    parser.add_argument('--input-dir', type=str, default=".", help='Directory contenente i file di input')
    parser.add_argument('--output-dir', type=str, default="output_nuovo", help='Directory dove salvare le immagini')
    parser.add_argument('--shape', type=int, nargs=2, default=[4000, 250], help='Dimensioni del dataset (Range, Azimuth)')
    parser.add_argument('--filters', type=str, nargs='+', default=['lee', 'kuan', 'median', 'bilateral'],
                        help='Filtri da applicare (lee, kuan, frost, median, bilateral, nlm, tv)')
    parser.add_argument('--no-filters', action='store_true', help='Disabilita l\'applicazione dei filtri')
    
    # Analizza gli argomenti
    args = parser.parse_args()
    
    # Directory di input e output
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Dimensioni del dataset (Range, Azimuth)
    actual_shape = tuple(args.shape)
    print(f"Dimensioni dataset: {actual_shape}")
    
    # File paths
    npy_file = os.path.join(input_dir, "SlcCuPy_SuperOptimized.npy")
    split_file = os.path.join(input_dir, "SlcCuPy_SuperOptimized_SplitF32.dat")
    interleaved_file = os.path.join(input_dir, "SlcCuPy_SuperOptimized_InterleavedF32.dat")
    truth_file = os.path.join(input_dir, "truth_completa.dat")

    print("\nSeleziona quale set di file visualizzare:")
    print("  1) Output generati dallo script (SlcCuPy_SuperOptimized.*)")
    print("  2) File di verità (truth_completa.dat)")
    selection = input("Inserisci 1 o 2 e premi invio: ").strip()

    if selection == '1':
        files_to_process = [
            (npy_file, 'npy'),
            (split_file, 'split'),
            (interleaved_file, 'interleaved')
        ]
        shape_to_use = actual_shape
    elif selection == '2':
        files_to_process = [(truth_file, 'split')]
        # Imposta la shape corretta per truth_completa.dat
        shape_to_use = (4000, 32000)
    else:
        print("Selezione non valida. Esco.")
        exit(1)

    # Processa automaticamente tutti i file disponibili
    print("Elaborazione automatica di tutti i formati di file disponibili...")
    for file_path, file_type in files_to_process:
        if os.path.exists(file_path):
            print(f"\n============================================")
            print(f"Elaborazione {file_type}: {file_path}")
            print(f"============================================")
            try:
                apply_filters = not args.no_filters
                filter_types = args.filters if apply_filters else None
                if apply_filters:
                    print(f"Filtri che verranno applicati: {filter_types}")
                else:
                    print("Applicazione filtri disabilitata")
                visualize_sar_data(file_path, output_dir, file_type, shape_to_use, 
                                  apply_filters=apply_filters, filter_types=filter_types)
                print(f"Elaborazione di {file_path} completata con successo.")
            except Exception as e:
                print(f"Errore durante l'elaborazione di {file_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File non trovato: {file_path}")
    print("\nProcesso completato. Controlla la cartella", output_dir, "per tutte le varianti delle immagini.")