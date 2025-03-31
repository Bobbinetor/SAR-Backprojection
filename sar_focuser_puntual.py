import numpy as np
import time
import os
from numba import jit, prange
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.style import Style
from rich.prompt import Confirm

# Create Rich console
console = Console()

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
    console.print(Panel.fit(
        "[bold cyan]FOCALIZADOR VERSIÓN PUNTUAL[/bold cyan] [green](Precisione Corretta)[/green]", 
        box=box.ROUNDED, 
        style="blue"
    ))
    
    t1 = time.time()

    # Ottieni dimensioni come in IDL
    # IDL: siz = size(raw_data) & Nra = siz(1) & Naz = siz(2)
    if Raw_Data.ndim != 2:
        console.print("[bold red]Errore:[/bold red] Raw_Data deve essere un array 2D", style="red")
        raise ValueError("Raw_Data must be a 2D array")
    Nra, Naz = Raw_Data.shape

    # IDL: siz = size(xxg1) & dimx = siz(1) & dimy = siz(2)
    if xxg1.ndim != 2:
        console.print("[bold red]Errore:[/bold red] xxg1 deve essere un array 2D", style="red")
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
    with console.status("[blue]Appiattimento degli array di input (order='F')...[/blue]", spinner="dots"):
        xxg1_flat = xxg1.astype(np.float64).flatten(order='F')
        yyg1_flat = yyg1.astype(np.float64).flatten(order='F')
        zzg1_flat = zzg1.astype(np.float64).flatten(order='F')
        mask_flat = mask.astype(np.uint8).flatten(order='F')
        index_flat = index.astype(np.int32).flatten(order='F')
        Sint_flat = Sint.astype(np.int32).flatten(order='F')
    console.print("[green]✓[/green] Appiattimento completato", style="green")

    # Applica maschera per valori Sint non validi (come in IDL)
    # IDL: no_valid=where(sint eq -9999)
    # IDL: if (no_valid[0] ne -1) then begin mask[no_valid] = 0l; endif
    # Lavora sulla versione flat per coerenza con il loop
    no_valid_indices = np.where(Sint_flat == -9999)[0]
    if no_valid_indices.size > 0:
        console.print(f"[yellow]⚠[/yellow] Mascheramento di [cyan]{no_valid_indices.size}[/cyan] punti con Sint == -9999", style="yellow")
        mask_flat[no_valid_indices] = 0

    # Costanti IDL (probabilmente 0)
    ind_line_ra_min = 0
    index_min = 0

    # Gestione Checkpoint
    checkpoint_file = 'focalizador_px_checkpoint.npz'
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            with console.status(f"[blue]Caricamento checkpoint da {checkpoint_file}...[/blue]", spinner="dots"):
                with np.load(checkpoint_file) as checkpoint:
                    # Verifica che le dimensioni corrispondano
                    if checkpoint['focused_sample'].shape == focused_sample.shape:
                        focused_sample = checkpoint['focused_sample']
                        start_idx = int(checkpoint['next_idx']) # Assicura sia un intero Python
                        console.print(f"[green]✓[/green] Ripresa dell'elaborazione dall'indice [cyan]{start_idx}[/cyan]/[cyan]{dim}[/cyan]", style="green")
                    else:
                        console.print("[yellow]⚠[/yellow] Dimensioni del checkpoint non corrispondenti. Avvio da zero.", style="yellow")
                        start_idx = 0
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Errore nel caricamento del checkpoint: {e}. Avvio da zero.", style="yellow")
            start_idx = 0

    # IDL: for ii=0,n_elements(xxg1)-1l do begin
    # Loop principale pixel-per-pixel
    batch_size = 5000  # Aggiorna checkpoint ogni tot pixel
    print_interval = 100 # Ogni quanti pixel stampare progresso

    console.print(Panel("[bold blue]Avvio del processo di focalizzazione punto per punto[/bold blue]", 
                        style="blue", box=box.ROUNDED))

    # Utilizzo Rich Progress per una barra di avanzamento avanzata
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[blue]Elaborazione pixel...[/blue]", total=dim, start=start_idx)
        last_batch_time = time.time()
        processed_since_last = 0
        
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

            # Conta i pixel elaborati dall'ultimo aggiornamento
            processed_since_last += 1
            
            # Aggiorna la barra di progresso
            current_time = time.time()
            if ii == start_idx or (ii + 1) % print_interval == 0 or ii == dim - 1:
                elapsed_since_last = current_time - last_batch_time
                speed = processed_since_last / elapsed_since_last if elapsed_since_last > 0 else 0
                progress.update(task, completed=ii + 1, refresh=True, 
                                description=f"[blue]Elaborazione pixel[/blue] [cyan]{ii + 1}[/cyan]/[cyan]{dim}[/cyan] [yellow]({speed:.2f} px/s)[/yellow]")
                last_batch_time = current_time
                processed_since_last = 0

            # Salva checkpoint
            if (ii + 1) % batch_size == 0 and ii != dim - 1:
                try:
                    # Aggiorna temporaneamente la descrizione per indicare il salvataggio
                    progress.update(task, description=f"[yellow]Salvataggio checkpoint all'indice {ii+1}...[/yellow]")
                    
                    # Salva il checkpoint senza utilizzare console.status
                    np.savez(checkpoint_file,
                             focused_sample=focused_sample,
                             next_idx=np.int64(ii + 1)) # Salva indice successivo
                    
                    # Ripristina la descrizione originale
                    progress.update(task, description=f"[blue]Elaborazione pixel[/blue] [cyan]{ii + 1}[/cyan]/[cyan]{dim}[/cyan] [yellow]({speed:.2f} px/s)[/yellow]")
                except Exception as e:
                    # Indica l'errore nella descrizione invece di usare console.print
                    progress.update(task, description=f"[red]Errore checkpoint[/red] [cyan]{ii + 1}[/cyan]/[cyan]{dim}[/cyan] [yellow]({speed:.2f} px/s)[/yellow]")
                    # Continua l'elaborazione

    console.print("[bold green]✓ Loop di focalizzazione completato[/bold green]", style="green")

    # Riforma l'array risultato nelle dimensioni 2D originali, usando ordine Fortran
    # IDL: ;focused_sample = reform(focused_sample,dimx,dimy)
    with console.status("[blue]Ridimensionamento dell'array di output in 2D (order='F')...[/blue]", spinner="dots"):
        focused_sample_2d = np.reshape(focused_sample, (dimx, dimy), order='F')
    console.print("[green]✓[/green] Ridimensionamento completato", style="green")

    # Calcola tempo trascorso
    t2 = time.time()
    elapsed = t2 - t1
    elapsed_min = int(elapsed // 60)
    elapsed_sec = elapsed % 60
    
    time_text = Text()
    time_text.append("Tempo trascorso: ", style="cyan")
    if elapsed_min > 0:
        time_text.append(f"{elapsed_min} min ", style="green")
    time_text.append(f"{elapsed_sec:.2f} sec", style="green")
    console.print(time_text)

    # Rimuovi file checkpoint se esiste
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            console.print("[green]✓[/green] File di checkpoint rimosso con successo", style="green")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Impossibile rimuovere il file di checkpoint: {e}", style="yellow")

    return focused_sample_2d

# ###########################################################################
# # Script Principale di Esecuzione (come nel file IDL di test)
# ###########################################################################
if __name__ == "__main__":
    start_time_total = time.time()

    # Header accattivante con rich
    console.print(Panel(
        "[bold cyan]SAR Point-based Back-projection Focusing[/bold cyan]\n[green]Python/Numba Implementation - Precision Corrected Version[/green]", 
        box=box.DOUBLE, 
        style="blue",
        width=60,
        expand=False
    ))

    # Dimensioni come definite nel test IDL
    DimRg = 4000
    DimAz = 250
    Nra_raw = 4762 # Dimensione Range RawData
    Naz_raw = 40001 # Dimensione Azimuth RawData

    # Tabella di dimensioni
    dim_table = Table(title="Dimensioni degli Array", box=box.ROUNDED)
    dim_table.add_column("Tipo", style="cyan")
    dim_table.add_column("Range", style="green")
    dim_table.add_column("Azimuth", style="green")
    dim_table.add_row("Griglia di Output", str(DimRg), str(DimAz))
    dim_table.add_row("Dati Grezzi", str(Nra_raw), str(Naz_raw))
    console.print(dim_table)

    # Inizializzazione array con tipi NumPy corrispondenti a IDL
    # Usiamo float64 per double, int32 per long, uint8 per byte, complex64 per complex
    console.print(Panel("[bold blue]Inizializzazione degli Array NumPy[/bold blue]", style="blue", box=box.ROUNDED))
    try:
        with console.status("[blue]Allocazione memoria...[/blue]", spinner="dots"):
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
        console.print("[green]✓[/green] Array inizializzati con successo", style="green")
    except MemoryError:
        memory_gb = ((DimRg*DimAz*3*8) + (DimRg*DimAz*4*2) + (Nra_raw*Naz_raw*8) + (Naz_raw*3*8)) / (1024**3)
        console.print(f"[bold red]Errore:[/bold red] Memoria insufficiente per inizializzare gli array.", style="red")
        console.print(f"Tentativo di allocare circa [cyan]{memory_gb:.2f}[/cyan] GB", style="yellow")
        exit()

    console.print(Panel("[bold blue]FASE 1: CARICAMENTO DATI[/bold blue]", style="blue", box=box.DOUBLE))
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
    file_table = Table(title="Verifica File Necessari", box=box.ROUNDED)
    file_table.add_column("File", style="cyan")
    file_table.add_column("Stato", justify="center")
    
    for fname in required_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            file_table.add_row(fname, "[green]✓[/green]")
        else:
            file_table.add_row(fname, "[red]✗[/red]")
            all_files_found = False
    
    console.print(file_table)
    
    if not all_files_found:
        console.print("[bold red]Errore:[/bold red] Assicurarsi che tutti i file .dat siano nella directory corretta.", style="red")
        exit()
    else:
        console.print("[green]✓[/green] Tutti i file necessari sono presenti", style="green")

    # Carica dati da file binari (stile IDL readu)
    try:
        # Utilizziamo la barra di avanzamento per il caricamento file
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            load_task = progress.add_task("[blue]Caricamento file DEM...[/blue]", total=12)
            
            # DEM X
            progress.update(load_task, description="[blue]Caricamento DEM X...[/blue]")
            xxg1 = np.fromfile(os.path.join(data_dir, 'Dem_x_Dbl4000x250.dat'), dtype=np.float64).reshape((DimRg, DimAz), order='F')
            progress.advance(load_task)
            
            # DEM Y
            progress.update(load_task, description="[blue]Caricamento DEM Y...[/blue]")
            yyg1 = np.fromfile(os.path.join(data_dir, 'Dem_y_Dbl4000x250.dat'), dtype=np.float64).reshape((DimRg, DimAz), order='F')
            progress.advance(load_task)
            
            # DEM Z
            progress.update(load_task, description="[blue]Caricamento DEM Z...[/blue]")
            zzg1 = np.fromfile(os.path.join(data_dir, 'Dem_z_Dbl4000x250.dat'), dtype=np.float64).reshape((DimRg, DimAz), order='F')
            progress.advance(load_task)
            
            # Trajectory X
            progress.update(load_task, description="[blue]Caricamento Traiettoria X...[/blue]")
            wgsx1 = np.fromfile(os.path.join(data_dir, 'Traiett_x_Dbl40001.dat'), dtype=np.float64)
            progress.advance(load_task)
            
            # Trajectory Y
            progress.update(load_task, description="[blue]Caricamento Traiettoria Y...[/blue]")
            wgsy1 = np.fromfile(os.path.join(data_dir, 'Traiett_y_Dbl40001.dat'), dtype=np.float64)
            progress.advance(load_task)
            
            # Trajectory Z
            progress.update(load_task, description="[blue]Caricamento Traiettoria Z...[/blue]")
            wgsz1 = np.fromfile(os.path.join(data_dir, 'Traiett_z_Dbl40001.dat'), dtype=np.float64)
            progress.advance(load_task)
            
            # Index
            progress.update(load_task, description="[blue]Caricamento Index...[/blue]")
            Index = np.fromfile(os.path.join(data_dir, 'Index_Long_4000x250.dat'), dtype=np.int32).reshape((DimRg, DimAz), order='F')
            progress.advance(load_task)
            
            # Mask
            progress.update(load_task, description="[blue]Caricamento Mask...[/blue]")
            Mask = np.fromfile(os.path.join(data_dir, 'Mask_Byte_4000x250.dat'), dtype=np.uint8).reshape((DimRg, DimAz), order='F')
            progress.advance(load_task)
            
            # Sint
            progress.update(load_task, description="[blue]Caricamento Sint...[/blue]")
            Sint = np.fromfile(os.path.join(data_dir, 'Sint_Long_4000x250.dat'), dtype=np.int32).reshape((DimRg, DimAz), order='F')
            progress.advance(load_task)
            
            # RawData
            progress.update(load_task, description="[blue]Caricamento RawData (complex data)...[/blue]")
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
                console.print("[green]✓[/green] RawData caricato con successo usando formato interleaved", style="green")
                console.print(f"   Dimensioni: [cyan]{RawData.shape}[/cyan], Tipo: [cyan]{RawData.dtype}[/cyan]")
            else:
                # Prova formato split (tutti i reali, poi tutti gli immaginari)
                progress.update(load_task, description="[yellow]Formato interleaved non corretto. Provo formato split...[/yellow]")
                expected_len_split = Nra_raw * Naz_raw
                if data_f32.size == expected_len: # Riusa expected_len perché è lo stesso numero totale di float32
                    half_len = expected_len // 2
                    real_part = data_f32[:half_len].reshape((Nra_raw, Naz_raw), order='F')
                    imag_part = data_f32[half_len:].reshape((Nra_raw, Naz_raw), order='F')
                    RawData = real_part + 1j * imag_part # Risultato è complex64
                    console.print("[green]✓[/green] RawData caricato con successo usando formato split", style="green")
                    console.print(f"   Dimensioni: [cyan]{RawData.shape}[/cyan], Tipo: [cyan]{RawData.dtype}[/cyan]")
                else:
                    raise IOError(f"La dimensione del file RawData ({data_f32.size} float) non corrisponde alla dimensione prevista per formato interleaved o split ({expected_len} float).")
            progress.advance(load_task)
            
            # Parametri scalari
            progress.update(load_task, description="[blue]Caricamento parametri scalari...[/blue]")
            NearRange = np.fromfile(os.path.join(data_dir, 'NearRangeDbl_1_elemento.dat'), dtype=np.float64)[0]
            DeltaRange = np.fromfile(os.path.join(data_dir, 'DeltaRangeDbl_1_elemento.dat'), dtype=np.float64)[0]
            Lambda = np.fromfile(os.path.join(data_dir, 'LambdaDbl_1_elemento.dat'), dtype=np.float64)[0]
            progress.advance(load_task)

        # Mostro parametri in una tabella
        params_table = Table(title="Parametri Principali", box=box.ROUNDED)
        params_table.add_column("Parametro", style="cyan")
        params_table.add_column("Valore", style="green")
        params_table.add_row("NearRange", f"{NearRange:.6f}")
        params_table.add_row("DeltaRange", f"{DeltaRange:.6f}")
        params_table.add_row("Lambda", f"{Lambda:.6f}")
        console.print(params_table)

        # Controllo dimensioni traiettoria vs RawData
        if not (len(wgsx1) == len(wgsy1) == len(wgsz1) == Naz_raw):
            console.print(Panel(
                f"[yellow]⚠ Attenzione:[/yellow] La lunghezza della traiettoria non corrisponde alla dimensione azimuth di RawData!\n" +
                f"WGSX: [cyan]{len(wgsx1)}[/cyan], WGSY: [cyan]{len(wgsy1)}[/cyan], WGSZ: [cyan]{len(wgsz1)}[/cyan], RawData Naz: [cyan]{Naz_raw}[/cyan]",
                box=box.ROUNDED, style="yellow"
            ))

    except FileNotFoundError as e:
        console.print(f"[bold red]Errore:[/bold red] File non trovato - {e}", style="red")
        exit()
    except IOError as e:
        console.print(f"[bold red]Errore:[/bold red] Errore di I/O - {e}", style="red")
        exit()
    except ValueError as e:
        console.print(f"[bold red]Errore:[/bold red] Errore di valore (spesso problema di reshape) - {e}", style="red")
        exit()
    except Exception as e:
        console.print(f"[bold red]Errore imprevisto:[/bold red] {e}", style="red")
        exit()

    console.print(Panel("[bold blue]FASE 2: AVVIO PROCESSO DI FOCALIZZAZIONE[/bold blue]", style="blue", box=box.DOUBLE))
    # Chiama la funzione di focalizzazione corretta
    SlcPx = FocalizzatoreBpPx(xxg1, yyg1, zzg1, Mask, Index, wgsx1, wgsy1, wgsz1, RawData, NearRange, DeltaRange, Lambda, Sint)

    console.print(Panel("[bold blue]FASE 3: SALVATAGGIO RISULTATI[/bold blue]", style="blue", box=box.DOUBLE))
    try:
        output_filename_base = 'SlcPx_Python_Corrected'
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            save_task = progress.add_task("[blue]Salvataggio risultati...[/blue]", total=3)
        
            # 1. Salva in formato split float32 (real_part, imag_part) come nel codice originale
            progress.update(save_task, description=f"[blue]Salvataggio in formato split float32...[/blue]")
            output_filename_split = output_filename_base + '_SplitF32.dat'
            real_part_f32 = SlcPx.real.astype(np.float32)
            imag_part_f32 = SlcPx.imag.astype(np.float32)
            with open(output_filename_split, 'wb') as f:
                # Scrivi prima tutta la parte reale, poi tutta quella immaginaria
                real_part_f32.flatten(order='F').tofile(f)
                imag_part_f32.flatten(order='F').tofile(f)
            progress.advance(save_task)
            
            # 2. Salva in formato interleaved float32 (per potenziale compatibilità IDL readu)
            progress.update(save_task, description=f"[blue]Salvataggio in formato interleaved float32...[/blue]")
            output_filename_interleaved = output_filename_base + '_InterleavedF32.dat'
            # Assicurati che siano float32 prima di interleaving
            real_flat_f32 = SlcPx.real.astype(np.float32).flatten(order='F')
            imag_flat_f32 = SlcPx.imag.astype(np.float32).flatten(order='F')
            interleaved_f32 = np.empty(real_flat_f32.size * 2, dtype=np.float32)
            interleaved_f32[0::2] = real_flat_f32
            interleaved_f32[1::2] = imag_flat_f32
            with open(output_filename_interleaved, 'wb') as f:
                interleaved_f32.tofile(f)
            progress.advance(save_task)
            
            # 3. Salva in formato NumPy nativo (consigliato per riutilizzo in Python)
            progress.update(save_task, description=f"[blue]Salvataggio in formato NumPy nativo (.npy)...[/blue]")
            output_filename_npy = output_filename_base + '.npy'
            np.save(output_filename_npy, SlcPx) # Salva l'array complex128 direttamente
            progress.advance(save_task)
        
        # Tabella dei file di output
        output_table = Table(title="File di Output", box=box.ROUNDED)
        output_table.add_column("Tipo", style="cyan")
        output_table.add_column("File", style="green")
        output_table.add_column("Formato", style="blue")
        output_table.add_row("Split", output_filename_split, "float32 [Reali + Immaginari]")
        output_table.add_row("Interleaved", output_filename_interleaved, "float32 [Real1, Imag1, Real2, ...]")
        output_table.add_row("NumPy", output_filename_npy, "complex128 native")
        console.print(output_table)

    except Exception as e:
        console.print(f"[bold red]Errore nel salvataggio dei risultati:[/bold red] {e}", style="red")

    # Stampa tempo totale
    elapsed_total = time.time() - start_time_total
    minutes = int(elapsed_total // 60)
    seconds = int(elapsed_total % 60)
    
    console.print(Panel(
        f"[bold green]Processo completato![/bold green]\n" +
        f"Tempo totale di esecuzione: [cyan]{minutes}[/cyan] min [cyan]{seconds}[/cyan] sec",
        box=box.DOUBLE, style="green"
    ))
