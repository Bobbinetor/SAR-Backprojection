import numpy as np
import time
from numba import jit, prange
import os
from rich.console import Console
from rich.theme import Theme
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.spinner import Spinner
from rich.status import Status
from pyfiglet import Figlet
from rich.box import SIMPLE, DOUBLE
from rich.markdown import Markdown

# Definisci un tema personalizzato per i colori
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow bold",
    "error": "bold red",
    "success": "bold green",
    "progress": "blue",
    "title": "bold magenta",
    "highlight": "bold cyan",
    "normal": "white",
    "dim": "dim white",
    "accent": "#FF00AF",
})

# Crea la console per l'output formattato
console = Console(theme=custom_theme)

# Use Numba for just-in-time compilation to speed up the core calculation
@jit(nopython=True, parallel=True)
def process_aperture_segment(jj, xxg1_flat, yyg1_flat, zzg1_flat, mask_flat, index_flat, Sint_v_flat, 
                             wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, 
                             focused_sample, niter, Sint, Naz, Nra, beta):
    """
    Process a segment of the synthetic aperture with JIT compilation for speed
    """
    index1 = index_flat + jj
    
    # Find valid area - using numba compatible operations
    valid_indices = []
    for i in range(len(index_flat)):
        if (index1[i] <= Naz and index1[i] >= 1 and 
            np.ceil(Sint_v_flat[i]/2) >= abs(jj) and mask_flat[i] == 1):
            valid_indices.append(i)
    
    for idx in valid_indices:
        index1a = index1[idx]
        
        # Convert to 0-based index for Python
        idx_wgs = index1a - 1
        
        if idx_wgs < 0 or idx_wgs >= len(wgsx):
            continue
            
        xcomp = xxg1_flat[idx] - wgsx[idx_wgs]
        ycomp = yyg1_flat[idx] - wgsy[idx_wgs]
        zcomp = zzg1_flat[idx] - wgsz[idx_wgs]
        
        dist_patch2 = xcomp*xcomp + ycomp*ycomp + zcomp*zcomp
        dist_patch = np.sqrt(dist_patch2)
        
        # Calculate range index (adjust for Python indexing)
        ind_line_ra = int(round((dist_patch-near_range)/dra))
        
        # Check if the index is within bounds
        if 0 <= ind_line_ra < Nra:
            niter[idx] += 1
            
            # Calculate phase filter
            filter_val = np.exp(1j*(-4.*np.pi/lambda_val*dist_patch + beta*dist_patch2))
            
            # Get raw data and apply back-projection
            raw_ind_r = ind_line_ra
            raw_ind_a = idx_wgs
            
            if 0 <= raw_ind_r < Nra and 0 <= raw_ind_a < Naz:
                raw_value = Raw_Data[raw_ind_r, raw_ind_a]
                back_projected = dist_patch2 * raw_value * np.conj(filter_val)
                focused_sample[idx] += back_projected
    
    return focused_sample, niter

def FocalizzatoreBpMatriciale(xxg1, yyg1, zzg1, mask, index, wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, Sint_v):
    """
    Matrix-based SAR Focuser - Optimized version
    """
    # Mostra solo un titolo al posto del layout con header e body
    console.print(Panel(
        Markdown("# SAR FOCUSER - MATRIX VERSION"),
        border_style="accent",
        box=DOUBLE
    ))

    t1 = time.time()

    # Get dimensions of Raw_Data
    Nra, Naz = Raw_Data.shape

    # Get dimensions of the output grid
    nx_grid, ny_grid = xxg1.shape
    
    # Crea tabella per i parametri
    param_table = Table(show_header=False, box=SIMPLE)
    param_table.add_column("Parametro", style="info")
    param_table.add_column("Valore", style="normal")
    param_table.add_row("Dimensioni Raw Data", f"{Nra} × {Naz}")
    param_table.add_row("Dimensioni Griglia", f"{nx_grid} × {ny_grid}")
    
    console.print(Panel(param_table, title="[title]Parametri di processo[/title]", border_style="blue"))
    
    # Flatten arrays for easier processing (column-major order to match MATLAB behavior)
    with console.status("[info]Preparazione arrays...[/info]", spinner="dots"):
        xxg1_flat = xxg1.reshape(-1, order='F').copy()
        yyg1_flat = yyg1.reshape(-1, order='F').copy()
        zzg1_flat = zzg1.reshape(-1, order='F').copy()
        mask_flat = mask.reshape(-1, order='F').copy()
        index_flat = index.reshape(-1, order='F').copy()
        Sint_v_flat = Sint_v.reshape(-1, order='F').copy()
        
        dim = xxg1_flat.size

        # Constants
        alfa = 3.335987619777693e11
        c = 2.9979246e8
        beta = 4*np.pi*alfa/(c**2)

        # Initialize output arrays
        focused_sample = np.zeros(dim, dtype=np.complex128)
        niter = np.zeros(dim, dtype=np.int32)

        # Find invalid points in Sint_v
        no_valid = np.where(Sint_v_flat == -9999)[0]
        if len(no_valid) > 0:
            mask_flat[no_valid] = 0

        # Find maximum synthetic aperture length
        sint = np.max(Sint_v_flat)
    
    # Create checkpoint file path
    checkpoint_file = 'focalizador_checkpoint.npz'
    
    # Controlla e carica checkpoint silenziosamente
    start_jj = -int(np.ceil(sint/2))
    if os.path.exists(checkpoint_file):
        try:
            checkpoint = np.load(checkpoint_file)
            focused_sample = checkpoint['focused_sample']
            niter = checkpoint['niter']
            start_jj = checkpoint['next_jj']
        except Exception as e:
            console.print(f"[error]Errore nel caricamento del checkpoint: {e}[/error]")
            start_jj = -int(np.ceil(sint/2))
    
    # Process in smaller batches for better progress tracking
    batch_size = 10
    total_steps = 2 * int(np.ceil(sint/2)) + 1
    steps_done = start_jj + int(np.ceil(sint/2))
    
    # Barra di progresso avanzata
    console.print("\n[info]Starting synthetic aperture processing...[/info]")
    
    with Progress(
        TextColumn("[progress]{task.description}"),
        BarColumn(bar_width=None, pulse_style="cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_main = progress.add_task("[highlight]Processing synthetic aperture[/highlight]", total=total_steps, completed=steps_done)
        task_batch = progress.add_task("[dim]Current batch[/dim]", total=batch_size, visible=False)
        
        for batch_start in range(start_jj, int(np.ceil(sint/2))+1, batch_size):
            batch_end = min(batch_start + batch_size, int(np.ceil(sint/2))+1)
            batch_size_actual = batch_end - batch_start
            
            progress.update(task_batch, total=batch_size_actual, completed=0, visible=True)
            progress.update(task_batch, description=f"[dim]Batch {batch_start}→{batch_end-1}[/dim]")
            
            for jj in range(batch_start, batch_end):
                focused_sample, niter = process_aperture_segment(
                    jj, xxg1_flat, yyg1_flat, zzg1_flat, mask_flat, index_flat, Sint_v_flat,
                    wgsx, wgsy, wgsz, Raw_Data, near_range, dra, lambda_val, 
                    focused_sample, niter, sint, Naz, Nra, beta
                )
                
                # Aggiorna progresso
                progress.update(task_main, advance=1)
                progress.update(task_batch, advance=1)
            
            # Aggiorna task_batch come completato
            progress.update(task_batch, completed=batch_size_actual)
            
            # Save checkpoint after each batch (aggiornamento alla barra di progresso)
            progress.update(task_batch, description=f"[info]Saving checkpoint {batch_end}/{int(np.ceil(sint/2))}...[/info]")
            np.savez(checkpoint_file, 
                    focused_sample=focused_sample, 
                    niter=niter, 
                    next_jj=batch_end)
            
            # Aggiorna solo la descrizione nella barra di progresso
            completion_percent = ((batch_end + int(np.ceil(sint/2))) / total_steps) * 100
            progress.update(task_main, description=f"[highlight]Processing: {completion_percent:.1f}% complete[/highlight]")
            progress.update(task_batch, description=f"[dim]Batch {batch_start}→{batch_end-1} completed[/dim]")
            
            # Elimina completamente il panel del checkpoint per mantenere l'output compatto
            # Non stampiamo più il box dopo ogni salvataggio
    
    # Normalizzazione con feedback visivo (senza status)
    console.print("[info]Normalizing results...[/info]")
    area2 = np.where(niter > 0)[0]
    if len(area2) > 0:
        focused_sample[area2] = focused_sample[area2] / niter[area2]
        console.print(f"[success]Normalized {len(area2)} points[/success]")
    else:
        console.print("[warning]No points to normalize![/warning]")
    
    # Reshape the output to match the original grid
    console.print("[info]Final reshape...[/info]")
    focused_sample = np.reshape(focused_sample, (nx_grid, ny_grid), order='F')
    console.print("[success]Reshape completed[/success]")
    time.sleep(0.5)

    t2 = time.time() - t1
    
    # Statistiche finali in tabella
    stats_table = Table(title="[title]Processing Statistics[/title]", box=DOUBLE)
    stats_table.add_column("Parameter", style="info")
    stats_table.add_column("Value", justify="right", style="normal")

    stats_table.add_row("Processing time", f"{t2:.2f} seconds")
    stats_table.add_row("Grid size", f"{nx_grid}x{ny_grid}")
    stats_table.add_row("Processed points", f"{len(area2):,}")
    stats_table.add_row("Synthetic aperture", f"{int(sint)} samples")
    stats_table.add_row("Efficiency", f"{len(area2)/(t2+0.001):.2f} points/sec")

    console.print("\n")
    console.print(Panel(stats_table, border_style="accent"))

    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            console.print("[success]Checkpoint file successfully removed.[/success]")
        except Exception as e:
            console.print(f"[warning]Unable to remove checkpoint: {e}[/warning]")

    return focused_sample

# Main script
if __name__ == "__main__":
    # Inizia a monitorare il tempo totale
    start_time_total = time.time()
    
    # Display a cool ASCII banner
    f = Figlet(font='slant')
    console.print(f.renderText('SAR Focusing System'), style="accent")
    console.print(Panel("SAR Backprojection", border_style="cyan"))
    
    # Initialize arrays with visualizzazione stato
    DimRg = 4000
    DimAz = 250
    
    # Visualizza configurazione
    config_table = Table(show_header=False, box=SIMPLE)
    config_table.add_column("Parameter", style="info")
    config_table.add_column("Value", style="normal")
    config_table.add_row("Range Dimensions", str(DimRg))
    config_table.add_row("Azimuth Dimensions", str(DimAz))
    config_table.add_row("Mode", "Optimised Matrix")
    
    console.print(Panel(config_table, title="[title]Configuration[/title]", border_style="blue"))
    
    with Status("[info]Initialising arrays...[/info]", console=console) as status:
        xxg1 = np.zeros((DimRg, DimAz))
        yyg1 = np.zeros((DimRg, DimAz))
        zzg1 = np.zeros((DimRg, DimAz))
        Index = np.zeros((DimRg, DimAz), dtype=np.int32)
        Mask = np.zeros((DimRg, DimAz), dtype=np.uint8)
        Sint = np.zeros((DimRg, DimAz), dtype=np.int32)
        RawData = np.zeros((4762, 40001), dtype=complex)
        wgsx1 = np.zeros(40001)
        wgsy1 = np.zeros(40001)
        wgsz1 = np.zeros(40001)
        NearRange = 0.0
        DeltaRange = 0.0
        Lambda = 0.0
        status.update("[success]Arrays initialised.[/success]")
        time.sleep(0.5)

    # Crea tabella per il caricamento file
    file_table = Table(title="[title]File Loading Status[/title]")
    file_table.add_column("File", style="info")
    file_table.add_column("Status", justify="center")
    file_table.add_column("Size", justify="right")
    
    # Read data files (use Fortran ordering to match MATLAB)
    console.print("[highlight]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/highlight]")
    console.print("[title]PHASE 1: DATA LOADING[/title]")
    
    files_to_load = [
        ('Dem_x_Dbl4000x250.dat', 'xxg1'),
        ('Dem_y_Dbl4000x250.dat', 'yyg1'),
        ('Dem_z_Dbl4000x250.dat', 'zzg1'),
        ('Traiett_x_Dbl40001.dat', 'wgsx1'),
        ('Traiett_y_Dbl40001.dat', 'wgsy1'),
        ('Traiett_z_Dbl40001.dat', 'wgsz1'),
        ('Index_Long_4000x250.dat', 'Index'),
        ('Mask_Byte_4000x250.dat', 'Mask'),
        ('Sint_Long_4000x250.dat', 'Sint')
    ]
    
    for filename, varname in files_to_load:
        try:
            with console.status(f"[info]Loading {filename}...[/info]", spinner="dots") as status:
                if varname == 'xxg1':
                    with open(filename, 'rb') as f:
                        xxg1 = np.fromfile(f, dtype=np.float64).reshape((DimRg, DimAz), order='F')
                        filesize = os.path.getsize(filename) / (1024 * 1024)  # in MB
                elif varname == 'yyg1':
                    with open(filename, 'rb') as f:
                        yyg1 = np.fromfile(f, dtype=np.float64).reshape((DimRg, DimAz), order='F')
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                elif varname == 'zzg1':
                    with open(filename, 'rb') as f:
                        zzg1 = np.fromfile(f, dtype=np.float64).reshape((DimRg, DimAz), order='F')
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                elif varname == 'wgsx1':
                    with open(filename, 'rb') as f:
                        wgsx1 = np.fromfile(f, dtype=np.float64)
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                elif varname == 'wgsy1':
                    with open(filename, 'rb') as f:
                        wgsy1 = np.fromfile(f, dtype=np.float64)
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                elif varname == 'wgsz1':
                    with open(filename, 'rb') as f:
                        wgsz1 = np.fromfile(f, dtype=np.float64)
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                elif varname == 'Index':
                    with open(filename, 'rb') as f:
                        Index = np.fromfile(f, dtype=np.int32).reshape((DimRg, DimAz), order='F')
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                elif varname == 'Mask':
                    with open(filename, 'rb') as f:
                        Mask = np.fromfile(f, dtype=np.uint8).reshape((DimRg, DimAz), order='F')
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                elif varname == 'Sint':
                    with open(filename, 'rb') as f:
                        Sint = np.fromfile(f, dtype=np.int32).reshape((DimRg, DimAz), order='F')
                        filesize = os.path.getsize(filename) / (1024 * 1024)
                        
                time.sleep(0.2)  # Per visualizzare lo spinner
                        
            file_table.add_row(filename, "[success]✓[/success]", f"{filesize:.2f} MB")
        except Exception as e:
            file_table.add_row(filename, f"[error]✗[/error]", "N/A")
            console.print(f"[error]Errore nel caricamento di {filename}: {e}[/error]")
    
    # Special handling for complex data with animation
    console.print("\n[info]Loading complex data...[/info]")
    
    with Progress(
        TextColumn("[progress]Loading complex data"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console
    ) as progress:
        load_task = progress.add_task("", total=100)
        
        try:
            with open('RawData_Cmplx_4762x40001.dat', 'rb') as f:
                # Simula lettura progressiva di file grandi
                file_size = os.path.getsize('RawData_Cmplx_4762x40001.dat')
                
                # Simula la lettura progressiva (per file molto grandi)
                RawComplex = np.fromfile(f, dtype=np.float32)
                progress.update(load_task, completed=50)
                
                RawComplex = RawComplex.reshape(2, 4762*40001, order='F')
                progress.update(load_task, completed=70)
                
                RawComplex = RawComplex.reshape(2, 4762, 40001, order='F')
                progress.update(load_task, completed=85)
                
                RawData = np.complex128(RawComplex[0]) + 1j * np.complex128(RawComplex[1])
                
                # Transpose to match MATLAB's orientation
                RawData = RawData.T  # Now it's (40001, 4762)
                progress.update(load_task, completed=95)
                
                # Swap axes to match expected input shape
                RawData = np.swapaxes(RawData, 0, 1)  # Now it's (4762, 40001)
                progress.update(load_task, completed=100)
                
                complex_size = file_size / (1024 * 1024)  # in MB
                file_table.add_row('RawData_Cmplx_4762x40001.dat', "[success]✓[/success]", f"{complex_size:.2f} MB")
        except Exception as e:
            file_table.add_row('RawData_Cmplx_4762x40001.dat', f"[error]✗[/error]", "N/A")
            console.print(f"[error]Errore nel caricamento dei dati complessi: {e}[/error]")
    
    # Carica parametri singoli
    with console.status("[info]Loading parameters...[/info]", spinner="dots") as status:
        try:
            with open('NearRangeDbl_1_elemento.dat', 'rb') as f:
                NearRange = np.fromfile(f, dtype=np.float64)[0]
                file_table.add_row('NearRangeDbl_1_elemento.dat', "[success]✓[/success]", "< 1 MB")
        except Exception as e:
            file_table.add_row('NearRangeDbl_1_elemento.dat', f"[error]✗[/error]", "N/A")
            console.print(f"[error]Error loading NearRange: {e}[/error]")

        try:
            with open('DeltaRangeDbl_1_elemento.dat', 'rb') as f:
                DeltaRange = np.fromfile(f, dtype=np.float64)[0]
                file_table.add_row('DeltaRangeDbl_1_elemento.dat', "[success]✓[/success]", "< 1 MB")
        except Exception as e:
            file_table.add_row('DeltaRangeDbl_1_elemento.dat', f"[error]✗[/error]", "N/A")
            console.print(f"[error]Error loading DeltaRange: {e}[/error]")

        try:
            with open('LambdaDbl_1_elemento.dat', 'rb') as f:
                Lambda = np.fromfile(f, dtype=np.float64)[0]
                file_table.add_row('LambdaDbl_1_elemento.dat', "[success]✓[/success]", "< 1 MB")
        except Exception as e:
            file_table.add_row('LambdaDbl_1_elemento.dat', f"[error]✗[/error]", "N/A")
            console.print(f"[error]Error loading Lambda: {e}[/error]")
            
    console.print(file_table)
    console.print("[success]All data successfully loaded.[/success]")
    console.print("[highlight]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/highlight]")

    # Riepilogo prima della focalizzazione
    param_summary = f"""
    • NearRange: {NearRange:.6f}
    • DeltaRange: {DeltaRange:.6f}
    • Lambda: {Lambda:.6f}
    • Raw Data Dimensions: {RawData.shape[0]} × {RawData.shape[1]}
    • Grid Dimensions: {DimRg} × {DimAz}
    """
    
    console.print(Panel(param_summary, title="[title]Focusing Parameters[/title]", border_style="blue"))

    # Call the focusing function
    console.print("[title]PHASE 2: STARTING FOCUSING PROCESS[/title]")
    SlcMatr = FocalizzatoreBpMatriciale(xxg1, yyg1, zzg1, Mask, Index, wgsx1, wgsy1, wgsz1, RawData, NearRange, DeltaRange, Lambda, Sint)

    # Save results with visualization
    console.print("[highlight]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/highlight]")
    console.print("[title]PHASE 3: SAVING RESULTS[/title]")
    
    save_table = Table(title="[title]File Saving Status[/title]")
    save_table.add_column("File", style="info")
    save_table.add_column("Type", style="dim")
    save_table.add_column("Status", justify="center")
    save_table.add_column("Size", justify="right")
    
    # Save real and imaginary parts separately
    try:
        with console.status("[info]Saving real part...[/info]", spinner="point") as status:
            with open('SlcMatr_real.dat', 'wb') as f:
                SlcMatr.real.astype(np.float64).tofile(f)
            filesize = os.path.getsize('SlcMatr_real.dat') / (1024 * 1024)  # in MB
            time.sleep(0.5)
        save_table.add_row('SlcMatr_real.dat', "Real part", "[success]✓[/success]", f"{filesize:.2f} MB")
    except Exception as e:
        save_table.add_row('SlcMatr_real.dat', "Parte reale", "[error]✗[/error]", "N/A")
        console.print(f"[error]Error saving real part: {e}[/error]")
    
    try:
        with console.status("[info]Saving imaginary part...[/info]", spinner="point") as status:
            with open('SlcMatr_imag.dat', 'wb') as f:
                SlcMatr.imag.astype(np.float64).tofile(f)
            filesize = os.path.getsize('SlcMatr_imag.dat') / (1024 * 1024)  # in MB
            time.sleep(0.5)
        save_table.add_row('SlcMatr_imag.dat', "Imaginary part", "[success]✓[/success]", f"{filesize:.2f} MB")
    except Exception as e:
        save_table.add_row('SlcMatr_imag.dat', "Imaginary part", "[error]✗[/error]", "N/A")
        console.print(f"[error]Error saving imaginary part: {e}[/error]")
    
    # Also save as a single complex binary file
    try:
        with console.status("[info]Saving unified complex file...[/info]", spinner="point") as status:
            with open('SlcMatr_complex.dat', 'wb') as f:
                # Store in a format similar to the raw data: [real, imag]
                np.vstack((SlcMatr.real.flatten(order='F'), 
                        SlcMatr.imag.flatten(order='F'))).astype(np.float32).tofile(f)
            filesize = os.path.getsize('SlcMatr_complex.dat') / (1024 * 1024)  # in MB
            time.sleep(0.5)
        save_table.add_row('SlcMatr_complex.dat', "Complex", "[success]✓[/success]", f"{filesize:.2f} MB")
    except Exception as e:
        save_table.add_row('SlcMatr_complex.dat', "Complex", "[error]✗[/error]", "N/A")
        console.print(f"[error]Error saving complex file: {e}[/error]")
    
    console.print(save_table)
    
    # End of processing with banner and total time
    console.print("[highlight]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/highlight]")
    
    # Calculate total execution time
    elapsed_total = time.time() - start_time_total
    
    # Create message that includes execution time
    completion_message = f"""
[success]Process completed successfully![/success]

Total execution time: [bold]{elapsed_total:.2f}[/bold] seconds
"""
    
    console.print(Panel(completion_message, border_style="green", title="Completed"))