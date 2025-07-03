import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import json

# --- Utility functions ---
def load_complex_dat(filename, shape, fmt='interleaved'):
    """
    Load a .dat file as a complex array.
    fmt: 'interleaved' or 'split'
    shape: (rows, cols)
    """
    if fmt == 'interleaved':
        arr = np.fromfile(filename, dtype=np.float32)
        if arr.size != shape[0]*shape[1]*2:
            raise ValueError(f"File size mismatch for {filename} (expected {shape[0]*shape[1]*2}, got {arr.size})")
        arr = arr.reshape(-1, 2)
        cplx = arr[:,0] + 1j*arr[:,1]
        cplx = cplx.reshape(shape, order='F')
        return cplx
    elif fmt == 'split':
        arr = np.fromfile(filename, dtype=np.float32)
        n = shape[0]*shape[1]
        if arr.size != n*2:
            raise ValueError(f"File size mismatch for {filename} (expected {n*2}, got {arr.size})")
        real = arr[:n].reshape(shape, order='F')
        imag = arr[n:].reshape(shape, order='F')
        return real + 1j*imag
    else:
        raise ValueError(f"Unknown format: {fmt}")

def load_truth(filename, shape):
    arr = np.fromfile(filename, dtype=np.complex64)
    if arr.size != shape[0]*shape[1]:
        raise ValueError(f"File size mismatch for {filename} (expected {shape[0]*shape[1]}, got {arr.size})")
    return arr.reshape(shape, order='F')

# --- Main comparison ---
def compare_phase(file1, file2, shape, fmt1, fmt2, mask_zero_amp=True, plot=True, out_prefix=None):
    if fmt1 == 'truth':
        arr1 = load_truth(file1, shape)
    else:
        arr1 = load_complex_dat(file1, shape, fmt1)
    if fmt2 == 'truth':
        arr2 = load_truth(file2, shape)
    else:
        arr2 = load_complex_dat(file2, shape, fmt2)

    phase1 = np.angle(arr1)
    phase2 = np.angle(arr2)
    phase_diff = np.angle(np.exp(1j*(phase1 - phase2)))

    if mask_zero_amp:
        mask = (np.abs(arr1) > 1e-6) & (np.abs(arr2) > 1e-6)
        phase_diff_masked = np.where(mask, phase_diff, np.nan)
    else:
        phase_diff_masked = phase_diff

    # Detailed statistical analysis
    valid = np.isfinite(phase_diff_masked)
    phase_diff_valid = phase_diff_masked[valid]
    
    if len(phase_diff_valid) == 0:
        print("ERROR: No valid phase difference data found!")
        return None, None
    
    # Basic statistics
    mean = np.mean(phase_diff_valid)
    std = np.std(phase_diff_valid)
    median = np.median(phase_diff_valid)
    
    # Percentiles
    p1 = np.percentile(phase_diff_valid, 1)
    p5 = np.percentile(phase_diff_valid, 5)
    p95 = np.percentile(phase_diff_valid, 95)
    p99 = np.percentile(phase_diff_valid, 99)
    
    # Range and IQR
    min_val = np.min(phase_diff_valid)
    max_val = np.max(phase_diff_valid)
    q25 = np.percentile(phase_diff_valid, 25)
    q75 = np.percentile(phase_diff_valid, 75)
    iqr = q75 - q25
    
    # Bias analysis (systematic offset)
    bias_rad = mean
    bias_deg = bias_rad * 180 / np.pi
    
    # RMS error
    rms = np.sqrt(np.mean(phase_diff_valid**2))
    
    # Count of pixels with large phase errors (>π/4)
    large_errors = np.sum(np.abs(phase_diff_valid) > np.pi/4)
    large_error_percent = (large_errors / len(phase_diff_valid)) * 100
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("DETAILED PHASE DIFFERENCE ANALYSIS")
    print("="*60)
    print(f"Total pixels analyzed:     {np.sum(valid):,}")
    print(f"Valid pixels:              {len(phase_diff_valid):,}")
    print(f"Invalid/masked pixels:     {np.sum(~valid):,}")
    print("\nPHASE STATISTICS (radians):")
    print(f"  Mean (bias):             {mean:.6f} rad ({bias_deg:.3f}°)")
    print(f"  Median:                  {median:.6f} rad ({median*180/np.pi:.3f}°)")
    print(f"  Standard deviation:      {std:.6f} rad ({std*180/np.pi:.3f}°)")
    print(f"  RMS error:               {rms:.6f} rad ({rms*180/np.pi:.3f}°)")
    print(f"  Range: [{min_val:.6f}, {max_val:.6f}] rad")
    print(f"         [{min_val*180/np.pi:.3f}°, {max_val*180/np.pi:.3f}°]")
    print(f"  IQR (Q75-Q25):           {iqr:.6f} rad ({iqr*180/np.pi:.3f}°)")
    print("\nPERCENTILES (radians):")
    print(f"  1%:   {p1:.6f} rad ({p1*180/np.pi:.3f}°)")
    print(f"  5%:   {p5:.6f} rad ({p5*180/np.pi:.3f}°)")
    print(f"  25%:  {q25:.6f} rad ({q25*180/np.pi:.3f}°)")
    print(f"  75%:  {q75:.6f} rad ({q75*180/np.pi:.3f}°)")
    print(f"  95%:  {p95:.6f} rad ({p95*180/np.pi:.3f}°)")
    print(f"  99%:  {p99:.6f} rad ({p99*180/np.pi:.3f}°)")
    print("\nERROR ANALYSIS:")
    print(f"  Pixels with |error| > π/4: {large_errors:,} ({large_error_percent:.2f}%)")
    print(f"  Pixels with |error| > π/8: {np.sum(np.abs(phase_diff_valid) > np.pi/8):,} ({np.sum(np.abs(phase_diff_valid) > np.pi/8)/len(phase_diff_valid)*100:.2f}%)")
    print(f"  Pixels with |error| > π/16: {np.sum(np.abs(phase_diff_valid) > np.pi/16):,} ({np.sum(np.abs(phase_diff_valid) > np.pi/16)/len(phase_diff_valid)*100:.2f}%)")
    print(f"  Pixels with |error| > π/32: {np.sum(np.abs(phase_diff_valid) > np.pi/32):,} ({np.sum(np.abs(phase_diff_valid) > np.pi/32)/len(phase_diff_valid)*100:.2f}%)")
    print("="*60)

    if plot:
        # Crea la cartella di output se necessario
        if out_prefix:
            out_dir = os.path.dirname(out_prefix)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(18,10))
        # Fase assoluta file 1
        plt.subplot(2,3,1)
        plt.imshow(phase1, cmap='twilight', aspect='auto')
        plt.colorbar(label='Phase (rad)')
        plt.title('Phase file 1')
        # Fase assoluta file 2
        plt.subplot(2,3,2)
        plt.imshow(phase2, cmap='twilight', aspect='auto')
        plt.colorbar(label='Phase (rad)')
        plt.title('Phase file 2')
        # Mappa differenza di fase
        plt.subplot(2,3,3)
        plt.imshow(phase_diff_masked, cmap='twilight', aspect='auto', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(label='Phase diff (rad)')
        plt.title('Phase difference map')
        # Modulo file 1
        plt.subplot(2,3,4)
        plt.imshow(np.abs(arr1), cmap='gray', aspect='auto')
        plt.colorbar(label='Amplitude')
        plt.title('Amplitude file 1')
        # Modulo file 2
        plt.subplot(2,3,5)
        plt.imshow(np.abs(arr2), cmap='gray', aspect='auto')
        plt.colorbar(label='Amplitude')
        plt.title('Amplitude file 2')
        # Istogramma differenza di fase
        plt.subplot(2,3,6)
        plt.hist(phase_diff_masked[valid].ravel(), bins=100, color='C0', alpha=0.7)
        plt.xlabel('Phase diff (rad)')
        plt.ylabel('Count')
        plt.title('Phase diff histogram')
        plt.tight_layout()
        if out_prefix:
            plt.savefig(f"{out_prefix}_phase_diff.png", dpi=150)
            plt.close()
        else:
            plt.close()

    # Create detailed statistics dictionary for return
    stats = {
        'total_pixels': int(np.sum(valid)),
        'valid_pixels': int(len(phase_diff_valid)),
        'invalid_pixels': int(np.sum(~valid)),
        'mean_rad': float(mean),
        'mean_deg': float(bias_deg),
        'median_rad': float(median),
        'median_deg': float(median*180/np.pi),
        'std_rad': float(std),
        'std_deg': float(std*180/np.pi),
        'rms_rad': float(rms),
        'rms_deg': float(rms*180/np.pi),
        'min_rad': float(min_val),
        'max_rad': float(max_val),
        'range_rad': float(max_val - min_val),
        'iqr_rad': float(iqr),
        'percentiles': {
            'p1': float(p1), 'p5': float(p5), 'p25': float(q25), 'p75': float(q75), 'p95': float(p95), 'p99': float(p99)
        },
        'large_errors': {
            'pi_4': int(large_errors),
            'pi_8': int(np.sum(np.abs(phase_diff_valid) > np.pi/8)),
            'pi_16': int(np.sum(np.abs(phase_diff_valid) > np.pi/16)),
            'pi_32': int(np.sum(np.abs(phase_diff_valid) > np.pi/32))
        }
    }

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare phase of two SAR .dat files.')
    parser.add_argument('--file1', required=True, help='First file (.dat)')
    parser.add_argument('--file2', required=True, help='Second file (.dat, e.g. truth)')
    parser.add_argument('--shape', default='4000,32000', help='Shape, e.g. 4000,32000')
    parser.add_argument('--fmt1', default='interleaved', choices=['interleaved','split','truth'], help='Format of file1')
    parser.add_argument('--fmt2', default='truth', choices=['interleaved','split','truth'], help='Format of file2')
    parser.add_argument('--no-mask', action='store_true', help='Do not mask zero amplitude')
    parser.add_argument('--no-plot', action='store_true', help='Do not plot')
    parser.add_argument('--out', default=None, help='Prefix for output image')
    args = parser.parse_args()

    shape = tuple(map(int, args.shape.split(',')))
    stats = compare_phase(
        args.file1, args.file2, shape,
        args.fmt1, args.fmt2,
        mask_zero_amp=not args.no_mask,
        plot=not args.no_plot,
        out_prefix=args.out
    )
    
    # Save statistics to JSON file
    if args.out and stats:
        stats_filename = f"{args.out}_statistics.json"
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nDetailed statistics saved to: {stats_filename}")
