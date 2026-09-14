"""
Argument parser for EEG-based emotion recognition experiments.

Default values are tuned for the DEAP dataset with EEGNet,
following the LibEER framework conventions.

Usage:
    from args import get_args

    args = get_args()
    # Access via: args.dataset, args.batch_size, etc.
"""

import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="EEG Emotion Recognition — LOSO Cross-Validation"
    )

    # ==========================================================================
    # Dataset
    # ==========================================================================
    parser.add_argument('--dataset', type=str, default='deap',
                        choices=['deap', 'seed'],
                        help="Dataset: 'deap' (default) or 'seed'")
    parser.add_argument('--dataset_path', type=str,
                        default='D:/Project1/Work3/DEAP_data/data_preprocessed_python',
                        help="Path to dataset directory. "
                             "DEAP default: .../DEAP_data/data_preprocessed_python")
    parser.add_argument('--sessions', type=int, nargs='+', default=[1],
                        help="Session indices for SEED (1-based). DEAP always 1. "
                             "Example: '--sessions 1 2 3'")

    # ==========================================================================
    # Label
    # ==========================================================================
    parser.add_argument('--label_used', type=str, default='valence',
                        choices=['valence', 'arousal'],
                        help="Emotion dimension to classify (default: valence)")
    parser.add_argument('--bounds', type=float, nargs=2, default=[5.0, 5.0],
                        help="Binarization thresholds [low, high] (default: 5 5)")

    # ==========================================================================
    # Preprocessing
    # ==========================================================================
    parser.add_argument('--sample_length', type=int, default=128,
                        help="Segment length in timepoints (DEAP default: 128, SEED: 200)")
    parser.add_argument('--stride', type=int, default=128,
                        help="Stride between consecutive segments (default: 128)")
    parser.add_argument('--no_bandpass', action='store_true',
                        help="Skip bandpass filtering")

    # ==========================================================================
    # Training (LibEER EEGNet defaults for DEAP)
    # ==========================================================================
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument('--epochs', type=int, default=300,
                        help="Number of training epochs (default: 300)")
    parser.add_argument('--lr', type=float, default=0.02,
                        help="Learning rate (default: 0.02 for DEAP, 0.001 for SEED)")
    parser.add_argument('--seed', type=int, default=2024,
                        help="Base random seed (default: 2024)")

    # ==========================================================================
    # LOSO + Validation
    # ==========================================================================
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="Fraction of remaining subjects for validation, ~20%% (default: 0.2)")
    parser.add_argument('--num_seeds', type=int, default=5,
                        help="Number of random seeds per LOSO fold (default: 5)")

    # ==========================================================================
    # Device
    # ==========================================================================
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cuda:0', 'cpu'],
                        help="Device: 'auto' (use GPU if available), 'cuda', 'cpu'")

    # ==========================================================================
    # Output
    # ==========================================================================
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help="Directory to save result files")
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help="Directory to save log files")

    args = parser.parse_args()

    # ---- Device resolution ----
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args


def print_device_info(device):
    """Print GPU/CPU information for the user."""
    print("-" * 50)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"CUDA version: {torch.version.cuda}")
        if device.type == 'cuda':
            print(f"Using device: {device}")
        else:
            print(f"GPU available but using: {device}")
    else:
        print("No GPU detected — running on CPU")
        print(f"Using device: {device}")
    print("-" * 50)
