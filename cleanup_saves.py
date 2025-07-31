#!/usr/bin/env python3
"""Utility script to manage saved checkpoints and visualizations."""

import os
import glob
import argparse
from datetime import datetime
import shutil


def get_file_info(filepath):
    """Get file information including size and modification time."""
    stat = os.stat(filepath)
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(stat.st_mtime)
    return size_mb, mtime


def cleanup_old_saves(directory, keep_n=20, pattern="*.pth"):
    """Keep only the N most recent files matching pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    
    if len(files) <= keep_n:
        print(f"Found {len(files)} files in {directory}, keeping all (threshold: {keep_n})")
        return
    
    # Sort by modification time
    files_with_time = [(f, os.path.getmtime(f)) for f in files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    # Keep the most recent ones
    to_keep = files_with_time[:keep_n]
    to_remove = files_with_time[keep_n:]
    
    print(f"\nCleaning {directory}:")
    print(f"  - Found {len(files)} files")
    print(f"  - Keeping {len(to_keep)} most recent")
    print(f"  - Removing {len(to_remove)} older files")
    
    total_size_removed = 0
    for filepath, _ in to_remove:
        size_mb, _ = get_file_info(filepath)
        total_size_removed += size_mb
        os.remove(filepath)
        print(f"    Removed: {os.path.basename(filepath)} ({size_mb:.1f} MB)")
    
    print(f"  - Total space freed: {total_size_removed:.1f} MB")


def organize_saves():
    """Organize saves into appropriate directories."""
    # Create directories if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Move checkpoint files
    checkpoint_patterns = ['*.pth', '*checkpoint*.pt']
    moved_count = 0
    
    for pattern in checkpoint_patterns:
        for filepath in glob.glob(pattern):
            if os.path.dirname(filepath) != 'checkpoints':
                dest = os.path.join('checkpoints', os.path.basename(filepath))
                shutil.move(filepath, dest)
                moved_count += 1
                print(f"Moved {filepath} -> {dest}")
    
    # Move visualization files
    viz_patterns = ['*.png', '*.jpg', '*visual*.pdf']
    
    for pattern in viz_patterns:
        for filepath in glob.glob(pattern):
            if os.path.dirname(filepath) != 'visualizations':
                dest = os.path.join('visualizations', os.path.basename(filepath))
                shutil.move(filepath, dest)
                moved_count += 1
                print(f"Moved {filepath} -> {dest}")
    
    if moved_count > 0:
        print(f"\nOrganized {moved_count} files into appropriate directories")
    else:
        print("\nAll files already organized")


def list_saves():
    """List all saved files with information."""
    directories = ['checkpoints', 'visualizations', '.']
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        files = glob.glob(os.path.join(directory, '*'))
        if not files:
            continue
            
        print(f"\n{directory}:")
        print("-" * 60)
        
        total_size = 0
        file_info = []
        
        for filepath in files:
            if os.path.isfile(filepath):
                size_mb, mtime = get_file_info(filepath)
                total_size += size_mb
                file_info.append((os.path.basename(filepath), size_mb, mtime))
        
        # Sort by modification time
        file_info.sort(key=lambda x: x[2], reverse=True)
        
        for name, size, mtime in file_info[:20]:  # Show only top 20
            print(f"  {name:<40} {size:>8.1f} MB   {mtime.strftime('%Y-%m-%d %H:%M')}")
        
        if len(file_info) > 20:
            print(f"  ... and {len(file_info) - 20} more files")
        
        print(f"\n  Total: {len(file_info)} files, {total_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Manage training saves')
    parser.add_argument('action', choices=['list', 'organize', 'cleanup', 'auto'],
                        help='Action to perform')
    parser.add_argument('--keep-models', type=int, default=20,
                        help='Number of model checkpoints to keep (default: 20)')
    parser.add_argument('--keep-images', type=int, default=100,
                        help='Number of visualizations to keep (default: 100)')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_saves()
    
    elif args.action == 'organize':
        organize_saves()
    
    elif args.action == 'cleanup':
        cleanup_old_saves('checkpoints', keep_n=args.keep_models, pattern='*.pth')
        cleanup_old_saves('visualizations', keep_n=args.keep_images, pattern='*.png')
    
    elif args.action == 'auto':
        print("Running automatic organization and cleanup...")
        organize_saves()
        cleanup_old_saves('checkpoints', keep_n=args.keep_models, pattern='*.pth')
        cleanup_old_saves('visualizations', keep_n=args.keep_images, pattern='*.png')
        print("\nDone! Summary:")
        list_saves()


if __name__ == '__main__':
    main()