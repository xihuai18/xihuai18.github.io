#!/usr/bin/env python3
"""
OG Image Compression Script (Seamless Version)

This script provides a completely seamless OG image handling experience:
- Users always specify original images in og_image field
- Script automatically creates compressed -og.jpg versions
- Script automatically updates references to use -og.jpg versions

This runs during CI build, so users never need to worry about compression.

Usage:
    python scripts/compress_og_images.py [--dry-run]

Requirements:
    - Pillow (pip install Pillow)
"""

import os
import sys
import re
import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed. Installing...")
    os.system(f"{sys.executable} -m pip install Pillow")
    from PIL import Image


# Configuration
MAX_FILE_SIZE_KB = 500  # Target size in KB (leave some margin below 600KB)
RECOMMENDED_WIDTH = 1200
RECOMMENDED_HEIGHT = 630
JPEG_QUALITY_START = 85
JPEG_QUALITY_MIN = 60
OG_SUFFIX = "-og"  # Suffix for OG image versions


def get_project_root():
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def get_og_image_path(original_path):
    """
    Get the OG image path for a given original image path.
    Example: assets/img/foo/bar.png -> assets/img/foo/bar-og.jpg
    """
    path = Path(original_path)
    return str(path.parent / f"{path.stem}{OG_SUFFIX}.jpg").replace('\\', '/')


def is_og_version(img_path):
    """Check if the image path is already an OG version."""
    return OG_SUFFIX in Path(img_path).stem


def find_og_images(project_root):
    """
    Find all og_image references in markdown files.
    Returns a dict mapping markdown file path to og_image path.
    Only returns references that are NOT already -og versions.
    """
    og_images = {}  # {md_file: og_image_path}
    
    # Search in both _posts/ and _pages/ directories
    search_dirs = [project_root / "_posts", project_root / "_pages"]
    
    # Pattern to match og_image in front matter
    og_image_pattern = re.compile(r'^og_image:\s*(.+)$', re.MULTILINE)
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for md_file in search_dir.glob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = og_image_pattern.findall(content)
                for match in matches:
                    img_path = match.strip().strip('"').strip("'")
                    if img_path.startswith('/'):
                        img_path = img_path[1:]
                    
                    # Skip if already pointing to -og version
                    if is_og_version(img_path):
                        continue
                    
                    og_images[str(md_file)] = img_path
    
    return og_images


def get_file_size_kb(file_path):
    """Get file size in KB."""
    return os.path.getsize(file_path) / 1024


def create_og_image(src_path, dst_path, dry_run=False):
    """
    Create a compressed OG image version.
    Always creates a -og.jpg version, regardless of original size.
    
    Returns:
        tuple: (success, original_size_kb, new_size_kb, message)
    """
    if not os.path.exists(src_path):
        return False, 0, 0, f"Source file not found: {src_path}"
    
    original_size_kb = get_file_size_kb(src_path)
    
    if dry_run:
        return True, original_size_kb, 0, "Would create OG version (dry run)"
    
    try:
        with Image.open(src_path) as img:
            original_mode = img.mode
            
            # Convert RGBA to RGB for JPEG
            if original_mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif original_mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if larger than recommended dimensions
            width, height = img.size
            if width > RECOMMENDED_WIDTH or height > RECOMMENDED_HEIGHT:
                ratio = min(RECOMMENDED_WIDTH / width, RECOMMENDED_HEIGHT / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"  Resized from {width}x{height} to {new_width}x{new_height}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Try different quality levels to achieve target size
            quality = JPEG_QUALITY_START
            new_size_kb = original_size_kb
            
            while quality >= JPEG_QUALITY_MIN:
                img.save(dst_path, 'JPEG', quality=quality, optimize=True)
                new_size_kb = get_file_size_kb(dst_path)
                
                if new_size_kb <= MAX_FILE_SIZE_KB:
                    return True, original_size_kb, new_size_kb, f"Created OG version (quality={quality})"
                
                quality -= 5
            
            # If still too large, try more aggressive resizing
            if new_size_kb > MAX_FILE_SIZE_KB:
                width, height = img.size
                for scale in [0.8, 0.7, 0.6, 0.5]:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    resized.save(dst_path, 'JPEG', quality=JPEG_QUALITY_MIN, optimize=True)
                    new_size_kb = get_file_size_kb(dst_path)
                    
                    if new_size_kb <= MAX_FILE_SIZE_KB:
                        return True, original_size_kb, new_size_kb, f"Created OG version with resize to {new_width}x{new_height}"
            
            # Even if we couldn't get below target, we still created a smaller version
            return True, original_size_kb, new_size_kb, f"Created OG version (best effort: {new_size_kb:.1f}KB)"
            
    except Exception as e:
        return False, original_size_kb, 0, f"Error: {str(e)}"


def update_og_image_reference(md_file, old_path, new_path, dry_run=False):
    """Update og_image reference in a markdown file."""
    if old_path == new_path:
        return False
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_ref = '/' + old_path if not old_path.startswith('/') else old_path
    new_ref = '/' + new_path if not new_path.startswith('/') else new_path
    
    # Only update og_image line
    og_image_pattern = re.compile(r'^(og_image:\s*)' + re.escape(old_ref) + r'(\s*)$', re.MULTILINE)
    
    if og_image_pattern.search(content):
        if dry_run:
            return True
        def replacer(match):
            return match.group(1) + new_ref + match.group(2)
        new_content = og_image_pattern.sub(replacer, content)
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Seamlessly compress OG images for social media sharing')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()
    
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    print(f"Target max size: {MAX_FILE_SIZE_KB}KB")
    print(f"Recommended dimensions: {RECOMMENDED_WIDTH}x{RECOMMENDED_HEIGHT}")
    print()
    
    # Find all OG images (only those not already pointing to -og versions)
    og_images = find_og_images(project_root)
    
    if not og_images:
        print("No og_image references need processing (all already using -og versions or none found)")
        return 0
    
    # Group by unique image paths
    unique_images = {}
    for md_file, img_path in og_images.items():
        if img_path not in unique_images:
            unique_images[img_path] = []
        unique_images[img_path].append(md_file)
    
    print(f"Found {len(unique_images)} unique OG image(s) to process:")
    for img in unique_images:
        print(f"  - {img}")
    print()
    
    # Process each unique image
    errors = 0
    updates = {}  # {old_path: new_path}
    
    for img_rel_path, md_files in unique_images.items():
        print(f"Processing: {img_rel_path}")
        
        src_path = project_root / img_rel_path
        if not src_path.exists():
            print(f"  ✗ Image not found: {src_path}")
            errors += 1
            continue
        
        # Always create -og.jpg version
        og_rel_path = get_og_image_path(img_rel_path)
        og_path = project_root / og_rel_path
        
        success, orig_kb, new_kb, message = create_og_image(src_path, og_path, args.dry_run)
        
        if success:
            print(f"  ✓ {message}")
            if new_kb > 0:
                reduction = (1 - new_kb / orig_kb) * 100 if orig_kb > 0 else 0
                print(f"    Original: {orig_kb:.1f}KB → OG: {new_kb:.1f}KB ({reduction:.1f}% reduction)")
            updates[img_rel_path] = og_rel_path
        else:
            print(f"  ✗ {message}")
            errors += 1
        print()
    
    # Update markdown files
    if updates:
        print("Updating markdown files...")
        for md_file, img_path in og_images.items():
            if img_path in updates:
                new_path = updates[img_path]
                if update_og_image_reference(md_file, img_path, new_path, args.dry_run):
                    print(f"  {'Would update' if args.dry_run else 'Updated'}: {Path(md_file).name}")
        print()
    
    if errors == 0:
        print("✓ All OG images processed successfully!")
    else:
        print(f"⚠ {errors} error(s) occurred")
    
    return errors


if __name__ == '__main__':
    sys.exit(main())
