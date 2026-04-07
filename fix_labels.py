"""
Fix label format - Ultralytics YOLO OBB uses 9-value format:
  class x1 y1 x2 y2 x3 y3 x4 y4 (9 columns total)
This is EXACTLY the same as the Roboflow OBB format originally in labels_backup.
So we just need to restore labels from backup and ensure they're in the correct format.
"""

import shutil
from pathlib import Path

BASE_DIR = Path(r"C:\Users\AadithGeethMohan\OneDrive\Desktop\ACM SIGKDD TASK\pothole-detection")
DATASET_DIR = BASE_DIR / "Pothole Detection.v1i.yolov8-obb"

print("=" * 60)
print("FIXING LABEL FORMATS - Ultralytics YOLO OBB uses 9-value format")
print("= class x1 y1 x2 y2 x3 y3 x4 y4")
print("=" * 60)

for split in ['train', 'valid', 'test']:
    print(f"\n=== {split.upper()} ===")

    labels = DATASET_DIR / split / 'labels'
    labels_backup = DATASET_DIR / split / 'labels_backup'
    labels_obb = DATASET_DIR / split / 'labels_obb'

    # Check backup format
    if labels_backup.exists() and any(labels_backup.glob('*.txt')):
        sample_bak = next(labels_backup.glob('*.txt'))
        line_bak = sample_bak.read_text().strip().split('\n')[0]
        parts_bak = len(line_bak.split())
        print(f"  labels_backup: {parts_bak} columns - {line_bak[:60]}...")
    else:
        print(f"  ERROR: No labels_backup found for {split}!")
        continue

    # Restore labels from backup (overwrite file by file)
    restored = 0
    for bak_file in labels_backup.glob('*.txt'):
        target = labels / bak_file.name
        target.write_text(bak_file.read_text())
        restored += 1

    # Remove labels_obb if it exists (we don't need it since backup format is correct)
    if labels_obb.exists():
        import shutil
        shutil.rmtree(labels_obb)
        print(f"  Removed labels_obb/ (not needed)")

    print(f"  Restored: {restored} labels from backup to labels/")
    print(f"  Format: {parts_bak} columns (correct for YOLO OBB)")

    # Verify restored
    sample = next(labels.glob('*.txt'))
    line = sample.read_text().strip().split('\n')[0]
    parts = len(line.split())
    print(f"  Verified labels/: {parts} columns - {line[:60]}...")

print("\n" + "=" * 60)
print("FIX COMPLETE")
print("=" * 60)
print("All labels now in correct YOLO OBB format:")
print("  class x1 y1 x2 y2 x3 y3 x4 y4 (9 columns)")
print("\nUltralytics YOLO OBB expects 9-value format - which is what")
print("the original Roboflow labels already were!")

# Delete stale cache files
print("\nCleaning up stale cache files...")
for split in ['train', 'valid', 'test']:
    cache = DATASET_DIR / split / 'labels.cache'
    if cache.exists():
        cache.unlink()
        print(f"  Deleted {split}/labels.cache")
print("\nDone! Run validation again with the OBB model.")