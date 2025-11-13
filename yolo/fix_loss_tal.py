"""
Simple script to fix the loss_tal.py file
Run this from your project directory
"""

from pathlib import Path
import shutil

def fix_loss_tal():
    """Fix the loss_tal.py file with proper patching"""
    
    loss_file = Path('./yolov9/utils/loss_tal.py')
    
    if not loss_file.exists():
        print(f" File not found: {loss_file}")
        return False
    
    print(f" Fixing {loss_file}...")
    
    # Backup original
    backup_file = loss_file.with_suffix('.py.backup')
    if not backup_file.exists():
        shutil.copy(loss_file, backup_file)
        print(f" Created backup: {backup_file}")
    
    # Read the file
    with open(loss_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove any existing patches and bad indentation
    print(" Cleaning up old patches...")
    cleaned_lines = []
    skip_until_blank = False
    
    for i, line in enumerate(lines):
        # Skip old patch lines
        if 'PATCHED' in line or skip_until_blank:
            if 'PATCHED' in line:
                skip_until_blank = True
            # Keep skipping until we find the actual code line or blank
            if 'pred_distri, pred_scores' in line:
                skip_until_blank = False
                cleaned_lines.append(line)
            elif line.strip() == '' and skip_until_blank:
                skip_until_blank = False
            continue
        
        # Remove lines with just "feats = [f if isinstance..."
        if 'feats = [f if isinstance(f, torch.Tensor)' in line:
            continue
            
        cleaned_lines.append(line)
    
    # Now apply the correct patch
    print(" Applying new patch...")
    final_lines = []
    patched = False
    
    for i, line in enumerate(cleaned_lines):
        # Find the target line
        if 'pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(' in line:
            if not patched:
                # Get the indentation
                indent = len(line) - len(line.lstrip())
                spaces = ' ' * indent
                
                # Add the patch
                final_lines.append(f'{spaces}# PATCHED: Handle YOLOv9 AuxLoss tuple structure\n')
                final_lines.append(f'{spaces}if isinstance(feats, (tuple, list)) and len(feats) > 0:\n')
                final_lines.append(f'{spaces}    if isinstance(feats[0], (tuple, list)):\n')
                final_lines.append(f'{spaces}        feats = feats[0]  # Extract main predictions\n')
                patched = True
        
        final_lines.append(line)
    
    if not patched:
        print("  Could not find the target line to patch!")
        print("    Looking for: pred_distri, pred_scores = torch.cat([xi.view(feats[0]...")
        return False
    
    # Write the fixed file
    with open(loss_file, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    
    print(" File patched successfully!")
    print("\nThe patch adds 4 lines before the pred_distri line:")
    print("  - Checks if feats is a tuple/list")
    print("  - Checks if it contains nested tuples (AuxLoss structure)")
    print("  - Extracts just the main predictions")
    
    return True


def verify_syntax():
    """Try to import the module to verify syntax"""
    print("\n Verifying Python syntax...")
    
    import subprocess
    import sys
    
    result = subprocess.run(
        [sys.executable, '-m', 'py_compile', './yolov9/utils/loss_tal.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(" Syntax is valid!")
        return True
    else:
        print(" Syntax error found:")
        print(result.stderr)
        return False


def main():
    print("=" * 60)
    print("YOLOv9 loss_tal.py Fixer")
    print("=" * 60)
    print()
    
    if fix_loss_tal():
        if verify_syntax():
            print("\n" + "=" * 60)
            print(" SUCCESS! You can now run your training script.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("  File patched but has syntax errors.")
            print("Backup saved as: yolov9/utils/loss_tal.py.backup")
            print("You may need to restore and manually edit.")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(" Failed to patch file.")
        print("Try manually editing yolov9/utils/loss_tal.py")
        print("=" * 60)


if __name__ == "__main__":
    main()