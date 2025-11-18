"""
Quick fix for plots.py Pillow compatibility issue
"""

from pathlib import Path

def fix_plots():
    plots_file = Path('./yolov9/utils/plots.py')
    
    if not plots_file.exists():
        print(f" File not found: {plots_file}")
        return False
    
    print(f" Fixing {plots_file}...")
    
    # Read file
    with open(plots_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix the problematic line
    fixed = False
    for i, line in enumerate(lines):
        # Look for the getsize line
        if 'self.font.getsize(label)' in line and 'w, h =' in line:
            print(f"Found getsize at line {i+1}")
            
            # Get indentation
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            
            # Replace the line
            lines[i] = f'{spaces}# PATCHED: Pillow 10+ compatibility - use getbbox instead of getsize\n'
            lines.insert(i+1, f'{spaces}bbox = self.font.getbbox(label)\n')
            lines.insert(i+2, f'{spaces}w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # text width, height\n')
            
            fixed = True
            print(" Patched successfully")
            break
    
    if not fixed:
        print("  getsize line not found or already patched")
        
        # Check if there's a syntax error to fix
        for i, line in enumerate(lines):
            if 'bbox = self.font.getbbox(label)' in line:
                # Check previous line for proper comment
                if i > 0 and '# PATCHED' in lines[i-1]:
                    # Check indentation
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    curr_indent = len(line) - len(line.lstrip())
                    
                    if prev_indent != curr_indent:
                        print(f"  Found indentation mismatch at line {i+1}")
                        print(f"    Previous line indent: {prev_indent}")
                        print(f"    Current line indent: {curr_indent}")
                        
                        # Fix it
                        lines[i] = ' ' * prev_indent + line.lstrip()
                        if i+1 < len(lines):
                            lines[i+1] = ' ' * prev_indent + lines[i+1].lstrip()
                        
                        fixed = True
                        print(" Fixed indentation")
                        break
    
    if fixed:
        # Write back
        with open(plots_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f" File saved: {plots_file}")
        return True
    else:
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv9 plots.py Fixer - Pillow Compatibility")
    print("=" * 60)
    print()
    
    if fix_plots():
        print("\n" + "=" * 60)
        print(" SUCCESS! Run your training script again.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Manual fix needed:")
        print("1. Open: yolov9/utils/plots.py")
        print("2. Find line ~86 with: w, h = self.font.getsize(label)")
        print("3. Replace with:")
        print("   bbox = self.font.getbbox(label)")
        print("   w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]")
        print("=" * 60)