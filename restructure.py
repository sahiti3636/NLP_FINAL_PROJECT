import os
import glob
import shutil
import re

directories = ['pipeline_steps', 'tests_and_fixes', 'results_images', 'jsons', 'data_models']
for d in directories:
    os.makedirs(d, exist_ok=True)

# Step 1: define mapping rules
def get_dest_dir(f):
    if f.endswith('.json'):
        return 'jsons'
    if f.endswith('.png'):
        return 'results_images'
    if f.endswith('.npy') or f.endswith('.pkl') or f.endswith('.pt'):
        return 'data_models'
    if f.startswith('test_') or f.startswith('ablation_') or f.startswith('fix_') or f == 'log_ablation_emotion.txt':
        return 'tests_and_fixes'
    if f.startswith('step') and f.endswith('.py') and f not in ['step10_emotion_profiles.json']:
        return 'pipeline_steps'
    return None

import logging
logging.basicConfig(level=logging.INFO)

files = [f for f in os.listdir('.') if os.path.isfile(f) and get_dest_dir(f)]

file_moves = []
path_replacements = {}

for f in files:
    dest_dir = get_dest_dir(f)
    if dest_dir:
        file_moves.append((f, os.path.join(dest_dir, f)))
        path_replacements[f] = f"{dest_dir}/{f}"

# Refactor all .py files (including the ones currently in root before they move, and ones already moved? We'll just refactor them in place, then move them)
py_files = glob.glob('*.py')

for py_file in py_files:
    if py_file == 'restructure.py':
        continue
    with open(py_file, 'r') as file:
        content = file.read()
    
    modified = False
    for old_val, new_val in path_replacements.items():
        # Replace explicitly quoted filenames
        # e.g., 'ablation_emotion_results.png' -> 'results_images/ablation_emotion_results.png'
        pattern1 = f"'{old_val}'"
        pattern2 = f'"{old_val}"'
        
        if pattern1 in content:
            content = content.replace(pattern1, f"'{new_val}'")
            modified = True
            
        if pattern2 in content:
            content = content.replace(pattern2, f'"{new_val}"')
            modified = True
            
    if modified:
        with open(py_file, 'w') as file:
            file.write(content)
            logging.info(f"Updated paths in {py_file}")

# Now move the files
for src, dest in file_moves:
    if os.path.exists(src):
        shutil.move(src, dest)
        logging.info(f"Moved {src} to {dest}")

print("Restructure complete!")
