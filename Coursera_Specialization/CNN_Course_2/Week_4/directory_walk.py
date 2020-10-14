import os

root_dir = 'RPS/val/'

for _, _, files in os.walk(root_dir):
    for file in files:
        print(os.path.join(root_dir, file))