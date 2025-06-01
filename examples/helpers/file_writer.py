import json

def empty(file_path):
    with open(file_path, 'w') as f:
        pass

def write(file_path, content):
    with open(file_path, "a") as f:
        json.dump(content, f, indent=2)
        f.write("\n")