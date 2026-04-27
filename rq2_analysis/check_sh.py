import os

paths_to_check = [
    r"C:\Program Files\Git\bin\sh.exe",
    r"C:\Program Files\Git\usr\bin\sh.exe",
    r"C:\Program Files\Git\bin",
    r"C:\Program Files\Git\usr\bin"
]

for p in paths_to_check:
    exists = os.path.exists(p)
    print(f"Path: {p} | Exists: {exists}")
