import tomllib  # built-in in Python 3.11+

with open(r"C:\Users\Carlo\Downloads\uv.lock", "rb") as f:
    data = tomllib.load(f)

for pkg in data.get("package", []):
    name = pkg["name"]
    version = pkg["version"]
    print(f"pip install {name}=={version}")