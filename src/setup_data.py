import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "data" / "raw"
if not (target / "pallets__flask").exists():
    target.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(root / "data" / "snapshot" / "pallets__flask.zip").extractall(target)
    print("Unpacked snapshot")