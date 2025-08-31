from fastapi import FastAPI, UploadFile
import os
import subprocess
from pathlib import Path

app = FastAPI()

# Dossier temporaire pour sauvegarder les fichiers uploadés et les résultats
TMP_DIR = Path("/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Chemin vers le repo TripoSR téléchargé
TRIPOSR_REPO = Path(__file__).parent / "TripoSR-main"
RUN_PY = TRIPOSR_REPO / "run.py"

@app.post("/generate3d")
async def generate3d(file: UploadFile):
    # Sauvegarde de l'image uploadée
    tmp_input = TMP_DIR / "input.png"
    tmp_output_dir = TMP_DIR / "output"
    tmp_output_dir.mkdir(exist_ok=True)

    with open(tmp_input, "wb") as f:
        f.write(await file.read())

    # Appel du script run.py du repo TripoSR
    subprocess.run([
        "python",
        str(RUN_PY),
        str(tmp_input),
        "--output-dir",
        str(tmp_output_dir)
    ], check=True)

    # Récupération du mesh généré (TripoSR crée normalement un .ply dans output/)
    mesh_files = list(tmp_output_dir.glob("*.ply"))
    if not mesh_files:
        return {"error": "Aucun mesh généré par TripoSR."}

    mesh_path = str(mesh_files[0])  # Prend le premier mesh généré

    return {"mesh_path": mesh_path}
