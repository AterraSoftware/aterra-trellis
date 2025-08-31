from fastapi import FastAPI, UploadFile
from io import BytesIO
import os
import torch
import numpy as np
import open3d as o3d
from spar3d.model import SPAR3DModel  # À adapter selon le repo exact

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["SPCONV_ALGO"] = "native"  # Optionnel mais recommandé

# Charger le modèle SPAR3D depuis le repo ou checkpoint
model = SPAR3DModel.from_pretrained("Stability-AI/spar3d")  # vérifier le nom exact
model.to(device)
model.eval()

@app.post("/generate3d")
async def generate3d(file: UploadFile):
    # Charger l'image depuis la requête
    img_bytes = await file.read()
    img = np.array(o3d.io.read_image(BytesIO(img_bytes)))  # selon SPAR3D, adapter le loader

    # Exécution du modèle
    with torch.no_grad():
        mesh_output = model(img)  # adapter selon le forward du modèle SPAR3D

    # Sauvegarde temporaire du mesh
    tmp_path = "/tmp/output.ply"
    o3d.io.write_triangle_mesh(tmp_path, mesh_output)

    return {"mesh_path": tmp_path}
