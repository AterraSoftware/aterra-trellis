from fastapi import FastAPI, UploadFile
from io import BytesIO
import os
import torch
import numpy as np
import open3d as o3d

# Import depuis le repo SPAR3D (installer depuis GitHub dans requirements.txt)
from stable_point_aware_3d.spar3d.model import SPAR3DModel

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["SPCONV_ALGO"] = "native"  # Optionnel mais recommandé

# Charger le modèle SPAR3D depuis le repo ou checkpoint
model = SPAR3DModel.from_pretrained("Stability-AI/stable-point-aware-3d")  # repo GitHub
model.to(device)
model.eval()

@app.post("/generate3d")
async def generate3d(file: UploadFile):
    # Charger l'image depuis la requête
    img_bytes = await file.read()
    img_o3d = o3d.io.read_image(BytesIO(img_bytes))  # Open3D Image
    img_np = np.asarray(img_o3d)  # Convertir en numpy array si nécessaire par le modèle
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # Ajouter batch dim

    # Exécution du modèle
    with torch.no_grad():
        mesh_output = model(img_tensor)  # adapter selon le forward exact de SPAR3D

    # Sauvegarde temporaire du mesh
    tmp_path = "/tmp/output.ply"
    o3d.io.write_triangle_mesh(tmp_path, mesh_output)

    return {"mesh_path": tmp_path}
