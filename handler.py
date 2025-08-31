from fastapi import FastAPI, UploadFile
from io import BytesIO
import os
import sys  # <-- ajouter ceci
import torch
import numpy as np
import open3d as o3d
from PIL import Image

# Ajouter le repo local TripoSR
sys.path.append(os.path.join(os.path.dirname(__file__), "TripoSR-main"))
from model import TripoSRModel

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le modèle TripoSR depuis un checkpoint ou repo
model = TripoSRModel.from_pretrained("VAST-AI-Research/TripoSR")  # vérifier le nom exact
model.to(device)
model.eval()

@app.post("/generate3d")
async def generate3d(file: UploadFile):
    # Charger l'image depuis la requête
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalisation
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,H,W)

    # Exécution du modèle
    with torch.no_grad():
        mesh_output = model(img_tensor)  # Adapter selon le forward exact de TripoSR

    # Sauvegarde temporaire du mesh
    tmp_path = "/tmp/output.ply"
    if isinstance(mesh_output, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(tmp_path, mesh_output)
    else:
        # si le modèle renvoie un dict avec vertices/faces
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_output['vertices'])
        mesh.triangles = o3d.utility.Vector3iVector(mesh_output['faces'])
        o3d.io.write_triangle_mesh(tmp_path, mesh)

    return {"mesh_path": tmp_path}
