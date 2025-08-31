from fastapi import FastAPI, UploadFile
from io import BytesIO
from PIL import Image
import os
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["SPCONV_ALGO"] = "native"  # Optionnel mais recommandé

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
if device == "cuda":
    pipeline = pipeline.cuda()

@app.post("/generate3d")
async def generate3d(file: UploadFile):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    outputs = pipeline.run(img, seed=1)
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )
    # Sauvegarde temporaire
    tmp_path = "/tmp/output.glb"
    glb.export(tmp_path)
    return {"glb_path": tmp_path}
