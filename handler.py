from fastapi import FastAPI, UploadFile
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from io import BytesIO
from PIL import Image

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "microsoft/trellis-image-large"

# Important: allow remote code for Trellis
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True
).to(device)

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return {"caption": text}
