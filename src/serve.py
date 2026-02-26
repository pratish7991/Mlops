import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.models.multimodal_model import MultiModalModel

app = FastAPI(title="Multimodal Biometric API")

MODEL_PATH = "runs/latest/model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiModalModel(num_classes=45)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


class InferenceRequest(BaseModel):
    fingerprint: list
    left: list
    right: list


@app.get("/")
def health():
    return {"status": "Model server running"}


@app.post("/predict")
def predict(data: InferenceRequest):
    with torch.no_grad():
        fingerprint = torch.tensor(data.fingerprint).to(device)
        left = torch.tensor(data.left).to(device)
        right = torch.tensor(data.right).to(device)

        output = model(fingerprint, left, right)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
