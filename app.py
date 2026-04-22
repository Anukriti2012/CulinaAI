from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
from typing import List
from pathlib import Path

from imageSimilarityInference import (
    predict_top_k,
    rerank_with_ingredients,
    deduplicate_predictions,
    get_unique_ingredients,
    get_detailed_recipe_data,
    expand_visual_group
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class RerankRequest(BaseModel):
    image_id: str
    ingredients: List[str]

@app.get("/ingredients")
def fetch_ingredients():
    return {"ingredients": get_unique_ingredients()}

@app.post("/predict-image")
async def predict_image(image: UploadFile):
    file_id = f"{uuid.uuid4()}_{image.filename}"
    image_path = UPLOAD_DIR / file_id

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    image_results = predict_top_k(image_path, k=10)

    expanded_results = []

    for r in image_results:
        group_dishes = expand_visual_group(r["dish"])

        for dish in group_dishes:
            expanded_results.append(
                get_detailed_recipe_data(dish, r["score"])
            )

    return {
        "image_id": file_id,
        "similar_dishes": deduplicate_predictions(expanded_results)
    }

@app.post("/rerank")
async def rerank_dishes(request: RerankRequest):
    image_path = UPLOAD_DIR / request.image_id
    if not image_path.exists(): raise HTTPException(status_code=404, detail="Image not found")
    
    image_results = predict_top_k(image_path, k=15)
    final_results = rerank_with_ingredients(image_results, request.ingredients)
    return {"recommended_dishes": deduplicate_predictions(final_results)[:10]}