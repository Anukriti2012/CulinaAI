import os
import json
import torch
import pickle
import numpy as np
import re

from typing import Union, List, Dict
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from transformers import ViTImageProcessor, ViTModel
from sklearn.metrics.pairwise import cosine_similarity

GLOBAL_SUBSTITUTES = {
    "milk": "Oat milk, Soy milk, or Almond milk (for dairy-free).",
    "yogurt": "Greek yogurt, Hung curd, or Coconut yogurt.",
    "sugar": "Honey, Jaggery, Stevia, or Maple syrup.",
    "butter": "Ghee, Coconut oil, or Margarine.",
    "potatoes": "Sweet potatoes, Cauliflower (for low carb), or Yam.",
    "rice": "Quinoa, Cauliflower rice, or Brown rice.",
    "mango": "Peach, Nectarine, or Papaya (similar texture).",
    "eggs": "Flaxseed meal (1 tbsp + 3 tbsp water) or Applesauce.",
    "paneer": "Tofu or Halloumi cheese.",
    "all purpose flour": "Whole wheat flour or Almond flour (gluten-free)."
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model" / "culinaai_modelTrainedGrouped"
DATASET_DIR = BASE_DIR / "NewCuisineDatasetGrouped"
EMBEDDING_PKL = BASE_DIR / "culinaaiGrouped_embeddings2.pkl"
RECIPE_JSON_PATH = BASE_DIR / "indian_recipes_updated.json"
GROUP_JSON_PATH = BASE_DIR / "visual_groups.json"

processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
vit = ViTModel.from_pretrained(MODEL_DIR).to(DEVICE)
vit.eval()

with open(RECIPE_JSON_PATH, "r", encoding="utf-8") as f:
    recipes = json.load(f)

with open(GROUP_JSON_PATH, "r") as f:
    visual_groups = json.load(f)

def normalize_name(name: str) -> str:
    return name.lower().strip().replace(" ", "_")

recipe_lookup: Dict[str, dict] = {normalize_name(r["name"]): r for r in recipes}
dish_to_ingredients = {name: set(i.lower() for i in r.get("ingredients_clean", [])) for name, r in recipe_lookup.items()}

def get_detailed_recipe_data(dish_name, score):
    normalized = normalize_name(dish_name)
    recipe_data = recipe_lookup.get(normalized)
    
    if not recipe_data:
        return {
            "dish": dish_name, 
            "ingredients": [], 
            "instructions": "No data found.", 
            "substitutes": "No data.", 
            "nutrition": {"calories": 0, "protein": 0, "fat": 0}
        }

    # Extract ingredients list
    ingredients = recipe_data.get("ingredients_clean", [])
    
    # Process Substitutes
    found_subs = []
    for ing in ingredients:
        ing_lower = ing.lower()
        if ing_lower in GLOBAL_SUBSTITUTES:
            found_subs.append(f"For {ing}: {GLOBAL_SUBSTITUTES[ing_lower]}")
    
    sub_string = " | ".join(found_subs) if found_subs else "No specific substitutes found."

    raw_cal = recipe_data.get("calories", "0")
    cal_match = re.search(r"(\d+\.?\d*)", str(raw_cal))
    calories = float(cal_match.group(1)) if cal_match else 0

    return {
        "dish": recipe_data["name"],
        "ingredients": ingredients, # ADDED THIS LINE
        "instructions": recipe_data.get("directions", "No instructions available."),
        "nutrition": {"calories": calories, "protein": 15, "fat": 10},
        "substitutes": sub_string,
        "score": float(score)
    }

def get_image_embedding(image_path: Union[str, Path]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = vit(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding / np.linalg.norm(embedding)

def predict_top_k(image_path, k=15):
    if not hasattr(predict_top_k, "embeddings"):
        with open(EMBEDDING_PKL, "rb") as f:
            data = pickle.load(f)
            predict_top_k.embeddings = data["embeddings"]
            predict_top_k.labels = data["labels"]
            predict_top_k.paths = data["paths"]

    query_emb = get_image_embedding(image_path)
    sims = cosine_similarity([query_emb], predict_top_k.embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:k]

    return [{"dish": predict_top_k.labels[i], "score": float(sims[i])} for i in top_idx]

def rerank_with_ingredients(image_results, user_ingredients, alpha=0.6, beta=0.4):
    final_results = []
    user_set = set(i.lower() for i in user_ingredients)

    for r in image_results:
        label = normalize_name(r["dish"])
        candidates = [normalize_name(d) for d in visual_groups.get(label, [label])]

        for dish_key in candidates:
            recipe_data = recipe_lookup.get(dish_key)
            if not recipe_data: continue

            dish_ing = dish_to_ingredients.get(dish_key, set())
            ing_score = len(user_set.intersection(dish_ing)) / len(dish_ing) if dish_ing else 0
            final_score = (alpha * r["score"]) + (beta * ing_score)
            
            detail = get_detailed_recipe_data(recipe_data["name"], r["score"])
            detail["final_score"] = round(float(final_score), 3)
            final_results.append(detail)

    return sorted(final_results, key=lambda x: x["final_score"], reverse=True)

def deduplicate_predictions(results):
    best = {}
    for r in results:
        dish = r["dish"]
        score = r.get("final_score", r.get("score", 0))
        if dish not in best or score > best[dish].get("final_score", 0):
            best[dish] = r
    return sorted(best.values(), key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)

def get_unique_ingredients():
    ingredients = set()
    for r in recipes:
        for ing in r.get("ingredients_clean", []):
            ingredients.add(ing.strip().lower())
    return sorted(list(ingredients))

def expand_visual_group(label: str):
    label = normalize_name(label)
    return visual_groups.get(label, [label])