import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import ollama
import numpy as np

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# ==========================================
# 1. MODEL LOADING
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        # Path adjustment: BASE_DIR is the Django project root. 
        # The model is one level up in the Hackathon_model folder.
        weights_path = settings.BASE_DIR / "best_map_model2.pth"
        
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print(f"✅ Loaded weights from {weights_path}")
        else:
            print(f"⚠️ Warning: {weights_path} not found. Using random weights.")
            
        model.to(DEVICE)
        model.eval()
        MODEL = model
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load model on startup
load_model()

# Image Transformations
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ==========================================
# 2. VIEWS
# ==========================================

def index(request):
    return render(request, 'index.html')

def is_likely_map(image):
    """
    Simple heuristic to reject images that are obviously not standard maps
    (e.g., too dark, too saturated/neon, or solid colors).
    """
    # Convert to HSV to check Brightness and Saturation
    hsv_img = image.convert('HSV')
    np_img = np.array(hsv_img)
    
    # Calculate means
    mean_saturation = np_img[:, :, 1].mean()
    mean_brightness = np_img[:, :, 2].mean()
    
    # Heuristic 1: Maps are usually bright (White/Paper background). 
    # Dark images (Space, Night photos) are likely not maps.
    # Threshold: Brightness < 60 (out of 255) is very dark.
    if mean_brightness < 60:
        return False, "Image is too dark to be a standard map."
        
    # Heuristic 2: Maps usually have pastel/muted colors.
    # Super neon/vibrant images (Game logos, Cartoons) have high saturation.
    # Threshold: Saturation > 100 (out of 255) is quite vivid.
    if mean_saturation > 100:
        return False, "colors are too saturated/vibrant for a map."
        
    return True, ""

@csrf_exempt
def analyze_map(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # Open image directly from memory
            img = Image.open(image_file).convert("RGB")
            
            # --- STEP 1: PRE-CHECK for "NOT A MAP" ---
            is_map, heuristic_reason = is_likely_map(img)
            if not is_map:
                # If heuristic says it's not a map, return immediately
                return JsonResponse({
                    'status': 'success', 
                    'label': "NOT A MAP", 
                    'reason': heuristic_reason,
                    'confidences': {
                        "NOT A MAP": 100.00,
                        "UNCERTAIN": 0.00
                    }
                })

            # --- STEP 2: AI INFERENCE ---
            # Transform
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # Inference
            global MODEL
            if MODEL is None:
                load_model()
            
            with torch.no_grad():
                outputs = MODEL(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # New Model Logic (best_map_model2.pth): 0=Good, 1=Bad
            good_prob = float(probs[0])
            bad_prob = float(probs[1])
            
            # ----------------------------------------------------
            # "NOT A MAP" DETECTION (Heuristic via Confidence)
            # ----------------------------------------------------
            confidence_threshold = 0.65
            
            reason = ""
            confidences = {}
            
            if max(good_prob, bad_prob) < confidence_threshold:
                label = "NOT A MAP"
                reason = "Low model confidence. Image lacks standard map features."
                confidences = {
                    "NOT A MAP": round((1.0 - max(good_prob, bad_prob)) * 100, 2),
                    "UNCERTAIN": round(max(good_prob, bad_prob) * 100, 2)
                }
            else:
                if good_prob >= bad_prob:
                    label = "GOOD MAP"
                    reason = "High clarity, distinct labels, and correct geometry detection."
                else:
                    label = "BAD MAP"
                    reason = "Detected distorted lines, missing labels, or unclear topology."
                
                confidences = {
                    "GOOD MAP": round(good_prob * 100, 2),
                    "BAD MAP": round(bad_prob * 100, 2)
                }
            
            return JsonResponse({
                'status': 'success', 
                'label': label, 
                'reason': reason,
                'confidences': confidences
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

@csrf_exempt
def chat_bot(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            history = data.get('history', []) 
            
            system_prompt = (
                "You are GeoBot, an enthusiastic AI assistant for 'GeoDetector'.\n"
                "Your mission: Help users understand map quality and cartography.\n"
                "CRITICAL: Keep your answer extremely concise and limited to exactly one short paragraph only. Do not use bullet points."
            )
            
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(history)
            messages.append({'role': 'user', 'content': user_message})
            
            response = ollama.chat(model='llama3.2', messages=messages)
            bot_reply = response['message']['content']
            
            return JsonResponse({'status': 'success', 'response': bot_reply})
            
        except Exception as e:
            print(f"Chat Error: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return JsonResponse({'status': 'error', 'message': 'Only POST allowed'})


