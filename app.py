# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import os
from face_shape_detector import decode_image, analyze_face, system_prompt
import openai
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import json
from fastapi import Request
import mediapipe as mp
import cv2
import numpy as np
import base64


## making a client with API KEY for calling gpt-4o
load_dotenv()

openai_client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS setup (allow all origins or restrict to your domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eyefit-frame-recommendation.netlify.app"],  # frontend url updated
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add session middleware with secret key
app.add_middleware(SessionMiddleware, secret_key="eyefit_frame_recommendation")

# Request model and data validation
class FaceRequest(BaseModel):
    image_base64: str
class LandmarkPoint(BaseModel):
    x: float
    y: float

class FaceLandmarks(BaseModel):
    chin: LandmarkPoint
    forehead: LandmarkPoint
    jaw_l: LandmarkPoint
    jaw_r: LandmarkPoint

class FaceData(BaseModel):
    face_shape: str
    face_width: float
    face_height: float
    aspect_ratio: float
    jaw_width: float
    cheekbone_width: float
    forehead_width: float
    eye_distance: float
    jawline_angle: float
    pitch_angle: float
    roll_angle: float
    skin_tone: str
    landmarks: FaceLandmarks

@app.get("/confirm")
def confirm():
    return {"status": "API working"}



## post request end point for the user data (user face)
@app.post("/analyze_face")
def analyze(req: FaceRequest, request: Request):
    try:
        # Decode the image
        img = decode_image(req.image_base64)

        # --- Your original face analysis result (shape, dimensions, etc.)
        result = analyze_face(img)  # from face_shape_detector

        if "error" in result:
            return JSONResponse(content=result, status_code=400)

        # --- Add MediaPipe full mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        image_np = img.copy()
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image_np)
        h, w, _ = image_np.shape

        if results.multi_face_landmarks:
            mesh_landmarks = [
                {"x": int(lm.x * w), "y": int(lm.y * h)}
                for lm in results.multi_face_landmarks[0].landmark
            ]
            result["mesh_landmarks"] = mesh_landmarks  # ✅ Append full mesh
        else:
            result["mesh_landmarks"] = []

        # Store in session
        request.session["face_data"] = result

        return result

    except Exception as e:
        return JSONResponse(
            content={"error": f"Server error: {str(e)}"},
            status_code=500
        )




# post request end point to post data to LLM for the frame recommendation 
@app.post("/recommend_frame")
async def recommend_frame(data: FaceData):
    face_data = data.model_dump()
    print("✅ Received face data:", face_data)

    # Now safely use face_data for LLM prompt
   

    system_prompt_text = system_prompt()

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": str(face_data)}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        print("🧠 Raw LLM content:\n", content)

        # 🧼 Remove any markdown if present
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        # ✅ Try parsing if it's a string
        if isinstance(content, str):
            parsed = json.loads(content)
        else:
            parsed = content  # already dict (rare in openai lib but check anyway)

        return JSONResponse(content={"success": True, "recommendation": parsed}, status_code=200)

    except Exception as e:
        print("❌ Error in recommend_frame:", e)
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)
