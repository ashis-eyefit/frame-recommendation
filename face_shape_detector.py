import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
from PIL import Image
import base64
from sqldb import get_db

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Landmark indices
JAW_LEFT, JAW_RIGHT = 234, 454
CHIN, FOREHEAD = 152, 10
LEFT_CHEEK, RIGHT_CHEEK = 127, 356
LEFT_FOREHEAD, RIGHT_FOREHEAD = 108, 338
EYE_LEFT, EYE_RIGHT = 33, 263

def decode_image(base64_string):
    img_bytes = base64.b64decode(base64_string.split(",")[-1])
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_landmark_coords(landmarks, width, height, idx):
    pt = landmarks.landmark[idx]
    return int(pt.x * width), int(pt.y * height)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def analyze_face(image_bgr):
    h, w, _ = image_bgr.shape
    results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # No face detected
    if not results.multi_face_landmarks:
        return {"error": "No face detected. Please face the camera directly."}

    # More than one face
    if len(results.multi_face_landmarks) > 1:
        return {"error": "Multiple faces detected. Please ensure only one person is in the frame."}

    lm = results.multi_face_landmarks[0]
    def get(idx): return get_landmark_coords(lm, w, h, idx)

    # Landmarks
    chin, forehead = get(CHIN), get(FOREHEAD)
    jaw_l, jaw_r = get(JAW_LEFT), get(JAW_RIGHT)
    cheek_l, cheek_r = get(LEFT_CHEEK), get(RIGHT_CHEEK)
    fore_l, fore_r = get(LEFT_FOREHEAD), get(RIGHT_FOREHEAD)
    eye_l, eye_r = get(EYE_LEFT), get(EYE_RIGHT)

    # Dimensions
    face_w = calculate_distance(jaw_l, jaw_r)
    face_h = calculate_distance(chin, forehead)
    cheekbone_w = calculate_distance(cheek_l, cheek_r)
    jaw_w = face_w
    forehead_w = calculate_distance(fore_l, fore_r)
    eye_dist = calculate_distance(eye_l, eye_r)
    aspect_ratio = face_h / face_w if face_w != 0 else 0

    # Angles
    jawline_angle = np.degrees(np.arctan2(jaw_r[1] - jaw_l[1], jaw_r[0] - jaw_l[0]))  # Yaw
    pitch_angle = np.degrees(np.arctan2(chin[1] - forehead[1], chin[0] - forehead[0]))  # Tilt
    roll_angle = np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))  # Side tilt

    # Skin tone (LAB space)
    try:
        patch = image_bgr[forehead[1]-10:forehead[1]+10, forehead[0]-10:forehead[0]+10]
        avg_color = cv2.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2Lab))[:3]
        L = avg_color[0]
        skin_tone = "light" if L > 70 else "medium" if L > 40 else "dark"
    except:
        skin_tone = "undetected"

    # Face shape classification
    def classify_face_shape():
        def close(a, b, tol=3): return abs(a - b) <= tol
        angle_abs = abs(jawline_angle)
        if angle_abs > 12: return "Not Centered"
        if close(jaw_w, cheekbone_w) and close(cheekbone_w, forehead_w):
            return "Round" if aspect_ratio <= 1.05 and angle_abs <= 4 else "Oval"
        if cheekbone_w > jaw_w + 5 and cheekbone_w > forehead_w + 5 and angle_abs >= 5:
            return "Diamond"
        if forehead_w > cheekbone_w and jaw_w < cheekbone_w and angle_abs <= 6:
            return "Heart"
        if jaw_w > cheekbone_w and close(forehead_w, jaw_w) and angle_abs >= 5:
            return "Square"
        if jaw_w > cheekbone_w and forehead_w < jaw_w - 5:
            return "Triangle"
        return "Oval"

    # Final sanity checks before response
    if face_w < 80 or face_h < 100 or eye_dist < 40 or eye_dist > 180:
        return {"error": "Face not properly framed or resolution too low. Please try again."}

    return {
        "face_shape": classify_face_shape(),
        "face_width": round(face_w, 2),
        "face_height": round(face_h, 2),
        "aspect_ratio": round(aspect_ratio, 2),
        "jaw_width": round(jaw_w, 2),
        "cheekbone_width": round(cheekbone_w, 2),
        "forehead_width": round(forehead_w, 2),
        "eye_distance": round(eye_dist, 2),
        "jawline_angle": round(jawline_angle, 2),
        "pitch_angle": round(pitch_angle, 2),
        "roll_angle": round(roll_angle, 2),
        "skin_tone": skin_tone,
        "landmarks": {
            "chin": {"x": chin[0], "y": chin[1]},
            "forehead": {"x": forehead[0], "y": forehead[1]},
            "jaw_l": {"x": jaw_l[0], "y": jaw_l[1]},
            "jaw_r": {"x": jaw_r[0], "y": jaw_r[1]}
        }
    }


## system prompt for LLM
def system_prompt():
    return """
        You are a commercial eyewear stylist with extensive knowlendge and provide perfect designed recommendation.

        Based on the user's facial geometry, skin tone, and other parameters, recommend the most suitable frame style and at least two ideal color options. Use only the frame and color styles from the lists below.

        Keep the reasoning summary very short and pointwise (no paragraphs). Do not include any extra commentary, formatting, markdown, or explanations.

        Available Frame Styles:
        - Cateye
        - Aviator
        - Clubmaster
        - Rectangle
        - Oval
        - Retro Aviator
        - Frameless
        - Butterfly

        Available Colors:
        - Black
        - Brown
        - Gunmetal Grey
        - Matte Blue
        - Transparent
        - Gold
        - Rose Gold
        - Tortoise Shell

        Respond strictly in raw JSON format (no markdown or backticks):

        {
        "recommended_frame": "<frame from list>",
        "recommended_color": ["<color1>", "<color2>", ...],
        "reasoning summary": [
            "<reason 1>",
            "<reason 2>",
            "<reason 3>"
        ]
        }

        """

### input data base for face analysis

def insert_face_data(face_data):
    conn = get_db()
    cursor = conn.cursor()

    query = """
        INSERT INTO face_analysis (
            face_shape, face_width, face_height, aspect_ratio, jaw_width,
            cheekbone_width, forehead_width, eye_distance, jawline_angle,
            pitch_angle, roll_angle, skin_tone,
            chin_x, chin_y, forehead_x, forehead_y,
            jaw_l_x, jaw_l_y, jaw_r_x, jaw_r_y
        ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    landmarks = face_data["landmarks"]
    values = (
        face_data["face_shape"],
        face_data["face_width"],
        face_data["face_height"],
        face_data["aspect_ratio"],
        face_data["jaw_width"],
        face_data["cheekbone_width"],
        face_data["forehead_width"],
        face_data["eye_distance"],
        face_data["jawline_angle"],
        face_data["pitch_angle"],
        face_data["roll_angle"],
        face_data["skin_tone"],
        landmarks["chin"]["x"], landmarks["chin"]["y"],
        landmarks["forehead"]["x"], landmarks["forehead"]["y"],
        landmarks["jaw_l"]["x"], landmarks["jaw_l"]["y"],
        landmarks["jaw_r"]["x"], landmarks["jaw_r"]["y"],
    )

    cursor.execute(query, values)
    conn.commit()
    inserted_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return inserted_id


#### output data of frame recomendation

def insert_frame_recommendation(face_id, frame_result):
    if face_id is None:
        raise ValueError("❌ face_id (face_analysis_id) is None. Cannot insert recommendation.")

    conn = get_db()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO frame_recommendation (
            face_analysis_id, recommended_frame, recommended_color, reasoning_summary
        ) VALUES (%s, %s, %s, %s)
    """

    values = (
        face_id,
        frame_result["recommended_frame"],
        ", ".join(frame_result["recommended_color"]),
        "\n".join(frame_result["reasoning summary"]),
    )

    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()
