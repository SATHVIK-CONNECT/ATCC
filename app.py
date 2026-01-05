import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from doctr.models import ocr_predictor
import re
import tempfile

# --- Page Configuration ---
st.set_page_config(page_title="ANPR & Helmet Enforcement", layout="wide")
st.title("🏍️ ANPR & Helmet Violation Detection System")

# --- Load Models ---
@st.cache_resource
def load_models():
    # Load YOLOv8 model (Ensure classes: motorcycle, helmet, no-helmet, plate)
    yolo = YOLO("runs/detect/train/weights/best.pt") 
    ocr = ocr_predictor(pretrained=True)
    return yolo, ocr

yolo_model, ocr_model = load_models()

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Select Input Mode", ["Image", "Video"])
    conf_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.45)
    if mode == "Video":
        crop_padding = st.slider("OCR Crop Padding", 0, 200, 30)
        frame_skip = st.slider("Frame Skip", 1, 30, 10)

    st.info("""
    **Label Logic:**
    - Green: Helmet Detected
    - Red: No-Helmet Detected OR No Helmet Found on Bike
    """)

# --- Helper: Hierarchical Logic ---
def get_bike_status(bike_box, all_boxes, names):
    """
    Determines if a bike has a helmet violation.
    Returns: is_safe (bool), color (hex), text (str)
    """
    bx1, by1, bx2, by2 = map(int, bike_box.xyxy[0])
    
    has_helmet = False
    has_violation_class = False # specifically "no-helmet" or "head"

    for box in all_boxes:
        label = names[int(box.cls[0])].lower()
        # Calculate center of the object
        ox_c = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        oy_c = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        
        # Check if object is inside the Bike Bounding Box
        if bx1 <= ox_c <= bx2 and by1 <= oy_c <= by2:
            if "no helmet" in label or "head" in label:
                has_violation_class = True
            elif "helmet" in label:
                has_helmet = True

    # Logic: Red if a bare head is seen OR if no helmet is found at all
    is_safe = has_helmet and not has_violation_class
    
    if is_safe:
        return True, "#28a745", "HELMET DETECTED"
    else:
        return False, "#dc3545", "VIOLATION: NO HELMET"

def check_helmet_status(boxes, names):
    """
    Scans detected boxes for helmet/no-helmet classes.
    Returns: status (bool), color (BGR tuple)
    """
    helmet_detected = False
    no_helmet_detected = False
    
    for box in boxes:
        label = names[int(box.cls[0])].lower()
        if "no-helmet" in label or "head" in label:
            no_helmet_detected = True
        elif "helmet" in label:
            helmet_detected = True
            
    # Priority: If any 'no-helmet' is seen, mark as Red. 
    # If only helmets are seen, mark Green.
    if no_helmet_detected:
        return False, (0, 0, 255) # Red
    elif helmet_detected:
        return True, (0, 255, 0)  # Green
    else:
        return None, (128, 128, 128) # Grey (Unknown)



def perform_ocr(image_np, boxes, names, padding):
    plates = []
    h, w, _ = image_np.shape
    
    # 1. Determine Helmet Status for the frame
    has_helmet, status_color = check_helmet_status(boxes, names)
    
    for box in boxes:
        if "plate" in names[int(box.cls[0])].lower():
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Padding
            y_min, y_max = max(0, y1 - padding), min(h, y2 + padding)
            x_min, x_max = max(0, x1 - padding), min(w, x2 + padding)
            
            crop = image_np[y_min:y_max, x_min:x_max]
            out = ocr_model([crop])
            export = out.export()
            
            text = "".join([word['value'] for page in export['pages'] 
                            for block in page['blocks'] 
                            for line in block['lines'] 
                            for word in line['words']])
            
            cleaned_text = clean_plate_text(text)
            if len(cleaned_text) > 2:
                plates.append({
                    "crop": crop, 
                    "text": cleaned_text, 
                    "conf": conf_val,
                    "has_helmet": has_helmet,
                    "color": status_color
                })
    return plates


# --- Main Logic ---
uploaded_file = st.file_uploader(f"Upload {mode}", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])

if uploaded_file:
    best_detections = {}

    if mode == "Image":
        img_np = np.array(Image.open(uploaded_file).convert("RGB"))
        results = yolo_model.predict(img_np, conf=conf_threshold)
        names = yolo_model.names
        boxes = results[0].boxes

        # Filter detections
        bikes = [b for b in boxes if names[int(b.cls[0])].lower() in ["motorcycle", "bike"]]
        plates = [b for b in boxes if "number plate" in names[int(b.cls[0])].lower()]

        st.image(results[0].plot(), caption="Detection Overview", use_container_width=True)

        if not bikes:
            st.warning("No motorcycles detected.")
        
        for b_idx, bike in enumerate(bikes):
            is_safe, status_color, status_text = get_bike_status(bike, boxes, names)
            
            # Find plates belonging to this bike
            bx1, by1, bx2, by2 = map(int, bike.xyxy[0])
            bike_plates = []
            for p in plates:
                px_c, py_c = (p.xyxy[0][0] + p.xyxy[0][2]) / 2, (p.xyxy[0][1] + p.xyxy[0][3]) / 2
                if bx1 <= px_c <= bx2 and by1 <= py_c <= by2:
                    bike_plates.append(p)

            for p_idx, p_box in enumerate(bike_plates):
                st.divider()
                st.markdown(f"### Bike {b_idx+1} | <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
                
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                padding_options = [100]
                instance_results = []

                # Multi-padding evaluation
                with st.spinner(f"Running OCR for Plate {p_idx+1}..."):
                    for pad in padding_options:
                        y_min, y_max = max(0, py1-pad), min(img_np.shape[0], py2+pad)
                        x_min, x_max = max(0, px1-pad), min(img_np.shape[1], px2+pad)
                        crop = img_np[y_min:y_max, x_min:x_max]
                        
                        out = ocr_model([crop])
                        words = [w['value'] for pg in out.export()['pages'] for bl in pg['blocks'] for ln in bl['lines'] for w in ln['words']]
                        confs = [w['confidence'] for pg in out.export()['pages'] for bl in pg['blocks'] for ln in bl['lines'] for w in ln['words']]
                        
                        text = clean_plate_text("".join(words))
                        if len(text) > 2:
                            instance_results.append({"pad": pad, "text": text, "conf": np.mean(confs) if confs else 0, "crop": crop})

                if instance_results:
                    top_3 = sorted(instance_results, key=lambda x: x['conf'], reverse=True)[:3]
                    cols = st.columns(3)
                    for i, res in enumerate(top_3):
                        with cols[i]:
                            st.markdown(f"""
                            <div style="background-color:{status_color}; padding:10px; border-radius:5px; text-align:center; color:white;">
                                <h2 style="margin:0;">{res['text']}</h2>
                                <p style="margin:0; font-size:0.8em;">Rank {i+1} (Pad {res['pad']})</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(res['crop'], use_container_width=True)
                else:
                    st.error(f"Plate {p_idx+1} detected but OCR failed at all padding levels.")

    elif mode == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty() 
        st_info = st.empty()   
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = yolo_model.predict(frame_rgb, conf=conf_threshold)
                
                annotated_frame = results[0].plot()
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Check for plates and update best detections
                current_plates = perform_ocr(frame_rgb, results[0].boxes, yolo_model.names, crop_padding)
                for p in current_plates:
                    plate_txt = p['text']
                    # Store if new plate or if current detection has higher confidence
                    if plate_txt not in best_detections or p['conf'] > best_detections[plate_txt]['conf']:
                        best_detections[plate_txt] = {
                            "crop": p['crop'],
                            "conf": p['conf']
                        }

        cap.release()

        # --- Display Summary After Video Completion ---
        st.write("---")
        st.subheader("🏁 Final Video Analysis Summary (Top 3 Detections)")
        
        if best_detections:
            # 1. Sort detections by confidence in descending order
            # 2. Slice the list to get only the top 3
            sorted_plates = sorted(
                best_detections.items(), 
                key=lambda item: item[1]['conf'], 
                reverse=True
            )[:3]

            st.markdown("### Highest Confidence Unique Plates:")
            
            # Create columns for a cleaner Top 3 display
            cols = st.columns(3)
            
            for idx, (plate_no, data) in enumerate(sorted_plates):
                with cols[idx]:
                    # Display in a green success box
                    st.success(f"**Rank {idx+1}**")
                    st.image(data['crop'], use_container_width=True)
                    st.markdown(f"""
                    **Plate:** `{plate_no}`  
                    **Confidence:** {data['conf']:.2%}
                    """)
        else:
            st.warning("No number plates were clearly identified in the video.")
