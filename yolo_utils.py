from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model_path = "models/yolo.pt"
model = YOLO(model_path)  # charge le modèle une fois
def get_labels():
    """
    Retourne les classes détectées par le modèle YOLO.
    """
    if not hasattr(model, 'names'):
        raise ValueError("Le modèle n'a pas de noms de classes définis.")
    return model.names
def apply_model(image: Image.Image, threshold: float = 0.5, return_labels=False):
    if not isinstance(image, Image.Image):
        raise ValueError("L'entrée n'est pas une image PIL valide")

    image_np = np.array(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"L'image doit avoir 3 canaux (RGB), mais a {image_np.shape}")

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)[0]

    boxes_coords = []
    labels = []

    for box in results.boxes:
        conf = float(box.conf)
        if conf < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes_coords.append([x1, y1, x2, y2])

        label = int(box.cls)
        labels.append(model.names[label])
        label_text = f"{model.names[label]} {conf:.2f}"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    image_rgb_out = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb_out)

    if return_labels:
        return pil_image, boxes_coords, labels
    return pil_image, boxes_coords
