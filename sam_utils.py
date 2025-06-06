from ultralytics import SAM
import numpy as np
from PIL import Image
import random
import cv2
from scipy import ndimage


def apply_SAM_multi(image: Image.Image, bounding_boxes: list) -> list:
    model = SAM("models/sam2_b.pt")  # modèle SAM

    if not isinstance(image, Image.Image):
        raise ValueError("L'entrée n'est pas une image PIL valide")

    image_np = np.array(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("L'image doit avoir 3 canaux (RGB)")

    image_rgb = image_np
    img_height, img_width = image_rgb.shape[:2]

    mask_images = []

    for bbox in bounding_boxes:
        results = model(image_rgb, bboxes=[bbox])

        if not results or not results[0].masks:
            continue

        # Récupérer le masque
        mask = results[0].masks[0].data.cpu().numpy().squeeze().astype(bool)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue


        # Trouver le rectangle englobant du masque
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        # Créer une image RGBA où l'objet est gardé, le reste transparent
        obj_rgba = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        obj_rgba[..., :3][mask] = image_rgb[mask]  # couleurs d'origine
        obj_rgba[..., 3][mask] = 255               # alpha à 255 pour l'objet

        # Crop autour de l'objet
        cropped_obj = obj_rgba[min_y:max_y+1, min_x:max_x+1, :]

        mask_images.append(Image.fromarray(cropped_obj, mode="RGBA"))

    # Si aucun masque valide, retourner l'image originale
    if not mask_images:
        mask_images = [image.convert("RGBA")]

    return mask_images