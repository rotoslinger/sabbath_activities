from ultralytics import YOLO
import cv2
import numpy as np


def extract_person_mask(input_path, output_path):
    model = YOLO("yolov8n-seg.pt")  # Segmentation model
    results = model(input_path)

    image = cv2.imread(input_path)
    result = results[0]

    if not result.masks:
        raise ValueError("No segmentation masks found.")

    for i, cls in enumerate(result.boxes.cls):
        if int(cls) == 0:  # COCO class 0 = person
            mask = result.masks.data[i].cpu().numpy()
            break
    else:
        raise ValueError("No person class mask found.")

    # Resize mask to image resolution
    mask = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(
        mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # --- SMOOTHING ---
    # Morphological close to fill small holes
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)

    # Gaussian blur to soften edge transitions
    mask_blurred = cv2.GaussianBlur(mask_closed, (5, 5), sigmaX=2)

    # Threshold to restore sharp binary mask
    _, smooth_mask = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)

    # Create final RGBA output
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = smooth_mask

    cv2.imwrite(output_path, rgba)
    print(f"✅ Saved curved-mask result to: {output_path}")


from rembg import remove
import cv2


def extract_person_mask(input_path, output_path):
    # Load original image (BGR)
    image = cv2.imread(input_path)

    # Use rembg (U^2-Net) to segment with clean alpha edges
    rgba = remove(image)  # Returns RGBA with natural, curved alpha

    # Save result
    cv2.imwrite(output_path, rgba)
    print(f"✅ Saved Trellis-style mask: {output_path}")


# Example usage
input_path = "/Users/boshi/Documents/github/sabbath_activities/Jesus/body_seg.png"
output_path = "/Users/boshi/Documents/github/sabbath_activities/Jesus/body_cutout.png"
extract_person_mask(input_path, output_path)
