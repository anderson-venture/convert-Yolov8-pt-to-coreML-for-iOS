#All required imports
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import math
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cv2
    import os
    import zipfile
    import glob
    from ultralytics import YOLO
    import ipywidgets as widgets
    from IPython.display import display
    from PIL import Image

#Instal ultralytics
    !pip install ultralytics
    
#Indicating paths to model and test images
    model_path = "/content/1600 Model_without_mosaic_rotation2_perspective0.0025.pt"
    zip_path = "/content/New images 1920x1440.zip"

    extract_dir = "/content/test_images"
    output_dir = "/content/predicted_images"
    
# Unzip test images
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"✅ Extracted images to: {extract_dir}")
    
# Load your YOLOv8 model
    model = YOLO(model_path)

    print(f"✅ Model loaded from: {model_path}")

#Defining classes and colours for classes
    # ✅ Define your 9 class names exactly as per your YAML
    class_names = [
        "table",
        "data_cell",
        "header_cell",
        "description_cell",
        "table_title",
        "rowsubtotals_cell",
        "rowtotal_cell",
        "columnsubtotals_cell",
        "columntotals_cell"
    ]

    # ✅ Define colors explicitly (fixed RGB tuples)
    colors = [
        (31, 119, 180),   # table
        (255, 127, 14),   # data_cell
        (44, 160, 44),    # header_cell
        (214, 39, 40),    # description_cell
        (148, 103, 189),  # table_title
        (140, 86, 75),    # rowsubtotals_cell
        (227, 119, 194),  # rowtotal_cell
        (127, 127, 127),  # columnsubtotals_cell
        (188, 189, 34)    # columntotals_cell
    ]

    # ✅ Print your color mapping for verification
    print("========== COLOR KEY ==========")
    for idx, (name, color) in enumerate(zip(class_names, colors)):
        print(f"Class {idx}: {name} -> RGB{color}")
    print("================================")
    
# ✅ Get list of all .jpeg images in the extracted folder
    image_paths = glob.glob(os.path.join(extract_dir, "**/*.jpeg"), recursive=True)

    print(f"✅ Found {len(image_paths)} .jpeg images to process.")
    
# ✅ Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"✅ Output directory ready at: {output_dir}")
    
#Resizes downwards but not upwards (Yolov8s model trained at 1600x1600)
    def resize_bicubic_max_side(img, target_size=1600):
        h, w = img.shape[:2]
        max_side = max(h, w)

        # Only resize if the largest dimension exceeds target_size
        if max_side > target_size:
            scale = target_size / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return resized_img, scale
        else:
            return img, 1.0  # No resizing needed, return original image and scale=1.0
    
#Script for running inference
    from shapely.geometry import box as shapely_box

    def resize_bicubic_max_side(img, target_size=1280):
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return resized, scale

    # 🧠 IoU calculator
    def iou(boxA, boxB):
        a = shapely_box(*boxA[:4])
        b = shapely_box(*boxB[:4])
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0

    # 🧠 Merge boxes with IoU threshold
    def deduplicate_boxes(boxes, iou_thresh=0.6):
        final_boxes = []
        for box in boxes:
            keep = True
            for kept in final_boxes:
                if iou(box, kept) > iou_thresh:
                    keep = False
                    break
            if keep:
                final_boxes.append(box)
        return final_boxes

    # 🔁 Loop over images
    for img_path in image_paths:
        # ✅ Load original image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not load {img_path}")
            continue

        # ✅ Step 5: YOLO inference (run only once)
        raw_boxes = []  # Format: (x1, y1, x2, y2, cls)
        results = model(img.copy(), verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            raw_boxes.append((x1, y1, x2, y2, cls))

        # ✅ Deduplicate overlapping boxes (IoU > 60%)
        all_boxes = deduplicate_boxes(raw_boxes, iou_thresh=0.6)

        # ✅ Step 6: Draw boxes on processed image
        for x1, y1, x2, y2, cls in all_boxes:
            color = colors[cls % len(colors)]  # Ensure color indexing is safe
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # ✅ Step 7: Save output image
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)

    print(f"✅ All enhanced + inferred images saved to: {output_dir}")
    
#Zipping predicted images
    !zip -r predicted_images.zip predicted_images

    print("✅ Zipped predicted_images.zip is ready for download!")
    
#Define predicted images path:
    predicted_image_paths = sorted(glob.glob(os.path.join(output_dir, "*.jpeg")))
    print(f"✅ Found {len(predicted_image_paths)} images in {output_dir}")
    
#Show images in colab:
    def show_image(index):
        img_path = predicted_image_paths[index]
        img = Image.open(img_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Image {index+1}/{len(predicted_image_paths)}\n{os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

    slider = widgets.IntSlider(value=0, min=0, max=len(predicted_image_paths)-1, step=1)
    widgets.interact(show_image, index=slider)
    
