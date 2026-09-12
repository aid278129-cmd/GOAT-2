import os
import glob
import xml.etree.ElementTree as ET
import shutil
import random
from PIL import Image

def convert_voc_to_yolo():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "dataset")

    # Destination directories
    train_img_dir = os.path.join(out_dir, "images", "train")
    val_img_dir = os.path.join(out_dir, "images", "val")
    train_lbl_dir = os.path.join(out_dir, "labels", "train")
    val_lbl_dir = os.path.join(out_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    sources = [
        {
            "img_dir": os.path.join(base_dir, "Indian_Number_Plates", "Sample_Images"),
            "xml_dir": os.path.join(base_dir, "Annotations", "Annotations"),
        },
        {
            "img_dir": os.path.join(base_dir, "number_plate_images_ocr", "number_plate_images_ocr"),
            "xml_dir": os.path.join(base_dir, "number_plate_annos_ocr", "number_plate_annos_ocr"),
        },
    ]

    all_samples = []

    for src in sources:
        img_dir = src["img_dir"]
        xml_dir = src["xml_dir"]
        if not os.path.exists(xml_dir):
            print(f"Warning: XML dir does not exist: {xml_dir}")
            continue

        xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
        for xml_file in xml_files:
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            # Try to find corresponding image (.jpg, .jpeg, .png)
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = os.path.join(img_dir, base_name + ext)
                if os.path.exists(candidate):
                    img_file = candidate
                    break

            if img_file:
                all_samples.append((xml_file, img_file))
            else:
                print(f"Image not found for: {xml_file}")

    print(f"Total valid annotated samples discovered: {len(all_samples)}")

    random.seed(42)
    random.shuffle(all_samples)

    # 85% train, 15% validation
    val_count = max(4, int(len(all_samples) * 0.15))
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    print(f"Train split: {len(train_samples)} samples, Val split: {len(val_samples)} samples")

    def process_split(samples, img_dest, lbl_dest):
        count = 0
        for xml_file, img_file in samples:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image dimensions from XML or PIL
                size_elem = root.find("size")
                if size_elem is not None and size_elem.find("width") is not None and int(float(size_elem.find("width").text)) > 0:
                    width = float(size_elem.find("width").text)
                    height = float(size_elem.find("height").text)
                else:
                    with Image.open(img_file) as im:
                        width, height = im.size

                yolo_boxes = []
                for obj in root.findall("object"):
                    name = obj.find("name").text.strip().lower()
                    if "plate" in name or name == "number_plate":
                        bnd = obj.find("bndbox")
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)

                        # Clamp coordinates to image boundaries
                        xmin = max(0.0, min(xmin, width))
                        ymin = max(0.0, min(ymin, height))
                        xmax = max(0.0, min(xmax, width))
                        ymax = max(0.0, min(ymax, height))

                        bw = xmax - xmin
                        bh = ymax - ymin
                        if bw <= 0 or bh <= 0:
                            continue

                        # Normalized center x, center y, width, height
                        x_center = (xmin + bw / 2.0) / width
                        y_center = (ymin + bh / 2.0) / height
                        w_norm = bw / width
                        h_norm = bh / height

                        # Class 0: number_plate
                        yolo_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                if not yolo_boxes:
                    print(f"Warning: No valid plate box found in {xml_file}")
                    continue

                safe_name = os.path.basename(img_file).replace(" ", "_").replace("(", "").replace(")", "")
                base_safe = os.path.splitext(safe_name)[0]

                target_img_path = os.path.join(img_dest, safe_name)
                target_lbl_path = os.path.join(lbl_dest, base_safe + ".txt")

                shutil.copy2(img_file, target_img_path)
                with open(target_lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_boxes) + "\n")

                count += 1
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
        return count

    n_train = process_split(train_samples, train_img_dir, train_lbl_dir)
    n_val = process_split(val_samples, val_img_dir, val_lbl_dir)

    # Write data.yaml for YOLO
    yaml_content = f"""path: {out_dir.replace('\\\\', '/')}
train: images/train
val: images/val

names:
  0: number_plate
"""
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Dataset successfully created at {out_dir}")
    print(f"Processed: {n_train} training samples, {n_val} validation samples.")
    print(f"Config written to {yaml_path}")

if __name__ == "__main__":
    convert_voc_to_yolo()
