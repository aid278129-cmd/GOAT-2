import os
import shutil
import sys

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_yaml = os.path.join(base_dir, "dataset", "data.yaml")
    models_dir = os.path.join(base_dir, "models")
    public_models_dir = os.path.join(base_dir, "public", "models")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(public_models_dir, exist_ok=True)
    target_pt = os.path.join(models_dir, "indian_plate_best.pt")

    print("=== Training Indian License Plate Detector (YOLOv8 Nano) ===")
    from ultralytics import YOLO

    # Initialize with previously best weights or yolov8n
    initial_weights = target_pt if os.path.exists(target_pt) else "yolov8n.pt"
    print(f"Starting fine-tuning from: {initial_weights}")
    model = YOLO(initial_weights)

    # Fine-tune for 60 epochs with realistic license-plate augmentations
    results = model.train(
        data=data_yaml,
        epochs=60,
        imgsz=640,
        batch=4,
        workers=0,      # Windows multiprocessing safety
        name="indian_plate_fine_tune",
        project=os.path.join(base_dir, "runs"),
        exist_ok=True,
        verbose=True,
        lr0=0.003,      # Lower initial LR for smooth fine-tuning
        lrf=0.01,
        degrees=10.0,   # Allow +/- 10 deg rotation for camera tilt
        translate=0.1,  # Position jitter
        scale=0.3,      # Scale jitter for close-up and far-away plates
        fliplr=0.0,     # IMPORTANT: Do not flip horizontally (numbers would mirror)
        mosaic=0.5,     # Contextual learning
        patience=20,    # Early stopping if plateaued
    )

    best_pt = os.path.join(base_dir, "runs", "indian_plate_fine_tune", "weights", "best.pt")
    target_pt = os.path.join(models_dir, "indian_plate_best.pt")

    if os.path.exists(best_pt):
        shutil.copy2(best_pt, target_pt)
        print(f"✅ Best weights saved to: {target_pt}")
    else:
        print("Warning: best.pt not found, checking last.pt")
        last_pt = os.path.join(base_dir, "runs", "indian_plate_run", "weights", "last.pt")
        if os.path.exists(last_pt):
            shutil.copy2(last_pt, target_pt)

    # Export to ONNX format
    print("=== Exporting Trained Model to ONNX ===")
    trained_model = YOLO(target_pt)
    onnx_path = trained_model.export(format="onnx", imgsz=640, dynamic=False)

    public_onnx = os.path.join(public_models_dir, "indian_plate_detector.onnx")
    if os.path.exists(onnx_path):
        shutil.copy2(onnx_path, public_onnx)
        print(f"✅ Exported ONNX model to web static directory: {public_onnx}")

    print("\n🎉 Model Training and ONNX Export Complete!")

if __name__ == "__main__":
    main()
