from ultralytics import YOLO
import os

# -------------------------------
# 1. Load Pretrained Model
# -------------------------------
model = YOLO("yolov8m.pt") # You can change to "yolov8n.pt" or any other variant

# -------------------------------
# 2. Train the Model
# -------------------------------
results = model.train(
 data="Weld-Classifier-1/data.yaml", # Relative path from current directory
 epochs=100,
 imgsz=640,
 batch=16, # Adjust as per your GPU capacity
 name="weld_defect_v1" # Custom run name
)

# -------------------------------
# 3. Locate the Best Model
# -------------------------------
best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
print(f"✅ Training complete. Best model saved at: {best_model_path}")

# -------------------------------
# 4. Validate the Model
# -------------------------------
print("🔍 Validating the trained model...")
model = YOLO(best_model_path)
metrics = model.val()
print(f"📊 Validation complete.")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

]

