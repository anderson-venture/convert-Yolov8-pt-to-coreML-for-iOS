#Exporting .pt to CoreML packages
import os
from ultralytics import YOLO

# ✅ Full path to your .pt model
model_path = "content/1600Model_without_mosaic_rotation2_perspective0.0025.pt"

# Check if model file exists
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    exit(1)

print(f"📱 Starting CoreML conversion for iOS...")
print(f"🔄 Loading model from: {model_path}")

try:
    # Load the model
    model = YOLO(model_path)
    print(f"✅ Model loaded successfully")
    
    # Export to CoreML with optimized settings for iOS
    print(f"🔄 Converting to CoreML format...")
    
    # Choose NMS setting (True = built-in NMS, False = manual NMS in iOS)
    include_nms = False  # Change to False if you want manual control in iOS
    
    if include_nms:
        print(f"📦 Including NMS in model (recommended for easier iOS integration)")
    else:
        print(f"⚙️  NMS disabled - you'll need to implement it in iOS code")
    
    success = model.export(
        format="coreml",
        dynamic=False,      # Fixed input size for better iOS performance
        simplify=True,      # Simplify the model graph
        nms=include_nms,    # Include/exclude NMS based on preference
        imgsz=1600,        # Match your training resolution
        half=False,        # Use FP32 for better compatibility
        int8=False         # Disable quantization for now
    )
    
    if success:
        # Find the exported model
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        expected_path = f"{model_name}.mlmodel"
        
        print(f"✅ CoreML export successful!")
        print(f"📁 Model exported as: {expected_path}")
        print(f"📱 Ready for iOS integration!")
        
        # Print model info
        print(f"\n📊 Model Details:")
        print(f"   • Input size: 1600x1600")
        print(f"   • Format: CoreML (.mlmodel)")
        print(f"   • Classes: 9 (table detection)")
        print(f"   • NMS: {'Included' if include_nms else 'Disabled (manual implementation needed)'}")
        print(f"   • Precision: FP32")
        
        if not include_nms:
            print(f"\n⚠️  NMS Implementation Notes:")
            print(f"   • You'll need to implement NMS in your iOS app")
            print(f"   • Raw model outputs will contain overlapping boxes")
            print(f"   • Consider IoU threshold ~0.5-0.7 for table cells")
            print(f"   • Confidence threshold ~0.25-0.5 depending on use case")
        
    else:
        print(f"❌ Export failed!")
        
except Exception as e:
    print(f"❌ Error during conversion: {str(e)}")
    print(f"💡 Make sure you have the required dependencies installed:")
    print(f"   pip install ultralytics coremltools")
    exit(1)
