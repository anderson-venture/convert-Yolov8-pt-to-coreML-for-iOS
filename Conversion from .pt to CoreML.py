#Conversion was done on Mac Terminal

    #instal ultralytics in local mac terminal environment
        pip install ultralytics


    #Exporting .pt to CoreML packages
        from ultralytics import YOLO

        # ✅ Full path to your .pt model
        model_path = "/Users/malvernnzwere/Library/CloudStorage/OneDrive-Personal/Yolo $

        # Load the model
        model = YOLO(model_path)

        # Export directly to CoreML
        model.export(
            format="coreml",
            dynamic=False,      # Avoid dynamic shape ops
            nms=False,          # Exclude NMS for better control in iOS
            imgsz=1600          # Match training resolution
        )

        print("✅ Export complete. Check the 'runs' folder for your .mlmodel file.")


    #Running the conversion:
        python3 ~/convert_pt_to_coreml.py
