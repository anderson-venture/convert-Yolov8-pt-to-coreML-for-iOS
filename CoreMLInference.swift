import Foundation
import CoreML
import AppKit

class CoreMLInference {
    
    static func runInference() {
        print("🚀 Starting CoreML YOLOv8 Table Detection")
        
        // === Load the CoreML Model ===
        // ✅ Try relative paths that work across different machines
        let possibleModelPaths = [
            "./1600Model_without_mosaic_rotation2_perspective0.0025.mlpackage",
            "./1600Model_without_mosaic_rotation2_perspective0.0025.mlmodel",
            "./1600Model_without_mosaic_rotation2_perspective0.0025.mlmodelc",
            "1600Model_without_mosaic_rotation2_perspective0.0025.mlpackage",
            "1600Model_without_mosaic_rotation2_perspective0.0025.mlmodel"
        ]
        
        var model: MLModel?
        var usedPath: String = ""
        
        for path in possibleModelPaths {
            let modelPath = URL(fileURLWithPath: path)
            if let loadedModel = try? MLModel(contentsOf: modelPath) {
                model = loadedModel
                usedPath = path
                break
            }
        }
        
        guard let finalModel = model else {
            print("❌ Failed to load CoreML model from any of these paths:")
            for path in possibleModelPaths {
                print("  - \(path)")
            }
            fatalError("Please check model path and ensure it exists")
        }
        print("✅ Model loaded successfully from: \(usedPath)")
        
        // Print model information for debugging
        let modelDescription = finalModel.modelDescription
        print("📋 Model inputs: \(modelDescription.inputDescriptionsByName.keys)")
        print("📋 Model outputs: \(modelDescription.outputDescriptionsByName.keys)")
        
        // === Load images from folder ===
        // ✅ Use relative path for better portability
        let imagesFolder = URL(fileURLWithPath: "./test_images/New images 1920x1440 og")
        let fileManager = FileManager.default
        
        guard let imageFiles = try? fileManager.contentsOfDirectory(at: imagesFolder, includingPropertiesForKeys: nil) else {
            fatalError("❌ Cannot access images folder: \(imagesFolder.path)")
        }
        
        let filteredImages = imageFiles.filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
        
        if filteredImages.isEmpty {
            print("⚠️ No JPG or PNG images found in folder.")
            return
        } else {
            print("📸 Found \(filteredImages.count) images.")
        }
        
        // === Define input image size used by the model ===
        let inputSize = CGSize(width: 1600, height: 1600)
        
        // === Perform inference on each image ===
        for (index, imageURL) in filteredImages.enumerated() {
            print("\n🔍 Processing (\(index + 1)/\(filteredImages.count)): \(imageURL.lastPathComponent)")
            
            guard let nsImage = NSImage(contentsOf: imageURL) else {
                print("❌ Could not load image: \(imageURL.lastPathComponent)")
                continue
            }
            
            // Resize + letterbox (downscale only)
            let (pixelBuffer, scale, xPad, yPad, resizedImage) = nsImage.resizedDownwardLetterboxedWithMetadata(to: Int(inputSize.width))
            
            guard let buffer = pixelBuffer else {
                print("❌ Couldn't convert to pixel buffer")
                continue
            }
            
            // Predict - try different input methods
            var prediction: MLFeatureProvider?
            
            // Method 1: Try using YOLOInput wrapper
            let input = YOLOInput(image: buffer)
            prediction = try? finalModel.prediction(from: input)
            
            // Method 2: If that fails, try direct input
            if prediction == nil {
                print("⚠️ YOLOInput failed, trying direct input...")
                let inputName = modelDescription.inputDescriptionsByName.keys.first ?? "image"
                print("🔍 Using input name: \(inputName)")
                
                if let directInput = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)]) {
                    prediction = try? finalModel.prediction(from: directInput)
                }
            }
            
            guard let finalPrediction = prediction else {
                print("❌ All inference methods failed for \(imageURL.lastPathComponent)")
                continue
            }
            
            print("📦 Available outputs: \(finalPrediction.featureNames)")
            
            // Try different possible output names (including more common ones)
            var rawOutput: MLMultiArray?
            let possibleOutputNames = ["output", "output0", "Identity", "Identity:0", "var_914", "1", "predictions", "detections"]
            
            for outputName in possibleOutputNames {
                if let output = finalPrediction.featureValue(for: outputName)?.multiArrayValue {
                    rawOutput = output
                    print("✅ Found output with name: '\(outputName)'")
                    break
                }
            }
            
            guard let output = rawOutput else {
                print("❌ Could not find valid output. Available outputs: \(finalPrediction.featureNames)")
                // List all outputs for debugging
                for featureName in finalPrediction.featureNames {
                    if let feature = finalPrediction.featureValue(for: featureName) {
                        print("  - \(featureName): \(feature.type)")
                        if let multiArray = feature.multiArrayValue {
                            print("    Shape: \(multiArray.shape)")
                        }
                    }
                }
                continue
            }
            
            // Parse detections
            let detections = YOLOv8OutputParser.parse(
                rawOutput: output,
                confidenceThreshold: 0.25, // Lower threshold to catch more detections
                inputImageSize: inputSize,
                scale: scale,
                xOffset: xPad,
                yOffset: yPad
            )
            
            print("🎯 Final detections: \(detections.count)")
            
            // Print detection details
            for (i, detection) in detections.enumerated() {
                print("  Detection \(i + 1): \(detection.className) (\(String(format: "%.1f", detection.score * 100))%) at \(detection.rect)")
            }
            
            // Draw and save annotated image
            let annotatedImage = resizedImage.drawDetections(detections)
            let outURL = imagesFolder.appendingPathComponent("pred_\(imageURL.lastPathComponent)")
            annotatedImage.saveJPG(to: outURL)
            print("✅ Saved: \(outURL.lastPathComponent)")
        }
        
        print("\n🎉 All done!")
    }
}

// Entry point moved to main.swift for Xcode compatibility