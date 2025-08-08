#!/usr/bin/env swift

import Foundation
import CoreML
import AppKit
import CoreVideo

// MARK: - Detection Structure
struct Detection {
    let rect: CGRect
    let score: Float
    let classIndex: Int
    let className: String
}

// MARK: - YOLOInput Feature Provider
class YOLOInput: MLFeatureProvider {
    var image: CVPixelBuffer
    
    init(image: CVPixelBuffer) {
        self.image = image
    }
    
    var featureNames: Set<String> {
        // Common input names for YOLO models
        return ["image", "input", "x", "images"]
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        // Accept any of the common input names
        if ["image", "input", "x", "images"].contains(featureName) {
            return MLFeatureValue(pixelBuffer: image)
        }
        return nil
    }
}

// MARK: - YOLOv8 Output Parser
class YOLOv8OutputParser {
    
    static let classNames = [
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
    
    static func parse(
        rawOutput: MLMultiArray,
        confidenceThreshold: Float = 0.25,
        inputImageSize: CGSize,
        scale: CGFloat,
        xOffset: CGFloat,
        yOffset: CGFloat
    ) -> [Detection] {
        
        print("🔍 Raw output shape: \(rawOutput.shape)")
        print("🔍 Raw output strides: \(rawOutput.strides)")
        
        // YOLOv8 output format: [1, 84, 8400] or [1, 8400, 84]
        // Where 84 = 4 (bbox) + 80 (COCO classes) but your model has 9 classes
        // So it should be [1, 13, 8400] = 4 (bbox) + 9 (classes)
        
        guard rawOutput.shape.count >= 2 else {
            print("❌ Unexpected output shape")
            return []
        }
        
        let numClasses = 9
        let expectedFeatures = 4 + numClasses // 13 total
        
        var detections: [Detection] = []
        
        // Handle different possible output formats
        let shape = rawOutput.shape.map { $0.intValue }
        print("🔍 Output dimensions: \(shape)")
        
        var numBoxes: Int
        var featuresPerBox: Int
        
        if shape.count == 3 {
            // Format: [batch, features, boxes] or [batch, boxes, features]
            if shape[1] == expectedFeatures {
                // [1, 13, 8400] format
                featuresPerBox = shape[1]
                numBoxes = shape[2]
            } else if shape[2] == expectedFeatures {
                // [1, 8400, 13] format
                numBoxes = shape[1]
                featuresPerBox = shape[2]
            } else {
                print("❌ Unexpected feature count. Expected \(expectedFeatures), got \(shape)")
                return []
            }
        } else {
            print("❌ Unsupported output format")
            return []
        }
        
        print("📊 Processing \(numBoxes) boxes with \(featuresPerBox) features each")
        
        // Parse detections
        for boxIndex in 0..<numBoxes {
            var maxConfidence: Float = 0
            var bestClassIndex: Int = 0
            
            // Get class confidences (skip first 4 bbox coordinates)
            for classIndex in 0..<numClasses {
                let featureIndex = 4 + classIndex
                let confidence = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: featureIndex, shape: shape)
                
                if confidence > maxConfidence {
                    maxConfidence = confidence
                    bestClassIndex = classIndex
                }
            }
            
            // Filter by confidence threshold
            guard maxConfidence >= confidenceThreshold else { continue }
            
            // Get bounding box coordinates
            let centerX = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 0, shape: shape)
            let centerY = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 1, shape: shape)
            let width = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 2, shape: shape)
            let height = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 3, shape: shape)
            
            // YOLOv8 outputs can be in different formats:
            // 1. Normalized (0-1) coordinates
            // 2. Pixel coordinates relative to input size
            
            // Check if coordinates are normalized (values between 0-1)
            let isNormalized = centerX <= 1.0 && centerY <= 1.0 && width <= 1.0 && height <= 1.0
            
            // Debug first few detections
            if boxIndex < 3 && maxConfidence > confidenceThreshold {
                print("🔍 Detection \(boxIndex): cx=\(centerX), cy=\(centerY), w=\(width), h=\(height), conf=\(maxConfidence), normalized=\(isNormalized)")
            }
            
            var scaledCenterX = centerX
            var scaledCenterY = centerY
            var scaledWidth = width
            var scaledHeight = height
            
            if isNormalized {
                // Convert from normalized to pixel coordinates
                scaledCenterX = centerX * Float(inputImageSize.width)
                scaledCenterY = centerY * Float(inputImageSize.height)
                scaledWidth = width * Float(inputImageSize.width)
                scaledHeight = height * Float(inputImageSize.height)
            }
            
            // Convert from center format to corner format
            let x1 = scaledCenterX - scaledWidth / 2
            let y1 = scaledCenterY - scaledHeight / 2
            let x2 = scaledCenterX + scaledWidth / 2
            let y2 = scaledCenterY + scaledHeight / 2
            
            // Convert coordinates back to original image space
            let originalX1 = max(0, (CGFloat(x1) - xOffset) / scale)
            let originalY1 = max(0, (CGFloat(y1) - yOffset) / scale)
            let originalX2 = (CGFloat(x2) - xOffset) / scale
            let originalY2 = (CGFloat(y2) - yOffset) / scale
            
            let rect = CGRect(
                x: originalX1,
                y: originalY1,
                width: originalX2 - originalX1,
                height: originalY2 - originalY1
            )
            
            let className = bestClassIndex < classNames.count ? classNames[bestClassIndex] : "unknown"
            
            let detection = Detection(
                rect: rect,
                score: maxConfidence,
                classIndex: bestClassIndex,
                className: className
            )
            
            detections.append(detection)
        }
        
        print("🎯 Found \(detections.count) raw detections")
        
        // Apply NMS to remove overlapping detections (matching Python script's 0.6 threshold)
        let nmsDetections = performNMS(detections: detections, iouThreshold: 0.6)
        print("🎯 After NMS: \(nmsDetections.count) final detections")
        
        return nmsDetections
    }
    
    private static func getOutputValue(_ output: MLMultiArray, boxIndex: Int, featureIndex: Int, shape: [Int]) -> Float {
        var index: Int
        
        if shape[1] == 13 {
            // [1, 13, 8400] format
            index = featureIndex * shape[2] + boxIndex
        } else {
            // [1, 8400, 13] format  
            index = boxIndex * shape[2] + featureIndex
        }
        
        return output[index].floatValue
    }
    
    private static func performNMS(detections: [Detection], iouThreshold: Float) -> [Detection] {
        // Sort by confidence (highest first)
        let sortedDetections = detections.sorted { $0.score > $1.score }
        var finalDetections: [Detection] = []
        
        for detection in sortedDetections {
            var shouldKeep = true
            
            for existingDetection in finalDetections {
                let iou = calculateIoU(detection.rect, existingDetection.rect)
                if iou > iouThreshold {
                    shouldKeep = false
                    break
                }
            }
            
            if shouldKeep {
                finalDetections.append(detection)
            }
        }
        
        return finalDetections
    }
    
    private static func calculateIoU(_ rect1: CGRect, _ rect2: CGRect) -> Float {
        let intersection = rect1.intersection(rect2)
        
        if intersection.isNull {
            return 0.0
        }
        
        let intersectionArea = intersection.width * intersection.height
        let unionArea = rect1.width * rect1.height + rect2.width * rect2.height - intersectionArea
        
        return Float(intersectionArea / unionArea)
    }
}

// MARK: - NSImage Extensions
extension NSImage {  
    func drawDetections(_ detections: [Detection]) -> NSImage {
        let size = self.size
        let newImage = NSImage(size: size)
        newImage.lockFocus()
        
        self.draw(at: .zero, from: CGRect(origin: .zero, size: size), operation: .sourceOver, fraction: 1.0)
        
        let context = NSGraphicsContext.current?.cgContext
        context?.setLineWidth(3.0)
        
        // Define colors for different classes
        let colors: [NSColor] = [
            .systemRed,      // table
            .systemOrange,   // data_cell
            .systemGreen,    // header_cell
            .systemBlue,     // description_cell
            .systemPurple,   // table_title
            .systemYellow,   // rowsubtotals_cell
            .systemPink,     // rowtotal_cell
            .systemTeal,     // columnsubtotals_cell
            .systemIndigo    // columntotals_cell
        ]
        
        for detection in detections {
            let rect = detection.rect
            let color = colors[detection.classIndex % colors.count]
            
            // Draw bounding box
            context?.setStrokeColor(color.withAlphaComponent(0.8).cgColor)
            context?.stroke(rect)
            
            // Draw filled background for label
            let labelBg = CGRect(x: rect.origin.x, y: rect.origin.y - 20, width: 120, height: 18)
            context?.setFillColor(color.withAlphaComponent(0.8).cgColor)
            context?.fill(labelBg)
            
            // Draw label text
            let label = String(format: "%@ %.0f%%", detection.className, detection.score * 100)
            let attributes: [NSAttributedString.Key: Any] = [
                .foregroundColor: NSColor.white,
                .font: NSFont.systemFont(ofSize: 12, weight: .bold)
            ]
            let labelText = NSString(string: label)
            labelText.draw(at: CGPoint(x: rect.origin.x + 2, y: rect.origin.y - 18), withAttributes: attributes)
        }
        
        newImage.unlockFocus()
        return newImage
    }
    
    func saveJPG(to url: URL, quality: CGFloat = 0.9) {
        guard let tiffData = self.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let jpgData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: quality]) else {
            print("❌ Failed to convert image to JPEG")
            return
        }
        
        do {
            try jpgData.write(to: url)
            print("✅ Saved JPEG to: \(url.path)")
        } catch {
            print("❌ Failed to write image to disk: \(error)")
        }
    }
    
    /// Convert NSImage directly to CVPixelBuffer without any preprocessing
    /// This matches the Python script behavior of feeding original image to model
    func toPixelBuffer() -> CVPixelBuffer? {
        return toPixelBuffer(targetSize: nil)
    }
    
    /// Convert NSImage to CVPixelBuffer with optional YOLO-style letterbox preprocessing
    /// - Parameter targetSize: If provided, applies YOLO letterbox preprocessing to this size
    func toPixelBuffer(targetSize: Int?) -> CVPixelBuffer? {
        guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("❌ Failed to get CGImage from NSImage")
            return nil
        }
        
        if let target = targetSize {
            // Apply YOLO letterbox preprocessing
            return letterboxPreprocess(cgImage: cgImage, targetSize: target)
        } else {
            // Use original size (no preprocessing)
            return createPixelBuffer(from: cgImage, width: cgImage.width, height: cgImage.height)
        }
    }
    
    /// YOLO-style letterbox preprocessing (matches Ultralytics implementation)
    /// Maintains aspect ratio, resizes to fit target size, adds padding
    private func letterboxPreprocess(cgImage: CGImage, targetSize: Int) -> CVPixelBuffer? {
        let originalWidth = CGFloat(cgImage.width)
        let originalHeight = CGFloat(cgImage.height)
        let targetFloat = CGFloat(targetSize)
        
        // Calculate scale factor (same as YOLO: min(target/w, target/h))
        let scale = min(targetFloat / originalWidth, targetFloat / originalHeight)
        
        // Calculate new dimensions (maintaining aspect ratio)
        let newWidth = Int(originalWidth * scale)
        let newHeight = Int(originalHeight * scale)
        
        // Calculate padding (to center the image)
        let padX = (targetSize - newWidth) / 2
        let padY = (targetSize - newHeight) / 2
        
        print("🔍 YOLO Letterbox: \(Int(originalWidth))x\(Int(originalHeight)) → \(newWidth)x\(newHeight) + pad(\(padX),\(padY))")
        
        // Create target size pixel buffer
        guard let buffer = createPixelBuffer(width: targetSize, height: targetSize) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                 width: targetSize,
                                 height: targetSize,
                                 bitsPerComponent: 8,
                                 bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                 space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            print("❌ Failed to create graphics context")
            return nil
        }
        
        // Fill with gray background (YOLO uses 114/255 = 0.447 gray)
        ctx.setFillColor(CGColor(red: 0.447, green: 0.447, blue: 0.447, alpha: 1.0))
        ctx.fill(CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
        
        // Draw resized image centered with padding
        let drawRect = CGRect(x: padX, y: padY, width: newWidth, height: newHeight)
        ctx.draw(cgImage, in: drawRect)
        
        return buffer
    }
    
    /// Helper to create pixel buffer from CGImage
    private func createPixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        return createPixelBuffer(width: width, height: height, drawImage: cgImage)
    }
    
    /// Helper to create empty pixel buffer
    private func createPixelBuffer(width: Int, height: Int, drawImage: CGImage? = nil) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                        kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            print("❌ Failed to create pixel buffer: \(status)")
            return nil
        }
        
        if let image = drawImage {
            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            
            guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                     space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                print("❌ Failed to create graphics context")
                return nil
            }
            
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        
        return buffer
    }
}

// MARK: - Main CoreML Inference Function
func runInference() {
    print("🚀 Starting CoreML YOLOv8 Table Detection")
    
    // === Load the CoreML Model ===
    // ✅ Try relative paths that work across different machines
    let possibleModelPaths = [
        "./1600Model_without_mosaic_rotation2_perspective0.0025.mlpackage",
        "1600Model_without_mosaic_rotation2_perspective0.0025.mlpackage",
        "./1600Model_without_mosaic_rotation2_perspective0.0025.mlmodel",
        "./1600Model_without_mosaic_rotation2_perspective0.0025.mlmodelc",
        "1600Model_without_mosaic_rotation2_perspective0.0025.mlmodel",
        "1600Model_without_mosaic_rotation2_perspective0.0025.mlmodelc"
    ]
    
    var model: MLModel?
    var usedPath: String = ""
    
    // First, let's check what files actually exist in the current directory
    print("🔍 Checking current directory contents...")
    let fileManager = FileManager.default
    let currentDir = fileManager.currentDirectoryPath
    print("📁 Current working directory: \(currentDir)")
    
    if let contents = try? fileManager.contentsOfDirectory(atPath: currentDir) {
        let modelFiles = contents.filter { $0.contains("1600Model") }
        print("📋 Found model-related files: \(modelFiles)")
    }
    
    for path in possibleModelPaths {
        let modelPath = URL(fileURLWithPath: path)
        print("🔍 Trying path: \(path)")
        print("   Full URL: \(modelPath.path)")
        print("   Exists: \(fileManager.fileExists(atPath: modelPath.path))")
        
        // Check if it's a directory (for .mlpackage)
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: modelPath.path, isDirectory: &isDirectory)
        print("   Is directory: \(isDirectory.boolValue)")
        
        if exists {
            do {
                // For .mlpackage, we need to load it as a bundle/directory
                if path.hasSuffix(".mlpackage") {
                    print("   📦 Loading as mlpackage bundle...")
                    // Try using MLModel.compileModel first for .mlpackage
                    let compiledURL = try MLModel.compileModel(at: modelPath)
                    let loadedModel = try MLModel(contentsOf: compiledURL)
                    model = loadedModel
                    usedPath = path
                    print("✅ Successfully compiled and loaded mlpackage from: \(path)")
                    break
                } else {
                    // For .mlmodel and .mlmodelc, load directly
                    let loadedModel = try MLModel(contentsOf: modelPath)
                    model = loadedModel
                    usedPath = path
                    print("✅ Successfully loaded model from: \(path)")
                    break
                }
            } catch {
                print("   ❌ Failed to load: \(error.localizedDescription)")
                
                // If compilation failed for .mlpackage, try loading directly
                if path.hasSuffix(".mlpackage") {
                    print("   🔄 Compilation failed, trying direct load...")
                    do {
                        let loadedModel = try MLModel(contentsOf: modelPath)
                        model = loadedModel
                        usedPath = path
                        print("✅ Successfully loaded mlpackage directly from: \(path)")
                        break
                    } catch {
                        print("   ❌ Direct load also failed: \(error.localizedDescription)")
                    }
                }
            }
        }
    }
    
    guard let finalModel = model else {
        print("❌ Failed to load CoreML model from any of these paths:")
        for path in possibleModelPaths {
            print("  - \(path)")
        }
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you're running the script from the correct directory")
        print("2. Check that the model file exists and has the correct name")
        print("3. For .mlpackage models, the entire folder should exist")
        print("4. Try using the absolute path to the model if relative paths don't work")
        print("\n📁 Current directory contents shown above - verify your model file is listed")
        return  // Changed from fatalError to graceful return
    }
    print("✅ Model loaded successfully from: \(usedPath)")
    
    // Print detailed model information for debugging
    let modelDescription = finalModel.modelDescription
    print("📋 Model inputs: \(modelDescription.inputDescriptionsByName.keys)")
    print("📋 Model outputs: \(modelDescription.outputDescriptionsByName.keys)")
    
    // Check input constraints to see if model has fixed or dynamic input size
    for (inputName, inputDesc) in modelDescription.inputDescriptionsByName {
        print("\n🔍 Input '\(inputName)' details:")
        print("   Type: \(inputDesc.type)")
        
        if case .image = inputDesc.type {
            if let imageConstraint = inputDesc.imageConstraint {
                print("   Pixel format: \(imageConstraint.pixelFormatType)")
                print("   Size constraint: \(imageConstraint.sizeConstraint)")
                
                switch imageConstraint.sizeConstraint.type {
                case .enumerated:
                    print("   ✅ FIXED SIZE MODEL - Accepts specific sizes only:")
                    if let enumeratedSizes = imageConstraint.sizeConstraint.enumeratedImageSizes {
                        for size in enumeratedSizes {
                            print("     - \(size.pixelsWide) x \(size.pixelsHigh)")
                        }
                    }
                case .range:
                    print("   ✅ DYNAMIC SIZE MODEL - Accepts range of sizes:")
                    let range = imageConstraint.sizeConstraint.pixelsWideRange
                    let heightRange = imageConstraint.sizeConstraint.pixelsHighRange
                    print("     Width: \(range.lowerBound) - \(range.upperBound)")
                    print("     Height: \(heightRange.lowerBound) - \(heightRange.upperBound)")
                case .unspecified:
                    print("   ✅ FLEXIBLE SIZE MODEL - No specific size constraints")
                @unknown default:
                    print("   ❓ Unknown size constraint type")
                }
            }
        } else if case .multiArray = inputDesc.type {
            if let multiArrayConstraint = inputDesc.multiArrayConstraint {
                print("   Shape: \(multiArrayConstraint.shape)")
                print("   Data type: \(multiArrayConstraint.dataType)")
            }
        }
    }
    
    // === Load images from folder ===
    // ✅ Use relative path for better portability
    let imagesFolder = URL(fileURLWithPath: "./test_images/New images 1920x1440 og")
    
    guard let imageFiles = try? fileManager.contentsOfDirectory(at: imagesFolder, includingPropertiesForKeys: nil, options: []) else {
        fatalError("❌ Cannot access images folder: \(imagesFolder.path)")
    }
    
    let filteredImages = imageFiles.filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
    
    if filteredImages.isEmpty {
        print("⚠️ No JPG or PNG images found in folder.")
        return
    } else {
        print("📸 Found \(filteredImages.count) images.")
    }
    
    // === Perform inference on each image ===
    for (index, imageURL) in filteredImages.enumerated() {
        print("\n🔍 Processing (\(index + 1)/\(filteredImages.count)): \(imageURL.lastPathComponent)")
        
        guard let nsImage = NSImage(contentsOf: imageURL) else {
            print("❌ Could not load image: \(imageURL.lastPathComponent)")
            continue
        }
        
        // ✅ Convert original image directly to pixel buffer (like Python script)
        // Let CoreML model handle all preprocessing internally (matching Ultralytics behavior)
        guard let originalPixelBuffer = nsImage.toPixelBuffer() else {
            print("❌ Couldn't convert original image to pixel buffer")
            continue
        }
        
        // Calculate letterbox parameters for coordinate conversion
        let originalSize = nsImage.size
        let modelInputSize: CGFloat = 1600  // Your model's expected input size
        
        // Calculate YOLO letterbox parameters
        let yoloScale = min(modelInputSize / originalSize.width, modelInputSize / originalSize.height)
        let newWidth = originalSize.width * yoloScale
        let newHeight = originalSize.height * yoloScale
        let xPad = (modelInputSize - newWidth) / 2.0
        let yPad = (modelInputSize - newHeight) / 2.0
        
        print("🔍 Letterbox params: scale=\(yoloScale), pad=(\(xPad),\(yPad))")
        
        // Predict - try different input methods with detailed error reporting
        var prediction: MLFeatureProvider?
        
        // First, let's inspect the image dimensions
        print("🔍 Image info: \(nsImage.size.width) x \(nsImage.size.height)")
        print("🔍 Pixel buffer: \(CVPixelBufferGetWidth(originalPixelBuffer)) x \(CVPixelBufferGetHeight(originalPixelBuffer))")
        
        // Method 1: Try using YOLOInput wrapper
        let input = YOLOInput(image: originalPixelBuffer)
        do {
            prediction = try finalModel.prediction(from: input)
            print("✅ YOLOInput method succeeded")
        } catch {
            print("❌ YOLOInput failed with error: \(error.localizedDescription)")
        }
        
        // Method 2: If that fails, try direct input
        if prediction == nil {
            print("⚠️ YOLOInput failed, trying direct input...")
            let inputName = modelDescription.inputDescriptionsByName.keys.first ?? "image"
            print("🔍 Using input name: \(inputName)")
            
            // Let's also check the expected input format
            if let inputDescription = modelDescription.inputDescriptionsByName[inputName] {
                print("🔍 Expected input type: \(inputDescription.type)")
                if case .image = inputDescription.type {
                    if let imageConstraint = inputDescription.imageConstraint {
                        print("🔍 Expected image format: \(imageConstraint.pixelFormatType)")
                        print("🔍 Expected image size: \(imageConstraint.pixelsWide) x \(imageConstraint.pixelsHigh)")
                    }
                }
            }
            
            do {
                let directInput = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: originalPixelBuffer)])
                prediction = try finalModel.prediction(from: directInput)
                print("✅ Direct input method succeeded")
            } catch {
                print("❌ Direct input failed with error: \(error.localizedDescription)")
            }
        }
        
        // Method 3: If both fail, try resizing to model's expected input size
        if prediction == nil {
            print("⚠️ Both methods failed, trying with resized image to 1600x1600...")
            
            if let resizedPixelBuffer = nsImage.toPixelBuffer(targetSize: 1600) {
                print("🔍 Resized pixel buffer: \(CVPixelBufferGetWidth(resizedPixelBuffer)) x \(CVPixelBufferGetHeight(resizedPixelBuffer))")
                
                do {
                    let inputName = modelDescription.inputDescriptionsByName.keys.first ?? "image"
                    let resizedInput = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: resizedPixelBuffer)])
                    prediction = try finalModel.prediction(from: resizedInput)
                    print("✅ Resized input method succeeded")
                } catch {
                    print("❌ Resized input failed with error: \(error.localizedDescription)")
                }
            }
        }
        
        guard let finalPrediction = prediction else {
            print("❌ All inference methods failed for \(imageURL.lastPathComponent)")
            print("💡 This might be due to incompatible image format or size")
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
        
        // Parse detections (using default YOLOv8 confidence threshold)
        let detections = YOLOv8OutputParser.parse(
            rawOutput: output,
            confidenceThreshold: 0.25, // Standard YOLOv8 threshold (matches Ultralytics default)
            inputImageSize: CGSize(width: modelInputSize, height: modelInputSize), // Use model input size
            scale: yoloScale,
            xOffset: xPad,
            yOffset: yPad
        )
        
        print("🎯 Final detections: \(detections.count)")
        
        // Print detection details
        for (i, detection) in detections.enumerated() {
            print("  Detection \(i + 1): \(detection.className) (\(String(format: "%.1f", detection.score * 100))%) at \(detection.rect)")
        }
        
        // Draw and save annotated image (using original image)
        let annotatedImage = nsImage.drawDetections(detections)
        let outURL = imagesFolder.appendingPathComponent("pred_\(imageURL.lastPathComponent)")
        annotatedImage.saveJPG(to: outURL)
        print("✅ Saved: \(outURL.lastPathComponent)")
    }
    
    print("\n🎉 All done!")
}

// MARK: - Main Entry Point
print("🚀 Starting CoreML YOLOv8 Table Detection - Standalone Version")
runInference()