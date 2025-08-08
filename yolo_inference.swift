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
        scale: CGFloat,
        originalImageSize: CGSize,
        targetSize: Int
    ) -> [Detection] {
        
        guard rawOutput.shape.count >= 2 else {
            print("❌ Invalid output shape: \(rawOutput.shape)")
            return []
        }
        
        let numClasses = 9
        let expectedFeatures = 4 + numClasses // 13 total
        var detections: [Detection] = []
        let shape = rawOutput.shape.map { $0.intValue }
        
        var numBoxes: Int
        var featuresPerBox: Int
        
        if shape.count == 3 {
            if shape[1] == expectedFeatures {
                // [1, 13, N] format
                featuresPerBox = shape[1]
                numBoxes = shape[2]
            } else if shape[2] == expectedFeatures {
                // [1, N, 13] format
                numBoxes = shape[1]
                featuresPerBox = shape[2]
            } else {
                print("❌ Unexpected feature count. Expected \(expectedFeatures), got shape: \(shape)")
                return []
            }
        } else {
            print("❌ Expected 3D tensor, got \(shape.count)D: \(shape)")
            return []
        }
        
        // Show first few values for debugging
        print("🔍 First box sample values:")
        for j in 0..<min(13, featuresPerBox) {
            let val = getOutputValue(rawOutput, boxIndex: 0, featureIndex: j, shape: shape)
            print("  [\(j)]: \(val)")
        }
        
        // Parse detections - YOLOv8 format
        for boxIndex in 0..<numBoxes {
            // Get bounding box coordinates (first 4 values)
            let centerX = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 0, shape: shape)
            let centerY = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 1, shape: shape)
            let width = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 2, shape: shape)
            let height = getOutputValue(rawOutput, boxIndex: boxIndex, featureIndex: 3, shape: shape)
            
            // Get class confidences (values 4-12 for 9 classes)
            var maxConfidence: Float = 0
            var bestClassIndex: Int = 0
            
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
            
            // Model outputs coordinates in the 1600x1600 input space
            // Convert from center format to corner format
            let x1 = centerX - width / 2
            let y1 = centerY - height / 2
            let x2 = centerX + width / 2
            let y2 = centerY + height / 2
            
            // Calculate the centering offsets used when creating the 1600x1600 pixel buffer
            let scaledImageWidth = originalImageSize.width * scale
            let scaledImageHeight = originalImageSize.height * scale
            let xOffset = (CGFloat(targetSize) - scaledImageWidth) / 2.0
            let yOffset = (CGFloat(targetSize) - scaledImageHeight) / 2.0
            
            // // Debug high-confidence detections only
            // if maxConfidence > 0.8 {
            //     print("🔍 High confidence detection: \(classNames[bestClassIndex]) (\(maxConfidence))")
            //     print("    Coordinates: (\(centerX), \(centerY), \(width), \(height))")
            // }
            
            // Convert coordinates back to original image space:
            // Model outputs coordinates in 1600x1600 space, need to:
            // 1. Subtract the centering offset (to get coordinates in scaled image space)
            // 2. Scale back to original image coordinates
            let originalX1 = max(0, (CGFloat(x1) - xOffset) / scale)
            let originalY1 = max(0, (CGFloat(y1) - yOffset) / scale)
            let originalX2 = (CGFloat(x2) - xOffset) / scale
            let originalY2 = (CGFloat(y2) - yOffset) / scale
            
            let rect = CGRect(
                x: originalX1,
                y: originalY1,
                width: max(0, originalX2 - originalX1),
                height: max(0, originalY2 - originalY1)
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
        
        // Apply NMS deduplication
        let nmsDetections = performNMS(detections: detections, iouThreshold: 0.6)
        
        return nmsDetections
    }
    
    private static func getOutputValue(_ output: MLMultiArray, boxIndex: Int, featureIndex: Int, shape: [Int]) -> Float {
        var index: Int
        
        if shape[1] == 13 {
            // [1, 13, 52500] format (from CoreML export)
            index = featureIndex * shape[2] + boxIndex
        } else {
            // [1, 52500, 13] format (alternative)
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
                // Only compare detections of the same class
                if detection.classIndex == existingDetection.classIndex {
                    let iou = calculateIoU(detection.rect, existingDetection.rect)
                    if iou > iouThreshold {
                        shouldKeep = false
                        break
                    }
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
    /// Resizes the image down (if needed) using bicubic interpolation, preserving aspect ratio.
    /// Returns: A tuple containing the pixel buffer, scale factor, and the final NSImage used for inference.
    func resizedBicubicMaxSide(to targetSize: Int = 1600) -> (CVPixelBuffer?, CGFloat, NSImage) {
        guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("❌ Failed to get CGImage from NSImage")
            return (nil, 1.0, self)
        }
        
        let originalWidth = CGFloat(cgImage.width)
        let originalHeight = CGFloat(cgImage.height)
        let maxSide = max(originalWidth, originalHeight)
        
        // Only resize if the largest dimension exceeds targetSize (like Python script)
        let scale: CGFloat
        let finalImage: NSImage
        
        if maxSide > CGFloat(targetSize) {
            scale = CGFloat(targetSize) / maxSide
            let newWidth = Int(originalWidth * scale)
            let newHeight = Int(originalHeight * scale)
            
            // Resize using bicubic interpolation (high quality)
            let resizedImage = NSImage(size: NSSize(width: newWidth, height: newHeight))
            resizedImage.lockFocus()
            NSGraphicsContext.current?.imageInterpolation = .high  // Bicubic equivalent
            self.draw(in: NSRect(x: 0, y: 0, width: newWidth, height: newHeight),
                      from: NSRect(origin: .zero, size: self.size),
                      operation: .copy, fraction: 1.0)
            resizedImage.unlockFocus()
            finalImage = resizedImage
        } else {
            // No resizing needed, return original image and scale=1.0
            scale = 1.0
            finalImage = self
        }
        
        // Convert to CVPixelBuffer - CoreML models often expect fixed input sizes
        // Let's create a square canvas like the original, but without letterboxing artifacts
        guard let finalCGImage = finalImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("❌ Failed to get CGImage from final image")
            return (nil, scale, finalImage)
        }
        
        // Use the target size for the pixel buffer (model likely expects 1600x1600)
        let bufferWidth = targetSize
        let bufferHeight = targetSize
        
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferWidthKey: bufferWidth,
            kCVPixelBufferHeightKey: bufferHeight,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ]
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault, bufferWidth, bufferHeight,
                                        kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            print("❌ Failed to create pixel buffer: \(status)")
            return (nil, scale, finalImage)
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                 width: bufferWidth,
                                 height: bufferHeight,
                                 bitsPerComponent: 8,
                                 bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                 space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            print("❌ Failed to create graphics context")
            return (nil, scale, finalImage)
        }
        
        // Fill with black background first
        ctx.setFillColor(CGColor.black)
        ctx.fill(CGRect(x: 0, y: 0, width: bufferWidth, height: bufferHeight))
        
        // Center the image in the buffer
        let imageWidth = finalCGImage.width
        let imageHeight = finalCGImage.height
        let xOffset = (bufferWidth - imageWidth) / 2
        let yOffset = (bufferHeight - imageHeight) / 2
        
        ctx.draw(finalCGImage, in: CGRect(x: xOffset, y: yOffset, width: imageWidth, height: imageHeight))
        
        return (buffer, scale, finalImage)
    }
    
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
}

// MARK: - Main CoreML Inference Function
func runInference() {
    // === Load the CoreML Model ===
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
    let fileManager = FileManager.default
    
    for path in possibleModelPaths {
        let modelPath = URL(fileURLWithPath: path)
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: modelPath.path, isDirectory: &isDirectory)
        
        if exists {
            do {
                if path.hasSuffix(".mlpackage") {
                    let compiledURL = try MLModel.compileModel(at: modelPath)
                    let loadedModel = try MLModel(contentsOf: compiledURL)
                    model = loadedModel
                    usedPath = path
                    break
                } else {
                    let loadedModel = try MLModel(contentsOf: modelPath)
                    model = loadedModel
                    usedPath = path
                    break
                }
            } catch {
                if path.hasSuffix(".mlpackage") {
                    do {
                        let loadedModel = try MLModel(contentsOf: modelPath)
                        model = loadedModel
                        usedPath = path
                        break
                    } catch {
                        continue
                    }
                }
            }
        }
    }
    
    guard let finalModel = model else {
        print("❌ Failed to load CoreML model")
        return
    }
    print("✅ Model loaded from: \(usedPath)")
    
    // Print essential model information
    let modelDescription = finalModel.modelDescription
    print("📋 Inputs: \(modelDescription.inputDescriptionsByName.keys)")
    print("📋 Outputs: \(modelDescription.outputDescriptionsByName.keys)")
    
    // === Load images from folder ===
    let imagesFolder = URL(fileURLWithPath: "./test_images/New images 1920x1440 og")
    
    guard let imageFiles = try? fileManager.contentsOfDirectory(at: imagesFolder, includingPropertiesForKeys: nil, options: []) else {
        print("❌ Cannot access images folder: \(imagesFolder.path)")
        return
    }
    
    let filteredImages = imageFiles.filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
    
    if filteredImages.isEmpty {
        print("❌ No images found")
        return
    }
    print("📸 Processing \(filteredImages.count) images")
    
    // === Define target size for resizing ===
    let targetSize = 1600
    
    // === Perform inference on each image ===
    for (index, imageURL) in filteredImages.enumerated() {
        print("\n🔍 Processing (\(index + 1)/\(filteredImages.count)): \(imageURL.lastPathComponent)")
        
        guard let nsImage = NSImage(contentsOf: imageURL) else {
            print("❌ Could not load image")
            continue
        }
        
        let (pixelBuffer, scale, resizedImage) = nsImage.resizedBicubicMaxSide(to: targetSize)
        
        guard let buffer = pixelBuffer else {
            print("❌ Failed to create pixel buffer")
            continue
        }
        
        // === STEP 1: Try inference ===
        var prediction: MLFeatureProvider?
        let inputName = modelDescription.inputDescriptionsByName.keys.first ?? "image"
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)])
            prediction = try finalModel.prediction(from: input)
            print("✅ Inference succeeded")
        } catch {
            print("❌ Inference failed: \(error)")
            continue
        }
        
        guard let finalPrediction = prediction else {
            print("❌ No prediction result")
            continue
        }
        
        // === STEP 2: Find output ===
        print("📦 Available outputs: \(finalPrediction.featureNames)")
        
        var rawOutput: MLMultiArray?
        let possibleOutputNames = ["output", "output0", "Identity", "Identity:0", "var_914", "1", "predictions", "detections"]
        
        for outputName in possibleOutputNames {
            if let output = finalPrediction.featureValue(for: outputName)?.multiArrayValue {
                rawOutput = output
                print("✅ Using output: '\(outputName)', shape: \(output.shape)")
                break
            }
        }
        
        guard let output = rawOutput else {
            print("❌ No valid output found")
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
        
        // === STEP 3: Parse detections ===
        let detections = YOLOv8OutputParser.parse(
            rawOutput: output,
            confidenceThreshold: 0.25,
            scale: scale,
            originalImageSize: nsImage.size,
            targetSize: targetSize
        )
        
        print("🎯 Found \(detections.count) detections")
        
        // === STEP 4: Save result ===
        let annotatedImage = resizedImage.drawDetections(detections)
        let outURL = imagesFolder.appendingPathComponent("pred_\(imageURL.lastPathComponent)")
        annotatedImage.saveJPG(to: outURL)
        print("✅ Saved result")
    }
    
    print("\n🎉 Done!")
}

// MARK: - Main Entry Point
runInference()