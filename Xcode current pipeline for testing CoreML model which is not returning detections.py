#Main inference code
    import Foundation
    import CoreML
    import AppKit

    // === Load the CoreML Model ===
    let modelPath = URL(fileURLWithPath: "/Users/malvernnzwere/Desktop/CoreMLCompiled/1600Model_without_mosaic_rotation2_perspective0.0025.mlmodelc")
    guard let model = try? MLModel(contentsOf: modelPath) else {
        fatalError("❌ Failed to load CoreML model")
    }
    print("✅ Model loaded successfully")

    // === Load images from folder ===
    let imagesFolder = URL(fileURLWithPath: "/Users/malvernnzwere/Library/CloudStorage/OneDrive-Personal/Yolo training folders/New images 1920x1440 og")
    let fileManager = FileManager.default
    let imageFiles = try! fileManager.contentsOfDirectory(at: imagesFolder, includingPropertiesForKeys: nil)
        .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }

    if imageFiles.isEmpty {
        print("⚠️ No JPG or PNG images found in folder.")
    } else {
        print("📸 Found \(imageFiles.count) images.")
    }

    // === Define input image size used by the model ===
    let inputSize = CGSize(width: 1600, height: 1600)

    // === Perform inference on each image ===
    for imageURL in imageFiles {
        print("🔍 Processing: \(imageURL.lastPathComponent)")

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

        // Predict
        let input = YOLOInput(image: buffer)
        guard let prediction = try? model.prediction(from: input) else {
            print("❌ Inference failed for \(imageURL.lastPathComponent)")
            continue
        }

        print("📦 Available outputs: \(prediction.featureNames)")

        guard let output = prediction.featureValue(for: "var_914")?.multiArrayValue else {
            print("❌ Invalid output format for \(imageURL.lastPathComponent)")
            continue
        }

        // Parse detections
        let detections = YOLOv8OutputParser.parse(
            rawOutput: output,
            confidenceThreshold: 0.3,
            inputImageSize: inputSize,
            scale: scale,
            xOffset: xPad,
            yOffset: yPad
        )
        print("🎯 Detections count: \(detections.count)")

        // Draw and save annotated image
        let annotatedImage = resizedImage.drawDetections(detections)
        let outURL = imagesFolder.appendingPathComponent("pred_\(imageURL.lastPathComponent)")
        annotatedImage.saveJPG(to: outURL)
        print("✅ Saved: \(outURL.lastPathComponent)")
    }

    print("\n🎉 All done!")

#Image resizing
    import AppKit
    import CoreVideo

    extension NSImage {
        /// Resizes the image down (if needed) and letterboxes it into a square canvas.
        /// - Returns: A tuple containing the pixel buffer, scale factor, xOffset, yOffset, and the final NSImage used for inference.
        func resizedDownwardLetterboxedWithMetadata(to targetSize: Int = 1600) -> (CVPixelBuffer?, CGFloat, CGFloat, CGFloat, NSImage) {
            guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                print("❌ Failed to get CGImage from NSImage")
                return (nil, 1.0, 0, 0, self)
            }

            let originalWidth = CGFloat(cgImage.width)
            let originalHeight = CGFloat(cgImage.height)

            let maxSide = max(originalWidth, originalHeight)
            let scale = maxSide > CGFloat(targetSize) ? CGFloat(targetSize) / maxSide : 1.0

            let newWidth = Int(originalWidth * scale)
            let newHeight = Int(originalHeight * scale)

            // Resize the image using bicubic interpolation
            let resizedImage = NSImage(size: NSSize(width: newWidth, height: newHeight))
            resizedImage.lockFocus()
            NSGraphicsContext.current?.imageInterpolation = .high
            self.draw(in: NSRect(x: 0, y: 0, width: newWidth, height: newHeight),
                      from: NSRect(origin: .zero, size: self.size),
                      operation: .copy, fraction: 1.0)
            resizedImage.unlockFocus()

            // Create square canvas with black background
            let canvasSize = NSSize(width: targetSize, height: targetSize)
            let finalCanvas = NSImage(size: canvasSize)
            finalCanvas.lockFocus()
            NSColor.black.setFill()
            NSBezierPath(rect: NSRect(origin: .zero, size: canvasSize)).fill()

            let xOffset = (CGFloat(targetSize) - CGFloat(newWidth)) / 2.0
            let yOffset = (CGFloat(targetSize) - CGFloat(newHeight)) / 2.0

            resizedImage.draw(in: NSRect(x: xOffset, y: yOffset, width: CGFloat(newWidth), height: CGFloat(newHeight)),
                              from: NSRect(origin: .zero, size: resizedImage.size),
                              operation: .copy, fraction: 1.0)
            finalCanvas.unlockFocus()

            // Convert to CVPixelBuffer
            var pixelBuffer: CVPixelBuffer?
            let attrs: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferWidthKey: targetSize,
                kCVPixelBufferHeightKey: targetSize,
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32ARGB
            ]

            CVPixelBufferCreate(kCFAllocatorDefault, targetSize, targetSize,
                                kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pixelBuffer)

            guard let buffer = pixelBuffer else {
                return (nil, scale, xOffset, yOffset, finalCanvas)
            }

            CVPixelBufferLockBaseAddress(buffer, [])
            if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                   width: targetSize,
                                   height: targetSize,
                                   bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                   space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue),
               let finalCG = finalCanvas.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                ctx.draw(finalCG, in: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])


            return (buffer, scale, xOffset, yOffset, finalCanvas)
        }
    }

#Output parser
    import AppKit
    import CoreVideo

    extension NSImage {
        /// Resizes the image down (if needed) and letterboxes it into a square canvas.
        /// - Returns: A tuple containing the pixel buffer, scale factor, xOffset, yOffset, and the final NSImage used for inference.
        func resizedDownwardLetterboxedWithMetadata(to targetSize: Int = 1600) -> (CVPixelBuffer?, CGFloat, CGFloat, CGFloat, NSImage) {
            guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                print("❌ Failed to get CGImage from NSImage")
                return (nil, 1.0, 0, 0, self)
            }

            let originalWidth = CGFloat(cgImage.width)
            let originalHeight = CGFloat(cgImage.height)

            let maxSide = max(originalWidth, originalHeight)
            let scale = maxSide > CGFloat(targetSize) ? CGFloat(targetSize) / maxSide : 1.0

            let newWidth = Int(originalWidth * scale)
            let newHeight = Int(originalHeight * scale)

            // Resize the image using bicubic interpolation
            let resizedImage = NSImage(size: NSSize(width: newWidth, height: newHeight))
            resizedImage.lockFocus()
            NSGraphicsContext.current?.imageInterpolation = .high
            self.draw(in: NSRect(x: 0, y: 0, width: newWidth, height: newHeight),
                      from: NSRect(origin: .zero, size: self.size),
                      operation: .copy, fraction: 1.0)
            resizedImage.unlockFocus()

            // Create square canvas with black background
            let canvasSize = NSSize(width: targetSize, height: targetSize)
            let finalCanvas = NSImage(size: canvasSize)
            finalCanvas.lockFocus()
            NSColor.black.setFill()
            NSBezierPath(rect: NSRect(origin: .zero, size: canvasSize)).fill()

            let xOffset = (CGFloat(targetSize) - CGFloat(newWidth)) / 2.0
            let yOffset = (CGFloat(targetSize) - CGFloat(newHeight)) / 2.0

            resizedImage.draw(in: NSRect(x: xOffset, y: yOffset, width: CGFloat(newWidth), height: CGFloat(newHeight)),
                              from: NSRect(origin: .zero, size: resizedImage.size),
                              operation: .copy, fraction: 1.0)
            finalCanvas.unlockFocus()

            // Convert to CVPixelBuffer
            var pixelBuffer: CVPixelBuffer?
            let attrs: [CFString: Any] = [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferWidthKey: targetSize,
                kCVPixelBufferHeightKey: targetSize,
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32ARGB
            ]

            CVPixelBufferCreate(kCFAllocatorDefault, targetSize, targetSize,
                                kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pixelBuffer)

            guard let buffer = pixelBuffer else {
                return (nil, scale, xOffset, yOffset, finalCanvas)
            }

            CVPixelBufferLockBaseAddress(buffer, [])
            if let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                   width: targetSize,
                                   height: targetSize,
                                   bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                   space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue),
               let finalCG = finalCanvas.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                ctx.draw(finalCG, in: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])


            return (buffer, scale, xOffset, yOffset, finalCanvas)
        }
    }
    
#Yolo input:
    //
    //  YOLOInput.swift
    //  YoloCoreMLInference
    //
    //  Created by Malvern Nzwere on 03/08/2025.
    //
    // === YOLOInput.swift ===
    import CoreML
    import Vision

    /// A wrapper to match the expected CoreML model input interface.
    class YOLOInput: MLFeatureProvider {
        var image: CVPixelBuffer

        init(image: CVPixelBuffer) {
            self.image = image
        }

        var featureNames: Set<String> {
            return ["image"]
        }

        func featureValue(for featureName: String) -> MLFeatureValue? {
            if featureName == "image" {
                return MLFeatureValue(pixelBuffer: image)
            }
            return nil
        }
    }

#Draw detection on image
    import AppKit

    extension NSImage {
        func drawDetections(_ detections: [Detection]) -> NSImage {
            let size = self.size
            let newImage = NSImage(size: size)
            newImage.lockFocus()

            self.draw(at: .zero, from: CGRect(origin: .zero, size: size), operation: .sourceOver, fraction: 1.0)

            let context = NSGraphicsContext.current?.cgContext
            context?.setLineWidth(2.0)
            context?.setStrokeColor(NSColor.systemRed.withAlphaComponent(0.8).cgColor)

            for detection in detections {
                let rect = detection.rect

                // Draw bounding box
                context?.stroke(rect)

                // Draw label
                let label = String(format: "%.0f%%", detection.score * 100)
                let attributes: [NSAttributedString.Key: Any] = [
                    .foregroundColor: NSColor.white,
                    .backgroundColor: NSColor.red,
                    .font: NSFont.systemFont(ofSize: 12, weight: .bold)
                ]
                let labelText = NSString(string: label)
                labelText.draw(at: CGPoint(x: rect.origin.x, y: rect.origin.y - 14), withAttributes: attributes)
            }

            newImage.unlockFocus()
            return newImage
        }
    }

#Saving images
    //
    //  NSImage+Save.swift
    //  YoloCoreMLInference
    //
    //  Created by Malvern Nzwere on 03/08/2025.
    //

    import AppKit

    extension NSImage {
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
