import Foundation
import CoreML

struct Detection {
    let rect: CGRect
    let score: Float
    let classIndex: Int
    let className: String
}

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
        
        print("🎯 Found \(detections.count) raw detections before NMS")
        
        // Apply NMS if model was exported without NMS
        let finalDetections = performNMS(detections: detections, iouThreshold: 0.5)
        
        print("🎯 Final detections after NMS: \(finalDetections.count)")
        
        return finalDetections
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