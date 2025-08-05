import CoreML
import Vision

/// A wrapper to match the expected CoreML model input interface.
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