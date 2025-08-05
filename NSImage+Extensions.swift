import AppKit
import CoreVideo

extension NSImage {
    /// Resizes the image down (if needed) and letterboxes it into a square canvas.
    /// Returns: A tuple containing the pixel buffer, scale factor, xOffset, yOffset, and the final NSImage used for inference.
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