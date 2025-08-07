// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "YOLOv8CoreML",
    platforms: [
        .macOS(.v12)  // Minimum macOS version for CoreML support
    ],
    targets: [
        .executableTarget(
            name: "YOLOv8CoreML",
            dependencies: [],
            path: ".",
            sources: [
                "main.swift",
                "CoreMLInference.swift", 
                "YOLOv8OutputParser.swift",
                "YOLOInput.swift",
                "NSImage+Extensions.swift"
            ]
        )
    ]
)