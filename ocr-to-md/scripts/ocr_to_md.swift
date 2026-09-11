#!/usr/bin/env swift
import CoreGraphics
import Foundation
import ImageIO
import Vision

let imageExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "heif", "tif", "tiff", "gif", "bmp"]

struct Box {
    let text: String
    let x: CGFloat
    let y: CGFloat
    let height: CGFloat
}

enum OCRError: Error, CustomStringConvertible {
    case loadFailed(String)
    case visionFailed(String)

    var description: String {
        switch self {
        case .loadFailed(let path): return "Could not load image: \(path)"
        case .visionFailed(let msg): return "Vision OCR failed: \(msg)"
        }
    }
}

func loadCGImage(url: URL) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let thumbOpts: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 2200,
        kCGImageSourceShouldCache: false,
        kCGImageSourceShouldAllowFloat: false,
    ]
    if let thumb = CGImageSourceCreateThumbnailAtIndex(src, 0, thumbOpts as CFDictionary) {
        return thumb
    }
    let opts: [CFString: Any] = [
        kCGImageSourceShouldCache: false,
        kCGImageSourceShouldAllowFloat: false,
    ]
    return CGImageSourceCreateImageAtIndex(src, 0, opts as CFDictionary)
}

func recognize(image: CGImage) throws -> [Box] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0.008
    request.recognitionLanguages = ["en-US"]
    if #available(macOS 13.0, *) {
        request.automaticallyDetectsLanguage = true
    }

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        throw OCRError.visionFailed(error.localizedDescription)
    }

    var boxes: [Box] = []
    for obs in request.results ?? [] {
        guard let candidate = obs.topCandidates(1).first else { continue }
        let bb = obs.boundingBox
        boxes.append(Box(text: candidate.string, x: bb.minX, y: bb.minY, height: bb.height))
    }
    return boxes
}

func boxesToText(_ boxes: [Box]) -> String {
    var sorted = boxes
    sorted.sort { a, b in
        if abs(a.y - b.y) > max(a.height, b.height) * 0.5 {
            return a.y > b.y
        }
        return a.x < b.x
    }

    var lines: [[Box]] = []
    for box in sorted {
        if let last = lines.last, let first = last.first,
           abs(first.y - box.y) <= max(first.height, box.height) * 0.65 {
            lines[lines.count - 1].append(box)
        } else {
            lines.append([box])
        }
    }
    return lines.map { $0.map(\.text).joined(separator: " ") }.joined(separator: "\n")
}

func markdown(title: String, embed: String, body: String) -> String {
    """
    # \(title)

    ![[\(embed)]]

    \(body)

    """
}

func printUsage() {
    fputs(
        """
        Usage: ocr_to_md.swift [--force] [--stdout] [--engine vision|tesseract] <image> [image...]

        Writes a sibling .md file next to each image (Obsidian wikilink + OCR text).
        --force    overwrite existing .md
        --stdout   print markdown instead of writing a file

        """,
        stderr
    )
}

var force = false
var stdoutOnly = false
var paths: [String] = []

var args = Array(CommandLine.arguments.dropFirst())
while let arg = args.first {
    args.removeFirst()
    switch arg {
    case "--force": force = true
    case "--stdout": stdoutOnly = true
    case "--engine":
        _ = args.first.map { args.removeFirst(); let _ = $0 }
    case "-h", "--help":
        printUsage()
        exit(0)
    default:
        if arg.hasPrefix("-") {
            fputs("Unknown flag: \(arg)\n", stderr)
            printUsage()
            exit(2)
        }
        paths.append(arg)
    }
}

if paths.isEmpty {
    printUsage()
    exit(2)
}

var failed = 0
for path in paths {
    let url = URL(fileURLWithPath: path).standardizedFileURL
    let ext = url.pathExtension.lowercased()
    if !imageExts.contains(ext) {
        fputs("Skip (not an image): \(path)\n", stderr)
        continue
    }

    guard let cgImage = loadCGImage(url: url) else {
        fputs("\(OCRError.loadFailed(url.path))\n", stderr)
        failed += 1
        continue
    }

    let text: String
    do {
        text = boxesToText(try recognize(image: cgImage))
    } catch {
        fputs("\(error)\n", stderr)
        failed += 1
        continue
    }

    let title = url.deletingPathExtension().lastPathComponent
    let md = markdown(title: title, embed: url.lastPathComponent, body: text.isEmpty ? "_No text recognized._" : text)

    if stdoutOnly {
        print(md)
        continue
    }

    let outURL = url.deletingPathExtension().appendingPathExtension("md")
    if FileManager.default.fileExists(atPath: outURL.path) && !force {
        fputs("Exists (use --force): \(outURL.path)\n", stderr)
        continue
    }

    do {
        try md.write(to: outURL, atomically: true, encoding: .utf8)
        print(outURL.path)
    } catch {
        fputs("Write failed \(outURL.path): \(error)\n", stderr)
        failed += 1
    }
}

exit(failed == 0 ? 0 : 1)
