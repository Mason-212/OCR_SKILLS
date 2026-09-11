#!/usr/bin/env swift
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import Vision

/// Load a photo (capped decode), rotate handwriting upright, enhance contrast,
/// detect columns / boxed frames / underlines, write tiles + Vision hints (on-device).
enum PrepError: Error { case load(String), write(String) }

let imageExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "heif", "tif", "tiff", "gif", "bmp"]
/// Same cap the OCR engine already used after shrink — avoid decoding 12MP HEIC.
let workingMaxSide = 3400
let tileMaxSide = 2200
let previewMaxSide = 1600
let orientMaxSide = 1400

let ciContext = CIContext(options: [
    .useSoftwareRenderer: false,
    .cacheIntermediates: false,
])

func loadCGImage(url: URL, maxSide: Int = workingMaxSide) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let thumbOpts: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: maxSide,
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

func rgbContext(width: Int, height: Int) -> CGContext? {
    CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )
}

func rotate(_ image: CGImage, times90: Int) -> CGImage {
    let t = ((times90 % 4) + 4) % 4
    if t == 0 { return image }
    var img = image
    for _ in 0..<t {
        let w = img.width
        let h = img.height
        guard let ctx = rgbContext(width: h, height: w) else { return img }
        ctx.translateBy(x: 0, y: CGFloat(w))
        ctx.rotate(by: -.pi / 2)
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        if let next = ctx.makeImage() { img = next }
    }
    return img
}

struct TextBox {
    let text: String
    let confidence: Float
    let x: CGFloat
    let y: CGFloat
    let w: CGFloat
    let h: CGFloat
    var midX: CGFloat { x + w * 0.5 }
}

func detectText(_ image: CGImage, accurate: Bool, correct: Bool) -> [TextBox] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = accurate ? .accurate : .fast
    request.usesLanguageCorrection = correct
    request.minimumTextHeight = 0.007
    request.recognitionLanguages = ["en-US"]
    if #available(macOS 13.0, *) {
        request.automaticallyDetectsLanguage = true
    }
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try? handler.perform([request])
    var boxes: [TextBox] = []
    for obs in request.results ?? [] {
        guard let cand = obs.topCandidates(1).first else { continue }
        let bb = obs.boundingBox
        boxes.append(TextBox(
            text: cand.string,
            confidence: cand.confidence,
            x: bb.minX,
            y: bb.minY,
            w: bb.width,
            h: bb.height
        ))
    }
    return boxes
}

let commonWords: Set<String> = [
    "the", "and", "of", "to", "in", "is", "a", "for", "on", "with", "as", "by",
    "from", "that", "this", "are", "was", "be", "or", "an", "at", "it", "not",
    "what", "when", "how", "all", "can", "one", "system", "earth", "living",
    "water", "air", "between", "parts", "include", "example", "community",
    "organism", "population", "ecosystem", "sphere", "chicken", "questions",
    "species", "habitat", "factors", "factor", "density", "carrying", "capacity",
    "limiting", "dependent", "independent", "size", "food", "shelter", "mates",
]

func englishHits(_ boxes: [TextBox]) -> Int {
    var hits = 0
    for box in boxes {
        let words = box.text.lowercased().split { !$0.isLetter }
        for w in words where commonWords.contains(String(w)) {
            hits += 1
        }
    }
    return hits
}

func orientationScore(_ boxes: [TextBox]) -> Double {
    guard !boxes.isEmpty else { return 0 }
    // Handwriting lines are wide when the page is upright; tall when the phone photo is sideways.
    let wide = boxes.filter { $0.w >= $0.h * 1.15 }.count
    let tall = boxes.filter { $0.h >= $0.w * 1.15 }.count
    let chars = boxes.reduce(0) { $0 + $1.text.count }
    let conf = Double(boxes.map(\.confidence).reduce(0, +)) / Double(boxes.count)
    let hits = englishHits(boxes)
    let horiz = Double(wide - tall)
    return (Double(chars) + Double(hits) * 14 + horiz * 24)
        * (0.35 + 0.65 * conf)
        * (1.0 + Double(hits))
}

func downscale(_ image: CGImage, maxSide: Int, quality: CGInterpolationQuality = .high) -> CGImage {
    let w = image.width
    let h = image.height
    let side = max(w, h)
    if side <= maxSide { return image }
    let scale = CGFloat(maxSide) / CGFloat(side)
    let nw = max(1, Int(CGFloat(w) * scale))
    let nh = max(1, Int(CGFloat(h) * scale))
    guard let ctx = rgbContext(width: nw, height: nh) else { return image }
    ctx.interpolationQuality = quality
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: nw, height: nh))
    return ctx.makeImage() ?? image
}

func upright(_ image: CGImage) -> CGImage {
    let preview = downscale(image, maxSide: orientMaxSide, quality: .high)
    var bestTimes = 0
    var bestScore = -1.0
    for t in 0..<4 {
        let rotated = t == 0 ? preview : rotate(preview, times90: t)
        let score = orientationScore(detectText(rotated, accurate: true, correct: true))
        if score > bestScore {
            bestScore = score
            bestTimes = t
        }
    }
    fputs("orient rotate=\(bestTimes) score=\(String(format: "%.1f", bestScore))\n", stderr)
    return bestTimes == 0 ? image : rotate(image, times90: bestTimes)
}

func enhanceHandwriting(_ image: CGImage) -> CGImage {
    let ci = CIImage(cgImage: image)
    // Fade blue graph-paper lines; push ink so handwriting stands off the page.
    let adjusted = ci.applyingFilter("CIColorControls", parameters: [
        kCIInputContrastKey: 1.48,
        kCIInputSaturationKey: 0.28,
        kCIInputBrightnessKey: 0.07,
    ])
    let sharp = adjusted.applyingFilter("CIUnsharpMask", parameters: [
        kCIInputRadiusKey: 2.1,
        kCIInputIntensityKey: 0.68,
    ])
    let rect = sharp.extent.integral
    return ciContext.createCGImage(sharp, from: rect) ?? image
}

func writeJPEG(_ image: CGImage, to url: URL, quality: Double) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) else {
        throw PrepError.write(url.path)
    }
    let props: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: quality]
    CGImageDestinationAddImage(dest, image, props as CFDictionary)
    if !CGImageDestinationFinalize(dest) {
        throw PrepError.write(url.path)
    }
}

func crop(_ image: CGImage, x: Int, y: Int, w: Int, h: Int) -> CGImage? {
    let rect = CGRect(x: x, y: y, width: w, height: h)
        .intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
    guard rect.width >= 32, rect.height >= 32 else { return nil }
    return image.cropping(to: rect)
}

func imageMidY(_ box: TextBox) -> CGFloat {
    1.0 - box.y - box.h * 0.5
}

func guessedColumns(_ boxes: [TextBox], y0: CGFloat, y1: CGFloat) -> Int {
    let xs = boxes.compactMap { box -> CGFloat? in
        let my = imageMidY(box)
        guard my >= y0 && my < y1 && box.w < 0.50 && box.text.count >= 2 else { return nil }
        return box.midX
    }.sorted()
    guard xs.count >= 4 else { return 1 }
    var gaps: [(CGFloat, CGFloat)] = []
    for i in 0..<(xs.count - 1) {
        let gap = xs[i + 1] - xs[i]
        if gap >= 0.08 {
            gaps.append((gap, (xs[i] + xs[i + 1]) / 2))
        }
    }
    gaps.sort { $0.0 > $1.0 }
    let wide = gaps.filter { $0.0 >= 0.14 }
    if wide.count >= 2 {
        let splits = [wide[0].1, wide[1].1].sorted()
        if splits[1] - splits[0] >= 0.14 {
            let a = xs.filter { $0 < splits[0] }.count
            let b = xs.filter { $0 >= splits[0] && $0 < splits[1] }.count
            let c = xs.filter { $0 >= splits[1] }.count
            if a >= 2 && b >= 2 && c >= 2 { return 3 }
        }
    }
    if let best = gaps.first, best.0 >= 0.10 {
        let left = xs.filter { $0 < best.1 }.count
        let right = xs.filter { $0 >= best.1 }.count
        if left >= 2 && right >= 2 { return 2 }
    }
    return 1
}

func isTwoColumn(_ boxes: [TextBox]) -> Bool {
    guessedColumns(boxes, y0: 0, y1: 1) >= 2
}

struct Cluster {
    var minX: CGFloat
    var minY: CGFloat
    var maxX: CGFloat
    var maxY: CGFloat
    var count: Int
    var width: CGFloat { maxX - minX }
    var height: CGFloat { maxY - minY }
}

func clusterText(_ boxes: [TextBox]) -> [Cluster] {
    let items = boxes.filter { $0.text.count >= 2 && $0.w < 0.92 }
    guard items.count >= 3 else { return [] }
    var parent = Array(0..<items.count)
    func find(_ i: Int) -> Int {
        var x = i
        while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }
        return x
    }
    func union(_ a: Int, _ b: Int) {
        let ra = find(a), rb = find(b)
        if ra != rb { parent[ra] = rb }
    }
    for i in 0..<items.count {
        let a = items[i]
        for j in (i + 1)..<items.count {
            let b = items[j]
            let gapX = max(0, max(a.x, b.x) - min(a.x + a.w, b.x + b.w))
            let gapY = max(0, max(a.y, b.y) - min(a.y + a.h, b.y + b.h))
            if gapX < 0.07 && gapY < 0.045 { union(i, j) }
        }
    }
    var groups: [Int: Cluster] = [:]
    for (i, box) in items.enumerated() {
        let r = find(i)
        if var c = groups[r] {
            c.minX = min(c.minX, box.x)
            c.minY = min(c.minY, box.y)
            c.maxX = max(c.maxX, box.x + box.w)
            c.maxY = max(c.maxY, box.y + box.h)
            c.count += 1
            groups[r] = c
        } else {
            groups[r] = Cluster(minX: box.x, minY: box.y, maxX: box.x + box.w, maxY: box.y + box.h, count: 1)
        }
    }
    return groups.values.filter { $0.count >= 2 && $0.width > 0.12 && $0.height > 0.04 }
}

func boxedClusters(_ clusters: [Cluster], twoColumn: Bool) -> [Cluster] {
    if twoColumn { return [] }
    return clusters.filter { c in
        c.width >= 0.22 && c.width <= 0.88
            && c.height >= 0.08 && c.height <= 0.42
            && c.count >= 3
            && c.width * c.height < 0.36
    }.sorted { $0.minY > $1.minY }
}

func underlinedPhrases(_ image: CGImage, boxes: [TextBox]) -> [String] {
    guard let ctx = rgbContext(width: image.width, height: image.height) else { return [] }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    guard let data = ctx.data else { return [] }
    let bpr = ctx.bytesPerRow
    let ptr = data.bindMemory(to: UInt8.self, capacity: bpr * image.height)
    let imgW = image.width
    let imgH = image.height
    var found: [String] = []
    for box in boxes {
        let phrase = box.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard phrase.count >= 6, box.w > 0.08 else { continue }
        let left = max(0, Int(box.x * CGFloat(imgW)))
        let right = min(imgW - 1, Int((box.x + box.w) * CGFloat(imgW)))
        let bottom = min(imgH - 1, max(0, Int((1.0 - box.y) * CGFloat(imgH))))
        let strip = max(4, Int(box.h * CGFloat(imgH) * 0.55))
        let y0 = min(imgH - 1, bottom + 1)
        let y1 = min(imgH - 1, bottom + strip)
        guard right - left > 24, y1 > y0 else { continue }
        let inset = max(3, (right - left) / 12)
        let x0 = left + inset
        let x1 = right - inset
        var bestRow = y0
        var bestDark = 0
        var y = y0
        while y <= y1 {
            var dark = 0
            var x = x0
            while x <= x1 {
                let o = y * bpr + x * 4
                let lum = (Int(ptr[o]) + Int(ptr[o + 1]) + Int(ptr[o + 2])) / 3
                if lum < 90 { dark += 1 }
                x += 2
            }
            if dark > bestDark {
                bestDark = dark
                bestRow = y
            }
            y += 1
        }
        let cols = max(1, (x1 - x0) / 2 + 1)
        // A hand underline is a thin dark stroke on one row, not letter descenders.
        var run = 0
        var bestRun = 0
        var x = x0
        while x <= x1 {
            let o = bestRow * bpr + x * 4
            let lum = (Int(ptr[o]) + Int(ptr[o + 1]) + Int(ptr[o + 2])) / 3
            if lum < 90 {
                run += 1
                if run > bestRun { bestRun = run }
            } else {
                run = 0
            }
            x += 2
        }
        var neighbor = 0
        for ny in [bestRow - 2, bestRow + 2] where ny >= y0 && ny <= y1 {
            var d = 0
            var nx = x0
            while nx <= x1 {
                let o = ny * bpr + nx * 4
                let lum = (Int(ptr[o]) + Int(ptr[o + 1]) + Int(ptr[o + 2])) / 3
                if lum < 90 { d += 1 }
                nx += 2
            }
            neighbor = max(neighbor, d)
        }
        let lineFrac = Double(bestDark) / Double(cols)
        let runFrac = Double(bestRun) / Double(cols)
        let thin = neighbor < Int(Double(bestDark) * 0.55)
        if lineFrac >= 0.62 && runFrac >= 0.42 && thin {
            found.append(phrase)
        }
    }
    return Array(Set(found)).sorted()
}

func imageTop(_ box: TextBox) -> CGFloat { 1.0 - box.y - box.h }
func imageBot(_ box: TextBox) -> CGFloat { 1.0 - box.y }

func spanningY(_ boxes: [TextBox], y0: CGFloat, y1: CGFloat) -> (CGFloat, CGFloat)? {
    let spans = boxes.filter { box in
        let my = imageMidY(box)
        return my >= y0 && my < y1 && box.w >= 0.68 && box.text.count >= 8
    }
    guard !spans.isEmpty else { return nil }
    let top = spans.map(imageTop).min()!
    let bot = spans.map(imageBot).max()!
    return (max(y0, top - 0.008), min(y1, bot + 0.012))
}

func tileRects(
    width: Int,
    height: Int,
    hintBoxes: [TextBox],
    boxes: [Cluster]
) -> [(name: String, column: String, zone: Int, x: Int, y: Int, w: Int, h: Int)] {
    var rects: [(String, String, Int, Int, Int, Int, Int)] = []
    let aspect = CGFloat(width) / CGFloat(max(1, height))
    // Phone photo of a few landscape lines: one tile. Splitting invents columns
    // and cuts sentences in half (then glm-ocr echoes the prompt).
    let shortLandscape = aspect >= 2.15 || height < 1300
    if shortLandscape {
        rects.append(("z0-C", "full", 0, 0, 0, width, height))
        return rects
    }
    let zones = 3
    let overlapN: CGFloat = 0.03
    let gutter = max(10, Int(Double(width) * 0.02))
    for i in 0..<zones {
        let y0n = max(0, CGFloat(i) / CGFloat(zones) - (i == 0 ? 0 : overlapN))
        let y1n = min(1, CGFloat(i + 1) / CGFloat(zones) + (i == zones - 1 ? 0 : overlapN))
        var colY0 = y0n
        var colY1 = y1n
        if let (sy0, sy1) = spanningY(hintBoxes, y0: y0n, y1: y1n) {
            let spanTop = Int(sy0 * CGFloat(height))
            let spanH = max(40, Int(sy1 * CGFloat(height)) - spanTop)
            rects.append(("z\(i)-F", "full", i, 0, spanTop, width, spanH))
            if (sy0 + sy1) / 2 < (y0n + y1n) / 2 {
                colY0 = min(y1n, sy1 + 0.005)
            } else {
                colY1 = max(y0n, sy0 - 0.005)
            }
        }
        let topY = Int(colY0 * CGFloat(height))
        let h = max(64, Int(colY1 * CGFloat(height)) - topY)
        guard h >= 64, colY1 - colY0 >= 0.06 else { continue }
        var cols = guessedColumns(hintBoxes, y0: colY0, y1: colY1)
        if shortLandscape { cols = 1 }
        if cols == 3 {
            let w = width / 3
            rects.append(("z\(i)-L", "left", i, 0, topY, w + gutter, h))
            rects.append(("z\(i)-M", "mid", i, w - gutter, topY, w + gutter * 2, h))
            rects.append(("z\(i)-R", "right", i, 2 * w - gutter, topY, width - (2 * w - gutter), h))
        } else if cols == 2 {
            let mid = width / 2
            rects.append(("z\(i)-L", "left", i, 0, topY, mid + gutter, h))
            rects.append(("z\(i)-R", "right", i, mid - gutter, topY, width - (mid - gutter), h))
        } else {
            rects.append(("z\(i)-C", "full", i, 0, topY, width, h))
        }
    }
    for (i, box) in boxes.prefix(3).enumerated() {
        let padX = max(8, Int(Double(width) * 0.012))
        let padY = max(8, Int(Double(height) * 0.01))
        let x = max(0, Int(box.minX * CGFloat(width)) - padX)
        let y = max(0, Int((1.0 - box.maxY) * CGFloat(height)) - padY)
        let w = min(width - x, Int(box.width * CGFloat(width)) + padX * 2)
        let h = min(height - y, Int(box.height * CGFloat(height)) + padY * 2)
        if w >= 64, h >= 48 {
            rects.append(("box-\(i)", "box", 0, x, y, w, h))
        }
    }
    return rects
}

func usage() {
    fputs("Usage: prepare_page.swift <image> <outdir>\n", stderr)
}

let args = Array(CommandLine.arguments.dropFirst())
guard args.count == 2 else { usage(); exit(2) }
let inURL = URL(fileURLWithPath: args[0]).standardizedFileURL
let outDir = URL(fileURLWithPath: args[1]).standardizedFileURL
if !imageExts.contains(inURL.pathExtension.lowercased()) {
    fputs("Not an image: \(inURL.path)\n", stderr)
    exit(2)
}

guard let loaded = loadCGImage(url: inURL) else {
    fputs("Could not load \(inURL.path)\n", stderr)
    exit(1)
}

try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let uprightPage = upright(loaded)
let page = enhanceHandwriting(downscale(uprightPage, maxSide: workingMaxSide))

let preview = downscale(page, maxSide: previewMaxSide, quality: .medium)
let hintBoxes = detectText(preview, accurate: true, correct: true)
let pageAspect = CGFloat(page.width) / CGFloat(max(1, page.height))
let twoCol = (pageAspect >= 2.15 || page.height < 1300) ? false : isTwoColumn(hintBoxes)
let clusters = clusterText(hintBoxes)
let boxed = boxedClusters(clusters, twoColumn: twoCol)
let underlines = underlinedPhrases(preview, boxes: hintBoxes)

var manifest: [[String: Any]] = []
let iw = page.width
let ih = page.height
var maxCols = 1
for (name, column, zone, xTop, yTop, w, h) in tileRects(width: iw, height: ih, hintBoxes: hintBoxes, boxes: boxed) {
    guard let tile = crop(page, x: xTop, y: yTop, w: w, h: h) else { continue }
    let send = downscale(tile, maxSide: tileMaxSide)
    let tileURL = outDir.appendingPathComponent("\(name).jpg")
    try writeJPEG(send, to: tileURL, quality: 0.92)
    let prompt: String
    if name.hasPrefix("box-") {
        prompt = "Box Recognition:"
    } else if column == "left" || column == "right" || column == "mid" {
        prompt = "Column Recognition:"
        if column == "mid" { maxCols = max(maxCols, 3) }
        else { maxCols = max(maxCols, 2) }
    } else {
        prompt = "Text Recognition:"
    }
    manifest.append([
        "file": tileURL.lastPathComponent,
        "prompt": prompt,
        "column": column,
        "zone": zone,
        "x": xTop,
        "y": yTop,
        "w": w,
        "h": h,
    ])
}

func redInkWords(_ image: CGImage) -> [String] {
    guard let ctx = rgbContext(width: image.width, height: image.height) else { return [] }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    guard let data = ctx.data else { return [] }
    let bpr = ctx.bytesPerRow
    let ptr = data.bindMemory(to: UInt8.self, capacity: bpr * image.height)
    var sawRed = false
    for y in 0..<image.height {
        for x in 0..<image.width {
            let o = y * bpr + x * 4
            let r = Int(ptr[o]), g = Int(ptr[o + 1]), b = Int(ptr[o + 2])
            let isRed = r > 130 && r > g + 35 && r > b + 35
            if isRed {
                sawRed = true
                ptr[o] = 20; ptr[o + 1] = 20; ptr[o + 2] = 20
            } else {
                ptr[o] = 255; ptr[o + 1] = 255; ptr[o + 2] = 255
            }
        }
    }
    guard sawRed, let mask = ctx.makeImage() else { return [] }
    let words = detectText(mask, accurate: false, correct: false)
        .flatMap { $0.text.split(whereSeparator: { $0.isWhitespace || $0.isPunctuation }).map(String.init) }
        .filter { $0.count >= 4 }
    return Array(Set(words)).sorted()
}

var hintWords: Set<String> = []
var hintLines: [String] = []
for box in hintBoxes.sorted(by: { $0.y == $1.y ? $0.x < $1.x : $0.y > $1.y }) {
    let line = box.text.trimmingCharacters(in: .whitespacesAndNewlines)
    if !line.isEmpty { hintLines.append(line) }
    for w in line.split(whereSeparator: { !$0.isLetter && $0 != "'" }) {
        let s = String(w)
        if s.count >= 3 { hintWords.insert(s) }
    }
}

let hints: [String: Any] = [
    "columns": max(maxCols, twoCol ? 2 : 1),
    "words": hintWords.sorted(),
    "lines": hintLines,
    "underlined": underlines,
]
let hintsURL = outDir.appendingPathComponent("vision_hints.json")
let hintsData = try JSONSerialization.data(withJSONObject: hints, options: [])
try hintsData.write(to: hintsURL)

let reds = redInkWords(preview)
let meta: [String: Any] = [
    "width": iw,
    "height": ih,
    "columns": max(maxCols, twoCol ? 2 : 1),
    "tiles": manifest,
    "red_words": reds,
]
let metaURL = outDir.appendingPathComponent("manifest.json")
let metaData = try JSONSerialization.data(withJSONObject: meta, options: [])
try metaData.write(to: metaURL)
print(outDir.path)
