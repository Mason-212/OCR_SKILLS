#!/usr/bin/env swift
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import Vision

/// Load a photo (capped decode), rotate handwriting upright, enhance contrast,
/// write one full-page JPEG for glm-ocr plus on-device Vision hints (spelling / underlines).
enum PrepError: Error { case load(String), write(String) }

let imageExts: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "heif", "tif", "tiff", "gif", "bmp"]
/// 8GB M1: one full page (not column crops). Spend RAM on resolution + 8k–10k context.
let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024.0 * 1024.0 * 1024.0)
let lowMem = ProcessInfo.processInfo.environment["OCR_LOW_MEM"] == "1" || ramGB <= 8.5
let workingMaxSide = lowMem ? 2800 : 3600
let tileMaxSide = lowMem ? 1792 : 2048
let previewMaxSide = lowMem ? 1400 : 1600
let orientMaxSide = lowMem ? 1200 : 1400
let jpegQuality = lowMem ? 0.88 : 0.93

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
    let preview = downscale(image, maxSide: orientMaxSide, quality: .medium)
    var bestTimes = 0
    var bestScore = -1.0
    for t in 0..<4 {
        let rotated = t == 0 ? preview : rotate(preview, times90: t)
        // Fast pass is enough to pick upright vs sideways; accurate OCR runs later.
        let score = orientationScore(detectText(rotated, accurate: false, correct: false))
        if score > bestScore {
            bestScore = score
            bestTimes = t
        }
    }
    fputs("orient rotate=\(bestTimes) score=\(String(format: "%.1f", bestScore))\n", stderr)
    return bestTimes == 0 ? image : rotate(image, times90: bestTimes)
}

func enhanceHandwriting(_ image: CGImage) -> CGImage {
    // Keep the photo close to what glm-ocr was trained on. Heavy contrast/sharpen
    // fragments ballpoint strokes and makes cursive look like separate glyphs.
    var ci = CIImage(cgImage: image)
    ci = ci.applyingFilter("CIHighlightShadowAdjust", parameters: [
        "inputShadowAmount": 0.32,
        "inputHighlightAmount": 0.78,
    ])
    ci = ci.applyingFilter("CIColorControls", parameters: [
        kCIInputContrastKey: 1.18,
        kCIInputSaturationKey: 0.86,
        kCIInputBrightnessKey: 0.02,
    ])
    ci = ci.applyingFilter("CIUnsharpMask", parameters: [
        kCIInputRadiusKey: 1.05,
        kCIInputIntensityKey: 0.32,
    ])
    let rect = ci.extent.integral
    return ciContext.createCGImage(ci, from: rect) ?? image
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

func looksLikeTable(_ boxes: [TextBox], y0: CGFloat, y1: CGFloat) -> Bool {
    let inBand = boxes.filter { box in
        let my = imageMidY(box)
        return my >= y0 && my < y1 && !box.text.trimmingCharacters(in: .whitespaces).isEmpty
    }
    guard inBand.count >= 8 else { return false }
    let short = inBand.filter { $0.text.count <= 10 && $0.w < 0.30 }
    return short.count >= 6 && short.count * 2 >= inBand.count
}

func guessedColumns(_ boxes: [TextBox], y0: CGFloat, y1: CGFloat) -> Int {
    if looksLikeTable(boxes, y0: y0, y1: y1) { return 1 }
    let xs = boxes.compactMap { box -> CGFloat? in
        let my = imageMidY(box)
        guard my >= y0 && my < y1 else { return nil }
        guard box.w > 0.04, box.w < 0.48, box.text.count >= 3 else { return nil }
        return box.midX
    }.sorted()
    guard xs.count >= 8 else { return 1 }
    var gaps: [(CGFloat, CGFloat)] = []
    for i in 0..<(xs.count - 1) {
        let gap = xs[i + 1] - xs[i]
        if gap >= 0.10 {
            gaps.append((gap, (xs[i] + xs[i + 1]) / 2))
        }
    }
    gaps.sort { $0.0 > $1.0 }
    let wide = gaps.filter { $0.0 >= 0.16 && $0.1 > 0.22 && $0.1 < 0.78 }
    if wide.count >= 2 {
        let splits = [wide[0].1, wide[1].1].sorted()
        if splits[1] - splits[0] >= 0.18 {
            let a = xs.filter { $0 < splits[0] }.count
            let b = xs.filter { $0 >= splits[0] && $0 < splits[1] }.count
            let c = xs.filter { $0 >= splits[1] }.count
            let smallest = min(a, min(b, c))
            let largest = max(a, max(b, c))
            if smallest >= 4 && largest <= smallest * 3 { return 3 }
        }
    }
    let centered = gaps.filter { $0.0 >= 0.13 && $0.1 >= 0.32 && $0.1 <= 0.68 }
    if let best = centered.first {
        let left = xs.filter { $0 < best.1 }.count
        let right = xs.filter { $0 >= best.1 }.count
        if left >= 4 && right >= 4 && min(left, right) * 4 >= max(left, right) {
            return 2
        }
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

func spanningBands(_ boxes: [TextBox]) -> [(CGFloat, CGFloat)] {
    var raw: [(CGFloat, CGFloat)] = []
    for box in boxes where box.w >= 0.70 && box.text.count >= 8 {
        let top = max(0, imageTop(box) - 0.006)
        let bot = min(1, imageBot(box) + 0.010)
        if bot - top < 0.18 {
            raw.append((top, bot))
        }
    }
    raw.sort { $0.0 < $1.0 }
    var merged: [(CGFloat, CGFloat)] = []
    for band in raw {
        if let last = merged.last, band.0 <= last.1 + 0.025 {
            merged[merged.count - 1] = (last.0, max(last.1, band.1))
        } else {
            merged.append(band)
        }
    }
    return merged.filter { $0.1 - $0.0 < 0.20 }
}

func appendBand(
    _ rects: inout [(String, String, Int, Int, Int, Int, Int)],
    name: String,
    column: String,
    zone: Int,
    width: Int,
    height: Int,
    y0: CGFloat,
    y1: CGFloat,
    x0: CGFloat = 0,
    x1: CGFloat = 1
) {
    let topY = max(0, Int(y0 * CGFloat(height)))
    let botY = min(height, Int(y1 * CGFloat(height)))
    let h = botY - topY
    let left = max(0, Int(x0 * CGFloat(width)))
    let right = min(width, Int(x1 * CGFloat(width)))
    let w = right - left
    guard h >= 64, w >= 64, y1 - y0 >= 0.05 else { return }
    rects.append((name, column, zone, left, topY, w, h))
}

func tileRects(
    width: Int,
    height: Int,
    hintBoxes: [TextBox],
    boxes: [Cluster]
) -> [(name: String, column: String, zone: Int, x: Int, y: Int, w: Int, h: Int)] {
    _ = hintBoxes
    _ = boxes
    var rects: [(String, String, Int, Int, Int, Int, Int)] = []
    // One full page, like attaching the photo to Cursor. Column/zone crops
    // cut handwriting mid-word and stitch it back in the wrong order.
    rects.append(("page", "full", 0, 0, 0, width, height))
    // Optional lower band: used only if glm-ocr hits the generation cap.
    if height >= 1500 {
        let y = Int(Double(height) * 0.40)
        let h = height - y
        if h >= 96 {
            rects.append(("page-lower", "full", 1, 0, y, width, h))
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
let layoutCols = guessedColumns(hintBoxes, y0: 0.06, y1: 0.94)
let clusters = clusterText(hintBoxes)
let boxed = boxedClusters(clusters, twoColumn: twoCol)
let underlines = underlinedPhrases(preview, boxes: hintBoxes)

var manifest: [[String: Any]] = []
let iw = page.width
let ih = page.height
let maxCols = max(layoutCols, twoCol ? 2 : 1)
for (name, column, zone, xTop, yTop, w, h) in tileRects(width: iw, height: ih, hintBoxes: hintBoxes, boxes: boxed) {
    guard let tile = crop(page, x: xTop, y: yTop, w: w, h: h) else { continue }
    let send = downscale(tile, maxSide: tileMaxSide)
    let tileURL = outDir.appendingPathComponent("\(name).jpg")
    try writeJPEG(send, to: tileURL, quality: jpegQuality)
    manifest.append([
        "file": tileURL.lastPathComponent,
        "prompt": "Text Recognition:",
        "column": column,
        "zone": zone,
        "x": xTop,
        "y": yTop,
        "w": w,
        "h": h,
        "continuation": name == "page-lower",
    ])
}

func redInkWords(_ image: CGImage) -> [String] {
    guard let ctx = rgbContext(width: image.width, height: image.height) else { return [] }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    guard let data = ctx.data else { return [] }
    let bpr = ctx.bytesPerRow
    let ptr = data.bindMemory(to: UInt8.self, capacity: bpr * image.height)
    var sawRed = false
    let step = lowMem ? 2 : 1
    for y in stride(from: 0, to: image.height, by: step) {
        for x in stride(from: 0, to: image.width, by: step) {
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
