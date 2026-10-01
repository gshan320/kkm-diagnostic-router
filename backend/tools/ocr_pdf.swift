// OCR a scanned PDF with Apple's Vision framework - no installs on macOS.
//
// Why: two MOH CPGs arrive as page images with no text layer at all (Sore
// Throat, Non-Variceal Upper GI Bleeding). Indexed as-is they contribute
// nothing. This writes <pdf>.ocr.json beside the PDF - one string per page -
// and ingest.py uses that text for any page whose own text layer is empty.
//
//   swift backend/tools/ocr_pdf.swift "backend/data/raw_pdfs/<file>.pdf"
//
// Accurate recognition, English, language correction on. Scale 2.5 renders
// small print in tables legibly; lower it if a large document is slow.

import Foundation
import PDFKit
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: swift ocr_pdf.swift <file.pdf>\n".data(using: .utf8)!)
    exit(2)
}
let path = args[1]
guard let doc = PDFDocument(url: URL(fileURLWithPath: path)) else {
    FileHandle.standardError.write("cannot open \(path)\n".data(using: .utf8)!)
    exit(1)
}

let scale: CGFloat = 2.5
var pages: [String] = []
// Per line: text, confidence and the normalised bounding box (origin top-left),
// so a caller can read an image region column by column and drop low-confidence
// noise (the twice-compressed flow charts of the MOH AMO guideline, 2026-09-30).
var lines: [[[String: Any]]] = []
for i in 0..<doc.pageCount {
    guard let page = doc.page(at: i) else { pages.append(""); lines.append([]); continue }
    let box = page.bounds(for: .mediaBox)
    let size = NSSize(width: box.width * scale, height: box.height * scale)
    let image = NSImage(size: size)
    image.lockFocus()
    if let ctx = NSGraphicsContext.current?.cgContext {
        ctx.setFillColor(NSColor.white.cgColor)
        ctx.fill(CGRect(origin: .zero, size: size))
        ctx.scaleBy(x: scale, y: scale)
        page.draw(with: .mediaBox, to: ctx)
    }
    image.unlockFocus()
    guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        pages.append(""); lines.append([]); continue
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["en-US"]
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    try? handler.perform([request])
    // Top-to-bottom, then left-to-right. Lines are bucketed into rows of
    // ~0.6% of the page height so the ordering is a strict total order - a
    // pairwise "close enough" comparison is not transitive and shuffles lines.
    func row(_ o: VNRecognizedTextObservation) -> Int { Int((1.0 - o.boundingBox.midY) * 160) }
    let obs = (request.results ?? []).sorted {
        let ra = row($0), rb = row($1)
        return ra != rb ? ra < rb : $0.boundingBox.minX < $1.boundingBox.minX
    }
    let text = obs.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    pages.append(text)
    lines.append(obs.compactMap { o -> [String: Any]? in
        guard let c = o.topCandidates(1).first else { return nil }
        let b = o.boundingBox
        return ["t": c.string, "c": Double(c.confidence),
                "x": Double(b.minX), "y": Double(1.0 - b.maxY), "w": Double(b.width), "h": Double(b.height)]
    })
    FileHandle.standardError.write("page \(i + 1)/\(doc.pageCount): \(text.count) chars\n".data(using: .utf8)!)
}

let out = URL(fileURLWithPath: path + ".ocr.json")
let data = try JSONSerialization.data(withJSONObject: ["engine": "Apple Vision (accurate)",
                                                       "pages": pages, "lines": lines], options: [.prettyPrinted])
try data.write(to: out)
print("\(pages.count) pages -> \(out.path)")
