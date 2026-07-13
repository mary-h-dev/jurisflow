# ADR-005 — OCR Strategy for Persian Legal Documents

**Date:** 2026-07  
**Status:** Accepted (dev) / Planned (prod)

---

## Context

Users need to upload legal documents in various formats:
- Scanned PDFs (court rulings, notarized documents)
- Text-based PDFs (downloaded law texts)
- Mobile photos of physical documents (contracts, receipts)
- iPhone HEIC images

Persian documents present specific OCR challenges:
- Right-to-left script
- Mixed Arabic-Persian characters
- Handwritten annotations
- Low-quality mobile photos

---

## Decision

Use a **two-tier OCR strategy**:

| Environment | Engine | Use Case |
|-------------|--------|----------|
| Development | Gemini 2.5 Flash Vision | All file types |
| Production v1 | Gemini 2.5 Flash Vision | All file types |
| Production v2 | GLM-OCR pipeline | High-volume, cost-sensitive |

---

## File Type Support

```
Accepted formats:
  PDF  → application/pdf
  JPEG → image/jpeg, image/jpg
  PNG  → image/png
  WebP → image/webp
  HEIC → image/heic (iPhone photos)
  HEIF → image/heif

Max size: 10MB
```

---

## PDF Processing Strategy

```
PDF uploaded
  ↓
Is text extractable? (< 100 chars in first 2 pages?)
  ├── No  → scanned  → Gemini Vision (PDF as binary)
  └── Yes → text-based → pypdf direct extraction (faster, cheaper)
```

---

## Confidence Scores by Method

| Method | Confidence | Use Case |
|--------|-----------|----------|
| pypdf direct | 0.95 | Text-based PDFs |
| Gemini Vision (PDF) | 0.85 | Scanned PDFs |
| Gemini Vision (image) | 0.80 | Mobile photos |

---

## Alternatives Considered

| Option | Reason Rejected |
|--------|----------------|
| EasyOCR | Poor Persian accuracy; slow on CPU |
| Tesseract | Persian model quality inconsistent |
| Azure OCR | Cost; vendor lock-in |
| AWS Textract | No Persian support |
| Surya | CPU-only; slow for production |
| **Gemini Vision** | **Accepted — excellent Persian, simple API** |
| **GLM-OCR** | **Planned for prod — 20 pages/sec, $0.04/1000 pages** |

---

## Architecture: Decoupled from Backend

OCR runs as a **Celery async task**, completely decoupled from the HTTP request:

```
HTTP Request (upload)
  ↓
Document saved to storage
  ↓
Celery task triggered (non-blocking)
  ↓
HTTP Response (202 Accepted + task_id)

[background]
Celery Worker
  ↓
pipeline/ocr/get_ocr_engine()
  ↓
GeminiOCR.extract_from_bytes() or extract_from_image_bytes()
  ↓
Document.status = COMPLETED
  ↓
Text available for agent analysis
```

This design means:
- HTTP response is immediate (no timeout)
- OCR engine can be swapped without changing the API
- Failed tasks retry up to 3 times automatically

---

## Consequences

**Positive:**
- Gemini handles Persian handwriting better than any open-source alternative
- Single engine handles all file types (no routing complexity)
- Async processing prevents HTTP timeouts for large files

**Negative:**
- Gemini Vision rate limits affect throughput
- HEIC format requires specific MIME type handling
- Production switch to GLM-OCR requires significant infrastructure (GPU)



---

## References & Further Reading
* **High-Throughput VLM OCR Insights:** [Blue Guardrails - High-Throughput VLM OCR](https://blueguardrails.com/en/blog/high-throughput-vlm-ocr) — Explores benchmarks, cost-efficiency, and architecture strategies for scaling Vision Language Models for document text extraction.
* **Gemini Vision API Documentation:** Google Cloud Vertex AI / Google AI Studio reference for multimodal ingestion.