"""PDF extraction strategies for different use cases.

This module provides different PDF extraction backends:
- UnstructuredExtractor: Advanced layout analysis (default, best quality)
- PyMuPDFExtractor: Fast extraction for clean PDFs
- VisionExtractor: OCR for scanned PDFs (macOS only)
- NougatExtractor: Specialized for academic papers with equations
"""

import io
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class PDFExtractor(ABC):
    """Abstract base class for PDF extraction strategies."""

    @abstractmethod
    def extract(self, file_like: io.BytesIO, pages: Optional[List[int]] = None) -> List[Dict]:
        """Extract text from PDF.

        Args:
            file_like: PDF file as BytesIO object
            pages: Optional list of page numbers to extract (1-indexed). None = all pages.

        Returns:
            List of text elements with metadata:
            [
                {
                    "text": "extracted text",
                    "metadata": {
                        "page_number": 1,
                        "points": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]  # optional coordinates
                    }
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this extractor."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this extractor and when to use it."""
        pass


class UnstructuredExtractor(PDFExtractor):
    """PDF extraction using unstructured.io with advanced layout analysis.

    Best for: General purpose, complex layouts, coordinate extraction
    Requires: unstructured[local-inference], detectron2
    Size: ~500MB dependencies
    """

    def extract(self, file_like: io.BytesIO, pages: Optional[List[int]] = None) -> List[Dict]:
        from unstructured.partition.auto import partition

        print("Parsing PDF with layout analysis (strategy='hi_res')...")
        if pages:
            print(f"  Page filter active: extracting only pages {sorted(pages)}")

        try:
            partitioned_elements = partition(
                file=file_like,
                strategy="hi_res",
                content_type="application/pdf",
                include_page_breaks=True
            )
            print(f"Unstructured 'hi_res' returned {len(partitioned_elements)} raw elements.")
        except Exception as e:
            print(f"Unstructured 'hi_res' strategy failed: {e}. Falling back to 'fast'.")
            try:
                file_like.seek(0)
                partitioned_elements = partition(
                    file=file_like,
                    strategy="fast",
                    content_type="application/pdf",
                    include_page_breaks=True
                )
                print(f"Unstructured 'fast' returned {len(partitioned_elements)} raw elements.")
            except Exception as e2:
                print(f"Unstructured 'fast' strategy also failed: {e2}.")
                return [{
                    "text": "Error: Unstructured parsing failed.",
                    "metadata": {"page_number": 1, "points": None}
                }]

        # Convert pages to set for faster lookup
        pages_set = set(pages) if pages else None

        element_list = []
        current_page = 1
        print("\n--- Processing elements (checking for points) ---")

        for i, el in enumerate(partitioned_elements):
            meta_dict = el.metadata.to_dict()

            page_num_meta = meta_dict.get("page_number")
            if page_num_meta is not None:
                current_page = page_num_meta

            # Skip if page filtering is enabled and current page not in list
            if pages_set and current_page not in pages_set:
                continue

            # Extract coordinate points if available
            points = None
            coords_meta = meta_dict.get("coordinates")
            if coords_meta:
                points = coords_meta.get("points")

            location_data = {
                "page_number": current_page,
                "points": points
            }

            element_text = str(el).strip()
            if element_text:
                element_list.append({
                    "text": element_text,
                    "metadata": location_data
                })

        print("--- Finished processing elements ---")
        if pages_set:
            print(f"Unstructured: Found {len(element_list)} text elements from pages {sorted(pages_set)}.")
        else:
            print(f"Unstructured: Found {len(element_list)} text elements from all pages.")

        if not element_list:
            return [{
                "text": "Warning: Unstructured found no text elements.",
                "metadata": {"page_number": 1, "points": None}
            }]

        return element_list

    def get_name(self) -> str:
        return "Unstructured (Advanced)"

    def get_description(self) -> str:
        return "Advanced layout analysis with coordinate extraction. Best for general use."


class PyMuPDFExtractor(PDFExtractor):
    """PDF extraction using PyMuPDF for fast, lightweight extraction.

    Best for: Clean PDFs with text layers, speed-critical applications
    Requires: pymupdf
    Size: ~15MB
    """

    def extract(self, file_like: io.BytesIO, pages: Optional[List[int]] = None) -> List[Dict]:
        import fitz  # PyMuPDF

        print("Parsing PDF with PyMuPDF (fast extraction)...")
        if pages:
            print(f"  Page filter active: extracting only pages {sorted(pages)}")

        doc = fitz.open(stream=file_like, filetype="pdf")
        element_list = []

        pages_set = set(pages) if pages else None

        for page_num in range(len(doc)):
            page_index = page_num + 1  # 1-indexed

            # Skip if page filtering is enabled and current page not in list
            if pages_set and page_index not in pages_set:
                continue

            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")

                        if line_text.strip():
                            # Extract bounding box coordinates
                            bbox = line.get("bbox")
                            points = None
                            if bbox:
                                # Convert to points format: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
                                x0, y0, x1, y1 = bbox
                                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

                            element_list.append({
                                "text": line_text.strip(),
                                "metadata": {
                                    "page_number": page_index,
                                    "points": points
                                }
                            })

        doc.close()

        if pages_set:
            print(f"PyMuPDF: Found {len(element_list)} text elements from pages {sorted(pages_set)}.")
        else:
            print(f"PyMuPDF: Found {len(element_list)} text elements from all pages.")

        if not element_list:
            return [{
                "text": "Warning: PyMuPDF found no text elements.",
                "metadata": {"page_number": 1, "points": None}
            }]

        return element_list

    def get_name(self) -> str:
        return "PyMuPDF (Fast)"

    def get_description(self) -> str:
        return "Fast extraction for clean PDFs with text layers. Lightweight and efficient."


class VisionExtractor(PDFExtractor):
    """PDF extraction using Apple Vision Framework for OCR.

    Best for: Scanned PDFs on macOS
    Requires: macOS, pyobjc-framework-Vision, pyobjc-framework-Quartz
    Platform: macOS only
    """

    def extract(self, file_like: io.BytesIO, pages: Optional[List[int]] = None) -> List[Dict]:
        import sys
        if sys.platform != "darwin":
            raise RuntimeError("VisionExtractor is only available on macOS")

        from Vision import VNRecognizeTextRequest, VNImageRequestHandler
        from Quartz import (
            CGPDFDocumentCreateWithProvider,
            CGDataProviderCreateWithCFData,
            CGPDFDocumentGetPage,
        )
        from Foundation import NSData
        import Quartz

        print("Parsing PDF with Apple Vision Framework (OCR)...")
        if pages:
            print(f"  Page filter active: extracting only pages {sorted(pages)}")

        # Load PDF
        pdf_data = NSData.dataWithBytes_length_(file_like.read(), len(file_like.getvalue()))
        provider = CGDataProviderCreateWithCFData(pdf_data)
        pdf_doc = CGPDFDocumentCreateWithProvider(provider)

        if not pdf_doc:
            return [{
                "text": "Error: Failed to load PDF with Vision Framework.",
                "metadata": {"page_number": 1, "points": None}
            }]

        num_pages = Quartz.CGPDFDocumentGetNumberOfPages(pdf_doc)
        element_list = []
        pages_set = set(pages) if pages else None

        for page_num in range(1, num_pages + 1):
            # Skip if page filtering is enabled and current page not in list
            if pages_set and page_num not in pages_set:
                continue

            page = CGPDFDocumentGetPage(pdf_doc, page_num)
            if not page:
                continue

            # Create image from PDF page
            # Note: This is simplified - full implementation would require more Vision API code
            # For now, return a placeholder
            element_list.append({
                "text": f"[Vision OCR not fully implemented for page {page_num}]",
                "metadata": {
                    "page_number": page_num,
                    "points": None
                }
            })

        if pages_set:
            print(f"Vision: Processed {len(element_list)} pages from {sorted(pages_set)}.")
        else:
            print(f"Vision: Processed {len(element_list)} pages.")

        return element_list

    def get_name(self) -> str:
        return "Apple Vision (OCR)"

    def get_description(self) -> str:
        return "OCR for scanned PDFs using Apple Vision Framework. macOS only."


class NougatExtractor(PDFExtractor):
    """PDF extraction using Nougat OCR for academic papers.

    Best for: Academic papers with equations, mathematical notation
    Requires: nougat-ocr, transformers
    Size: ~1.5GB model
    Speed: 5-15 seconds per page
    """

    def extract(self, file_like: io.BytesIO, pages: Optional[List[int]] = None) -> List[Dict]:
        print("Parsing PDF with Nougat OCR (academic papers)...")
        if pages:
            print(f"  Page filter active: extracting only pages {sorted(pages)}")

        # Note: Full Nougat implementation would require:
        # 1. Converting PDF pages to images
        # 2. Running Nougat model on each image
        # 3. Parsing the output markdown
        # This is a placeholder for now

        print("Warning: NougatExtractor is not fully implemented yet.")
        return [{
            "text": "[Nougat extraction not yet implemented]",
            "metadata": {"page_number": 1, "points": None}
        }]

    def get_name(self) -> str:
        return "Nougat (Academic)"

    def get_description(self) -> str:
        return "Specialized OCR for academic papers with equations. Slow but accurate."


def get_available_extractors() -> Dict[str, PDFExtractor]:
    """Get a dictionary of all available PDF extractors.

    Returns:
        Dictionary mapping extractor names to extractor instances
    """
    extractors = {
        "unstructured": UnstructuredExtractor(),
        "pymupdf": PyMuPDFExtractor(),
        "vision": VisionExtractor(),
        "nougat": NougatExtractor(),
    }
    return extractors


def get_default_extractor() -> PDFExtractor:
    """Get the default PDF extractor (UnstructuredExtractor).

    Returns:
        Default PDF extractor instance
    """
    return UnstructuredExtractor()
