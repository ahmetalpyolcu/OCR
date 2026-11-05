#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from tqdm import tqdm


@dataclass
class TextBlock:
    """XY-Cut ağacından çıkan her bir okuma bloğunu temsil eder."""
    order: int
    block_type: str  # "full_width" (Başlık/Tam Satır) veya "column" (Sütun)
    bbox: Tuple[int, int, int, int]
    text: str = ""


@dataclass
class XYCutOCRConfig:
    """Spiral temizliği ve Özyinelemeli XY-Cut hiperparametreleri."""
    lang: str = "tur+eng"
    spiral_margin_ratio_x: float = 0.12
    spiral_margin_ratio_y: float = 0.07
    column_search_range: Tuple[float, float] = (0.35, 0.65)
    x_valley_threshold: float = 0.07
    y_valley_threshold: float = 0.04
    min_y_gap_ratio: float = 0.012
    min_column_width_ratio: float = 0.25
    min_block_height_ratio: float = 0.035
    max_depth: int = 4
    tesseract_cmd: Optional[str] = None


class HybridXYCutOCR:
    def __init__(self, config: Optional[XYCutOCRConfig] = None):
        self.cfg = config or XYCutOCRConfig()
        if self.cfg.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.tesseract_cmd

    def preprocess_and_binarize(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        binary_inv = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=15
        )
        return self._deskew(gray, binary_inv)

    def _deskew(self, gray: np.ndarray, binary_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        edges = cv2.Canny(binary_inv, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100,
            minLineLength=max(30, gray.shape[1] // 6),
            maxLineGap=20
        )
        if lines is None:
            return gray, binary_inv

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -15.0 < angle < 15.0:
                angles.append(angle)

        if not angles:
            return gray, binary_inv

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.25:
            return gray, binary_inv

        h, w = gray.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        gray_rot = cv2.warpAffine(
            gray, rot_mat, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        bin_rot = cv2.warpAffine(
            binary_inv, rot_mat, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return gray_rot, bin_rot

    def remove_spirals(self, gray: np.ndarray, binary_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = binary_inv.shape
        clean_gray = gray.copy()
        clean_bin = binary_inv.copy()

        margin_x = int(w * self.cfg.spiral_margin_ratio_x)
        margin_y = int(h * self.cfg.spiral_margin_ratio_y)

        left_cut = self._find_spiral_gutter_cut(clean_bin[:, :margin_x], side="left")
        right_cut = self._find_spiral_gutter_cut(clean_bin[:, w - margin_x:], side="right")

        if left_cut > 0:
            clean_bin[:, :left_cut] = 0
            clean_gray[:, :left_cut] = 255
        if right_cut > 0:
            clean_bin[:, w - right_cut:] = 0
            clean_gray[:, w - right_cut:] = 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)
        valid_heights = stats[1:, cv2.CC_STAT_HEIGHT]
        filtered_h = valid_heights[(valid_heights > 5) & (valid_heights < h // 10)]
        median_char_h = float(np.median(filtered_h)) if len(filtered_h) > 0 else 18.0

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            in_left = (x + cw) < margin_x
            in_right = x > (w - margin_x)
            in_top = (y + ch) < margin_y
            touches_border = (x <= 2) or (y <= 2) or (x + cw >= w - 2) or (y + ch >= h - 2)

            if (
                touches_border
                or ((in_left or in_right) and (ch > median_char_h * 2.4 or cw > median_char_h * 2.4 or area < 15))
                or (in_top and (cw > median_char_h * 2.5 or ch > median_char_h * 2.5))
            ):
                clean_bin[labels == i] = 0
                clean_gray[labels == i] = 255

        return clean_gray, clean_bin

    def _find_spiral_gutter_cut(self, strip: np.ndarray, side: str) -> int:
        h, sw = strip.shape
        if sw == 0:
            return 0
        proj = np.sum(strip > 0, axis=0) / float(h)
        k = max(3, sw // 15)
        proj_smooth = np.convolve(proj, np.ones(k) / k, mode="same")
        if side == "right":
            proj_smooth = proj_smooth[::-1]

        peak_zone = proj_smooth[:int(sw * 0.7)]
        if len(peak_zone) == 0 or np.max(peak_zone) < 0.12:
            return 0

        peak_idx = int(np.argmax(peak_zone))
        for idx in range(peak_idx + 1, sw):
            if proj_smooth[idx] < 0.015:
                return idx + 4
        return 0

    def _build_layout_mask(self, clean_bin: np.ndarray) -> np.ndarray:
        h, w = clean_bin.shape
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1))
        horiz_lines = cv2.morphologyEx(clean_bin, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        no_lines = cv2.subtract(clean_bin, horiz_lines)

        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, h // 15)))
        vert_lines = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, vert_kernel, iterations=1)
        no_lines = cv2.subtract(no_lines, vert_lines)

        kw = max(9, int(w * 0.012))
        kh = max(5, int(h * 0.006))
        block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        return cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, block_kernel)

    def segment_xy_cut(self, clean_bin: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        h, w = clean_bin.shape
        layout_mask = self._build_layout_mask(clean_bin)

        ys, xs = np.where(layout_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [("full_width", (0, 0, w, h))]

        pad = 10
        root_bbox = (
            max(0, int(np.min(xs)) - pad),
            max(0, int(np.min(ys)) - pad),
            min(w, int(np.max(xs)) + pad),
            min(h, int(np.max(ys)) + pad)
        )

        raw_blocks: List[Tuple[str, Tuple[int, int, int, int]]] = []
        self._xy_cut_recursive(layout_mask, root_bbox, depth=0, page_shape=(h, w), out_blocks=raw_blocks)
        return raw_blocks

    def _xy_cut_recursive(
        self,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        depth: int,
        page_shape: Tuple[int, int],
        out_blocks: List[Tuple[str, Tuple[int, int, int, int]]],
        forced_type: str = "full_width"
    ) -> None:
        x1, y1, x2, y2 = bbox
        page_h, page_w = page_shape
        roi = mask[y1:y2, x1:x2]
        rh, rw = roi.shape

        min_h = int(page_h * self.cfg.min_block_height_ratio)
        min_w = int(page_w * self.cfg.min_column_width_ratio)

        if rh < min_h or rw < min_w or np.sum(roi > 0) < 200:
            return

        if depth >= self.cfg.max_depth:
            out_blocks.append((forced_type, bbox))
            return

        x_split = self._find_vertical_column_split(roi, page_w)
        if x_split is not None:
            left_bbox = self._tighten_bbox(mask, (x1, y1, x1 + x_split, y2), page_shape)
            right_bbox = self._tighten_bbox(mask, (x1 + x_split, y1, x2, y2), page_shape)

            if left_bbox:
                self._xy_cut_recursive(mask, left_bbox, depth + 1, page_shape, out_blocks, forced_type="column")
            if right_bbox:
                self._xy_cut_recursive(mask, right_bbox, depth + 1, page_shape, out_blocks, forced_type="column")
            return

        y_split = self._find_horizontal_zone_split(roi, page_shape)
        if y_split is not None:
            top_bbox = self._tighten_bbox(mask, (x1, y1, x2, y1 + y_split), page_shape)
            bottom_bbox = self._tighten_bbox(mask, (x1, y1 + y_split, x2, y2), page_shape)

            if top_bbox:
                self._xy_cut_recursive(mask, top_bbox, depth + 1, page_shape, out_blocks, forced_type=forced_type)
            if bottom_bbox:
                self._xy_cut_recursive(mask, bottom_bbox, depth + 1, page_shape, out_blocks, forced_type=forced_type)
            return

        out_blocks.append((forced_type, bbox))

    def _find_vertical_column_split(self, roi: np.ndarray, page_w: int) -> Optional[int]:
        rh, rw = roi.shape
        min_col_w = int(page_w * self.cfg.min_column_width_ratio)
        if rw < min_col_w * 2:
            return None

        vert_proj = np.sum(roi > 0, axis=0).astype(np.float32)
        smooth_w = max(11, int(rw * 0.02))
        smooth_proj = np.convolve(vert_proj, np.ones(smooth_w) / smooth_w, mode="same")

        max_density = float(np.max(smooth_proj))
        if max_density == 0:
            return None

        s1 = int(rw * self.cfg.column_search_range[0])
        s2 = int(rw * self.cfg.column_search_range[1])
        if s2 <= s1:
            return None

        center_zone = smooth_proj[s1:s2]
        min_val = float(np.min(center_zone))
        threshold = max_density * self.cfg.x_valley_threshold

        if min_val > threshold:
            return None

        valley_indices = np.where(center_zone <= max(min_val * 1.4, threshold))[0]
        split_x = s1 + int(np.median(valley_indices))

        gutter_half_w = max(6, int(rw * 0.015))
        gx1 = max(0, split_x - gutter_half_w)
        gx2 = min(rw, split_x + gutter_half_w)
        gutter_strip = roi[:, gx1:gx2]

        row_occupancy_in_gutter = np.mean(gutter_strip > 0, axis=1)
        bridging_rows = np.sum(row_occupancy_in_gutter > 0.35)
        if bridging_rows > max(8, int(rh * 0.03)):
            return None

        left_mass = float(np.sum(smooth_proj[:split_x]))
        right_mass = float(np.sum(smooth_proj[split_x:]))
        total_mass = left_mass + right_mass

        if (
            split_x < min_col_w
            or (rw - split_x) < min_col_w
            or left_mass < total_mass * 0.12
            or right_mass < total_mass * 0.12
        ):
            return None

        return split_x

    def _find_horizontal_zone_split(self, roi: np.ndarray, page_shape: Tuple[int, int]) -> Optional[int]:
        rh, rw = roi.shape
        page_h, page_w = page_shape
        min_block_h = int(page_h * self.cfg.min_block_height_ratio)
        min_gap_h = max(6, int(page_h * self.cfg.min_y_gap_ratio))

        if rh < min_block_h * 2:
            return None

        horiz_proj = np.sum(roi > 0, axis=1).astype(np.float32)
        max_h_density = float(np.max(horiz_proj))
        if max_h_density == 0:
            return None

        is_empty_row = horiz_proj <= (max_h_density * self.cfg.y_valley_threshold)

        valleys: List[Tuple[int, int]] = []
        in_valley = False
        v_start = 0

        for y in range(min_block_h, rh - min_block_h):
            if is_empty_row[y] and not in_valley:
                in_valley = True
                v_start = y
            elif not is_empty_row[y] and in_valley:
                in_valley = False
                if (y - v_start) >= min_gap_h:
                    valleys.append((v_start, y))
        if in_valley and ((rh - min_block_h) - v_start) >= min_gap_h:
            valleys.append((v_start, rh - min_block_h))

        if not valleys:
            return None

        best_split_y: Optional[int] = None
        best_score = -1.0

        for v_start, v_end in valleys:
            mid_y = (v_start + v_end) // 2
            gap_height = v_end - v_start

            top_roi = roi[:mid_y, :]
            bot_roi = roi[mid_y:, :]

            if np.sum(top_roi > 0) < 150 or np.sum(bot_roi > 0) < 150:
                continue

            top_unlocks_cols = self._find_vertical_column_split(top_roi, page_w) is not None
            bot_unlocks_cols = self._find_vertical_column_split(bot_roi, page_w) is not None

            if top_unlocks_cols or bot_unlocks_cols:
                score = 1000.0 + float(gap_height)
            else:
                gx1, gx2 = int(rw * 0.42), int(rw * 0.58)
                top_center_fill = float(np.mean(top_roi[:, gx1:gx2] > 0))
                bot_center_fill = float(np.mean(bot_roi[:, gx1:gx2] > 0))
                fill_diff = abs(top_center_fill - bot_center_fill)

                if fill_diff > 0.04:
                    score = fill_diff * 500.0 + float(gap_height)
                else:
                    continue

            if score > best_score:
                best_score = score
                best_split_y = mid_y

        return best_split_y

    def _tighten_bbox(
        self,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        page_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        page_h, page_w = page_shape
        sub = mask[y1:y2, x1:x2]
        ys, xs = np.where(sub > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        pad = 8
        return (
            max(0, x1 + int(np.min(xs)) - pad),
            max(0, y1 + int(np.min(ys)) - pad),
            min(page_w, x1 + int(np.max(xs)) + pad),
            min(page_h, y1 + int(np.max(ys)) + pad)
        )

    def _save_debug_image(
        self,
        clean_gray: np.ndarray,
        blocks: List[TextBlock],
        output_path: str
    ) -> None:
        vis = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
        for b in blocks:
            x1, y1, x2, y2 = b.bbox
            if b.block_type == "full_width":
                color = (0, 140, 255)
                label = f"#{b.order} [BASLIK / TAM]"
            else:
                color = (0, 180, 0)
                label = f"#{b.order} [SUTUN]"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                vis, label, (x1 + 5, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        cv2.imwrite(output_path, vis)


class BatchMarkdownOCR(HybridXYCutOCR):
    VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

    def process_image_array(
        self,
        image: np.ndarray,
        debug_output_path: Optional[str] = None
    ) -> List[TextBlock]:
        gray, binary_inv = self.preprocess_and_binarize(image)
        clean_gray, clean_bin = self.remove_spirals(gray, binary_inv)
        raw_segments = self.segment_xy_cut(clean_bin)

        tess_config = "--oem 3 --psm 6"
        blocks: List[TextBlock] = []

        for idx, (b_type, (x1, y1, x2, y2)) in enumerate(raw_segments, start=1):
            crop = clean_gray[y1:y2, x1:x2]
            raw_text = pytesseract.image_to_string(
                crop, lang=self.cfg.lang, config=tess_config
            ).strip()

            cleaned_text = self._clean_ocr_text(raw_text)
            if cleaned_text:
                blocks.append(TextBlock(
                    order=idx,
                    block_type=b_type,
                    bbox=(x1, y1, x2, y2),
                    text=cleaned_text
                ))

        if debug_output_path and blocks:
            self._save_debug_image(clean_gray, blocks, debug_output_path)

        return blocks

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"([a-zA-ZçğıöşüÇĞİÖŞÜ])-\s*\n\s*([a-zA-ZçğıöşüÇĞİÖŞÜ])", r"\1\2", text)

        lines = [line.strip() for line in text.splitlines()]
        reconstructed: List[str] = []
        list_pattern = re.compile(r"^(\d+[\.\)]|[a-zA-Z][\.\)]|[-•*▪►])\s+")

        for line in lines:
            if not line:
                if reconstructed and reconstructed[-1] != "":
                    reconstructed.append("")
                continue

            if list_pattern.match(line):
                reconstructed.append(line)
            elif not reconstructed or reconstructed[-1] == "":
                reconstructed.append(line)
            else:
                reconstructed[-1] = f"{reconstructed[-1]} {line}"

        return "\n".join(reconstructed).strip()

    @staticmethod
    def _format_block_as_markdown(block: TextBlock) -> str:
        text = block.text.strip()
        if not text:
            return ""

        if block.block_type == "full_width":
            lines = [l for l in text.splitlines() if l.strip()]
            if len(lines) <= 2 and len(text) < 140:
                clean_heading = " ".join(lines).lstrip("#").strip()
                return f"## {clean_heading}"

            return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())

        return text

    @staticmethod
    def _natural_sort_key(path: Path) -> List[Union[int, str]]:
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)
        ]

    def _iter_input_pages(self, input_path: Path, pdf_dpi: int = 300):
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            doc = fitz.open(str(input_path))
            zoom = pdf_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                yield (page_idx + 1, f"{input_path.stem}_sayfa_{page_idx + 1}", img_bgr)

            doc.close()

        elif input_path.is_dir():
            image_files = sorted(
                [p for p in input_path.iterdir() if p.suffix.lower() in self.VALID_IMAGE_EXTS],
                key=self._natural_sort_key
            )
            for idx, img_path in enumerate(image_files, start=1):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    yield (idx, img_path.stem, img_bgr)

    def export_single_target_to_md(
        self,
        input_path: Path,
        output_md_path: Path,
        debug_dir: Optional[Path] = None,
        pdf_dpi: int = 300
    ) -> Optional[Path]:
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            with fitz.open(str(input_path)) as d:
                total_pages = len(d)
        else:
            total_pages = len([p for p in input_path.iterdir() if p.suffix.lower() in self.VALID_IMAGE_EXTS])

        if total_pages == 0:
            return None

        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        md_sections: List[str] = [
            "---",
            f"title: \"{input_path.stem}\"",
            f"date: \"{datetime.now().strftime('%Y-%m-%d %H:%M')}\"",
            f"source: \"{input_path.name}\"",
            f"pages: {total_pages}",
            "---\n"
        ]

        page_gen = self._iter_input_pages(input_path, pdf_dpi=pdf_dpi)
        for page_num, page_label, img_bgr in tqdm(page_gen, total=total_pages, desc=f"OCR [{input_path.name}]"):
            debug_path = (
                str(debug_dir / f"{page_num:03d}_{page_label}_debug.jpg")
                if debug_dir else None
            )

            blocks = self.process_image_array(img_bgr, debug_output_path=debug_path)
            if not blocks:
                continue

            page_parts = [f"<!-- Sayfa {page_num} ({page_label}) -->"]
            for block in blocks:
                formatted = self._format_block_as_markdown(block)
                if formatted:
                    page_parts.append(formatted)
            page_parts.append("\n---")
            md_sections.append("\n\n".join(page_parts))

        final_md = "\n\n".join(md_sections).strip() + "\n"
        output_md_path.write_text(final_md, encoding="utf-8")
        return output_md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spiral Temizlemeli ve XY-Cut Çift Sütun Ayrıştırmalı Toplu Markdown OCR"
    )
    parser.add_argument("-i", "--input", required=True, help="Girdi PDF dosyası veya klasör yolu (Örn: input/)")
    parser.add_argument("-o", "--output", required=True, help="Çıktı Markdown (.md) dosyası veya klasör yolu (Örn: output/)")
    parser.add_argument("--debug-dir", default=None, help="Görsel analiz kutularının kaydedileceği klasör")
    parser.add_argument("--lang", default="tur+eng", help="Tesseract dil kodu (Varsayılan: tur+eng)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render çözünürlüğü (Varsayılan: 300)")
    parser.add_argument("--spiral-margin", type=float, default=0.12, help="Kenar spiral arama oranı (Varsayılan: 0.12)")
    parser.add_argument("--x-valley", type=float, default=0.07, help="Çift sütun oluk eşiği (Varsayılan: 0.07)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = XYCutOCRConfig(
        lang=args.lang,
        spiral_margin_ratio_x=args.spiral_margin,
        x_valley_threshold=args.x_valley
    )
    engine = BatchMarkdownOCR(cfg)

    in_path = Path(args.input)
    out_path = Path(args.output)
    dbg_base = Path(args.debug_dir) if args.debug_dir else None

    if not in_path.exists():
        print(f"[HATA] Girdi yolu bulunamadı: {in_path}", file=sys.stderr)
        return 1

    if in_path.is_file() and in_path.suffix.lower() == ".pdf":
        md_file = out_path if out_path.suffix.lower() == ".md" else (out_path / f"{in_path.stem}.md")
        res = engine.export_single_target_to_md(in_path, md_file, dbg_base, pdf_dpi=args.dpi)
        if res:
            print(f"[OK] Oluşturuldu: {res}")
        return 0

    if in_path.is_dir():
        out_dir = out_path.parent if out_path.suffix.lower() == ".md" else out_path
        out_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0

        for pdf_file in sorted(in_path.rglob("*.pdf")):
            target_md = out_dir / f"{pdf_file.stem}.md"
            sub_dbg = (dbg_base / pdf_file.stem) if dbg_base else None
            if engine.export_single_target_to_md(pdf_file, target_md, sub_dbg, pdf_dpi=args.dpi):
                print(f"[OK] PDF İşlendi -> {target_md}")
                processed_count += 1

        candidate_dirs = [in_path] + [d for d in sorted(in_path.rglob("*")) if d.is_dir()]
        for img_dir in candidate_dirs:
            has_images = any(p.suffix.lower() in BatchMarkdownOCR.VALID_IMAGE_EXTS for p in img_dir.iterdir() if p.is_file())
            if has_images:
                md_name = f"{img_dir.name}_notlar.md" if img_dir != in_path else "sayfa_goruntuleri.md"
                target_md = out_dir / md_name
                sub_dbg = (dbg_base / img_dir.name) if dbg_base else None
                if engine.export_single_target_to_md(img_dir, target_md, sub_dbg, pdf_dpi=args.dpi):
                    print(f"[OK] Görüntü Klasörü İşlendi -> {target_md}")
                    processed_count += 1

        if processed_count == 0:
            print("[BİLGİ] İşlenecek PDF veya sayfa görüntüsü bulunamadı.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
