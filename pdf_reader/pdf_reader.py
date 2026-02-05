import os
import re
import json
import csv
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from pypdf import PdfReader, PdfWriter


# =========================
# Regexes (easy to tweak)
# =========================

# Robust ORDER ID: "ORDER 293047", "Order #293047", "ORDER:293047", etc.
ORDER_ID_RE = re.compile(r"\bORDER\b\s*[:#]?\s*(\d{3,})\b", re.IGNORECASE)

# Size pattern examples:
# "M4 | W6", "M 4|W 6", "M4-W6", "M4 / W6"
SIZE_PATTERN_RE = re.compile(
    r"\bM\s*\d{1,2}\s*(?:\|\s*W|[-/]\s*W)\s*\d{1,2}\b", re.IGNORECASE
)

# SKU extraction: looks for lines that contain "SKU" then grabs the most plausible token.
# You may tweak these patterns for your PDFs.
SKU_LINE_RE = re.compile(r"\bSKU\b", re.IGNORECASE)

# Matches tokens like: SLG1528-12, SKT0001-10, SCS1030, KSX0001, FG-SGWP102
SKU_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}[A-Z0-9]{2,10}(?:-[A-Z0-9]{2,10})?\b")



# =========================
# Data models
# =========================

@dataclass
class PageData:
    page_index: int
    text_ok: bool
    order_id: Optional[str]
    product_names: List[str] = field(default_factory=list)
    skus: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class OrderGroup:
    group_key: str  # unique key even if order_id missing
    order_id: Optional[str]
    order_id_num: Optional[int]
    page_indices: List[int]
    pages: List[PageData]
    assigned_aisle: str = "Unknown"
    route_rank: int = 10**9
    match_methods: List[str] = field(default_factory=list)  # e.g., ["contains", "sku_prefix"]


# =========================
# Config loading
# =========================
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("aisle_route", ["Unknown"])
    cfg.setdefault("sku_contains", {})  # NEW

    # route rank: aisle -> walking order
    cfg["route_rank"] = {aisle: i for i, aisle in enumerate(cfg["aisle_route"])}
    if "Unknown" not in cfg["route_rank"]:
        cfg["route_rank"]["Unknown"] = len(cfg["route_rank"]) + 9999

    # Normalize sku_contains to uppercase and store (prefix, aisle, note)
    norm = []
    for prefix, obj in cfg["sku_contains"].items():
        if not prefix:
            continue
        prefix_u = str(prefix).strip().upper()
        aisle = obj.get("aisle", "Unknown") if isinstance(obj, dict) else str(obj)
        note = obj.get("note", "") if isinstance(obj, dict) else ""
        norm.append((prefix_u, aisle, note))

    # Most-specific first: longer prefixes win
    norm.sort(key=lambda x: len(x[0]), reverse=True)
    cfg["sku_contains_norm"] = norm

    return cfg


def normalize_text(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# PDF extraction
# =========================

def extract_order_id(page_text: str) -> Optional[str]:
    if not page_text:
        return None
    m = ORDER_ID_RE.search(page_text)
    return m.group(1) if m else None


def extract_product_names_from_text(page_text: str) -> List[str]:
    """
    Extract product-name candidates from a page.

    Problem this solves:
      Many PDFs put the size (M# | W#) on its own line, and the product name
      appears on the line ABOVE (or split across lines). The older approach
      grabbed the size line itself, causing lots of Unknown matches.

    Strategy:
      1) Find lines that contain a size token.
      2) For each size line:
         - If the line contains meaningful text besides size -> take it.
         - Else (it's basically just size) -> take the closest product-like line above.
      3) If no size lines found, fallback to scanning for product-like lines.

    Returns raw candidate product lines (NOT normalized here).
    """
    if not page_text:
        return []

    # Split into trimmed non-empty lines
    lines = [ln.strip() for ln in page_text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Helper: remove the size token from a line (does NOT uppercase)
    def _strip_size(s: str) -> str:
        s = re.sub(SIZE_PATTERN_RE, "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.strip(" -|/")

    # Helper: decide if a line looks like a product name vs header/metadata
    def _is_product_like(s: str) -> bool:
        if not s:
            return False
        up = s.upper()

        # Filter out common non-product lines
        bad_contains = [
            "ORDER", "SUBTOTAL", "TOTAL", "SHIP", "BILL", "DATE", "QTY", "PRICE",
            "AMOUNT", "PAYMENT", "DISCOUNT", "TAX", "CUSTOMER", "ADDRESS"
        ]
        if any(b in up for b in bad_contains):
            return False

        # If it's too short, it's probably not a product name
        if len(s) < 6:
            return False

        # Product lines often contain separators like '-' or '.'
        # (e.g., "LOGAN PUFF - BLACKDOPE", "SKTP.01 - BLACK")
        if (" - " in s) or ("SKTP." in up) or ("." in s) or ("MULE" in up) or ("STOMPER" in up) or ("PUFF" in up):
            return True

        # As a fallback, allow lines with at least 2 words
        if len(s.split()) >= 2:
            return True

        return False

    candidates: List[str] = []

    # 1) Use size lines as anchors
    for i, ln in enumerate(lines):
        if not SIZE_PATTERN_RE.search(ln):
            continue

        # Remove size token from that same line to see if product text exists there
        stripped = _strip_size(ln)

        # If after removing size we still have meaningful content, use it
        # Example: "LOGAN PUFF - BLACKDOPE - M4 | W6" -> "LOGAN PUFF - BLACKDOPE"
        if _is_product_like(stripped):
            candidates.append(stripped)
            continue

        # Otherwise, size line is probably just "M4 | W6" (or similar).
        # Look ABOVE for the closest product-like line.
        j = i - 1
        while j >= 0:
            above = lines[j].strip()
            above_stripped = _strip_size(above)

            if _is_product_like(above_stripped):
                candidates.append(above_stripped)
                break

            # Stop climbing if we hit another size line (likely another item block)
            if SIZE_PATTERN_RE.search(above):
                break

            j -= 1

    # 2) Fallback: if we found nothing using size anchors, scan page for product-like lines
    if not candidates:
        for ln in lines:
            ln2 = _strip_size(ln)
            if _is_product_like(ln2):
                candidates.append(ln2)

    # 3) Deduplicate while preserving order
    out: List[str] = []
    seen = set()
    for c in candidates:
        c2 = re.sub(r"\s+", " ", c).strip()
        if c2 and c2 not in seen:
            seen.add(c2)
            out.append(c2)

    return out



def extract_skus_from_text(page_text: str) -> List[str]:
    """
    Extract SKU-like tokens anywhere on the page.
    Works even if the PDF doesn't label them with 'SKU:'.
    """
    if not page_text:
        return []

    tokens = SKU_TOKEN_RE.findall(page_text.upper())

    # Filter out obvious non-SKUs if needed (optional)
    # Example: remove words that match the pattern accidentally.
    blacklist = {"ORDER", "COLOR", "UNITED", "STATES", "TEL"}
    cleaned = [t for t in tokens if t not in blacklist]

    # Dedup preserve order
    out, seen = [], set()
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out



def extract_page_data(reader: PdfReader, page_index: int) -> PageData:
    pd = PageData(page_index=page_index, text_ok=True, order_id=None)

    try:
        page = reader.pages[page_index]
        txt = page.extract_text()
        if not txt or not txt.strip():
            pd.text_ok = False
            pd.errors.append("Empty text extracted")
            return pd

        pd.order_id = extract_order_id(txt)
        pd.product_names = extract_product_names_from_text(txt)
        pd.skus = extract_skus_from_text(txt)

        return pd

    except Exception as e:
        pd.text_ok = False
        pd.errors.append(f"Extraction failed: {type(e).__name__}: {e}")
        return pd


# =========================
# Matching + sorting
# =========================

def match_item_to_aisle(
    cfg: Dict[str, Any],
    product_name: Optional[str],
    sku: Optional[str],
) -> Tuple[str, str]:
    """
    Returns (aisle, method) where method in:
      - "exact"
      - "contains"
      - "sku_prefix"
      - "unknown"
    """
    product_norm = normalize_text(product_name or "")
    sku_norm = normalize_text(sku or "")

    # a) exact match
    if product_norm and product_norm in cfg.get("_exact_matches_norm", {}):
        return cfg["_exact_matches_norm"][product_norm], "exact"

    # b) contains rules
    for rule in cfg.get("_contains_rules_norm", []):
        needle = rule.get("contains", "")
        if needle and product_norm and needle in product_norm:
            return rule.get("aisle", "Unknown"), "contains"

    # c) sku prefix rules
    for rule in cfg.get("_sku_prefix_rules_norm", []):
        pref = rule.get("prefix", "")
        if pref and sku_norm and sku_norm.startswith(pref):
            return rule.get("aisle", "Unknown"), "sku_prefix"

    return "Unknown", "unknown"


def aisle_rank(cfg: Dict[str, Any], aisle: str) -> int:
    return cfg.get("_route_rank", {}).get(aisle, cfg.get("_route_rank", {}).get("Unknown", 10**9))


def match_sku_to_aisle(cfg: Dict[str, Any], sku: str) -> Tuple[str, str]:
    """
    Returns (aisle, match_detail)
    match_detail includes the prefix used (helpful for CSV/debug).
    """
    sku_u = (sku or "").strip().upper()
    if not sku_u:
        return "Unknown", ""

    for pref, aisle, _note in cfg.get("sku_contains_norm", []):
        if sku_u.startswith(pref):
            return aisle, pref

    return "Unknown", ""

def assign_aisle_to_order(config: Dict[str, Any], order) -> None:
    """
    Assign order to the earliest aisle in the route among all matched SKUs.
    """
    best_aisle = "Unknown"
    best_rank = config["route_rank"].get("Unknown", 10**9)
    used_prefixes = []

    for page in order.pages:
        for sku in (page.skus or []):
            aisle, pref = match_sku_to_aisle(config, sku)
            rank = config["route_rank"].get(aisle, config["route_rank"].get("Unknown", 10**9))
            if rank < best_rank:
                best_rank = rank
                best_aisle = aisle
            if aisle != "Unknown" and pref:
                used_prefixes.append(pref)

    order.assigned_aisle = best_aisle
    order.route_rank = best_rank
    order.match_methods = ["sku_contains:" + ",".join(sorted(set(used_prefixes)))] if used_prefixes else ["unknown"]



def group_pages_into_orders(pages: List[PageData]) -> List[OrderGroup]:
    """
    Group pages into orders while preserving original PDF page order.

    NEW BEHAVIOR (fixes continuation pages):
      - If a page has an order_id -> start/continue that order group.
      - If a page has NO order_id -> treat it as a continuation page and attach it
        to the most recent order group seen in the PDF (the page right before it).
      - If the PDF starts with pages that have no order_id (rare), those pages
        become their own standalone groups.

    This ensures "page 2" stays attached to the order on "page 1".
    """

    groups: Dict[str, OrderGroup] = {}
    ordered_keys: List[str] = []  # preserves group appearance order

    last_group_key: Optional[str] = None  # the order group we last attached a page to

    for pd in pages:
        # Case 1: This page has a real order_id => group by that ID
        if pd.order_id:
            key = f"ORDER_{pd.order_id}"

            if key not in groups:
                oid_num = int(pd.order_id) if pd.order_id.isdigit() else None
                groups[key] = OrderGroup(
                    group_key=key,
                    order_id=pd.order_id,
                    order_id_num=oid_num,
                    page_indices=[],
                    pages=[],
                )
                ordered_keys.append(key)

            groups[key].pages.append(pd)
            last_group_key = key
            continue

        # Case 2: No order_id => continuation page
        # Attach to the most recent order group, if any.
        if last_group_key is not None:
            groups[last_group_key].pages.append(pd)
            continue

        # Case 3: No order_id and no prior group exists (PDF starts with missing header pages)
        # Make a standalone group so we don't lose it.
        key = f"NOORDER_BLOCK_START_{pd.page_index}"
        if key not in groups:
            groups[key] = OrderGroup(
                group_key=key,
                order_id=None,
                order_id_num=None,
                page_indices=[],
                pages=[],
            )
            ordered_keys.append(key)

        groups[key].pages.append(pd)
        last_group_key = key

    # Finalize: ensure each group's pages are in original page order
    out: List[OrderGroup] = []
    for k in ordered_keys:
        g = groups[k]
        g.pages.sort(key=lambda x: x.page_index)
        g.page_indices = [p.page_index for p in g.pages]
        out.append(g)

    return out



def sort_orders(cfg: Dict[str, Any], orders: List[OrderGroup]) -> List[OrderGroup]:
    """
    Sorting should be primarily by (aisle_route_rank, order_id numeric).
    If missing order_id_num, keep stable relative to each other by original first page.
    """
    def sort_key(o: OrderGroup):
        oid_num = o.order_id_num if o.order_id_num is not None else 10**18
        first_page = o.page_indices[0] if o.page_indices else 10**9
        return (o.route_rank, oid_num, first_page)

    return sorted(orders, key=sort_key)


# =========================
# Output writers
# =========================

def write_pdf(reader: PdfReader, sorted_orders: List[OrderGroup], output_pdf_path: str) -> None:
    writer = PdfWriter()
    for order in sorted_orders:
        for idx in order.page_indices:
            writer.add_page(reader.pages[idx])

    with open(output_pdf_path, "wb") as f:
        writer.write(f)


def write_report_csv(sorted_orders: List[OrderGroup], output_csv_path: str) -> None:
    headers = [
        "order_id",
        "pages_in_order",
        "assigned_aisle",
        "route_rank",
        "original_page_indices",
        "extracted_product_names",
        "extracted_skus",
        "match_methods",
        "page_extraction_errors",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for o in sorted_orders:
            all_products: List[str] = []
            all_skus: List[str] = []
            all_errors: List[str] = []

            for pg in o.pages:
                all_products.extend(pg.product_names or [])
                all_skus.extend(pg.skus or [])
                if pg.errors:
                    all_errors.extend([f"p{pg.page_index}: {e}" for e in pg.errors])

            # Dedup for readability
            def dedup(seq: List[str]) -> List[str]:
                seen = set()
                out = []
                for s in seq:
                    s2 = s.strip()
                    if s2 and s2 not in seen:
                        seen.add(s2)
                        out.append(s2)
                return out

            all_products = dedup(all_products)
            all_skus = dedup(all_skus)
            all_errors = dedup(all_errors)

            w.writerow([
                o.order_id or "",
                len(o.page_indices),
                o.assigned_aisle,
                o.route_rank,
                ";".join(str(i) for i in o.page_indices),
                " | ".join(all_products),
                ";".join(all_skus),
                ";".join(o.match_methods) if o.match_methods else "unknown",
                " || ".join(all_errors),
            ])


# =========================
# Main pipeline
# =========================

def run_sort_pipeline(pdf_path: str, config_path: str, status_cb) -> Tuple[str, str]:
    """
    Returns (output_pdf_path, output_csv_path)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cfg = load_config(config_path)

    status_cb(f"Loading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)

    status_cb(f"PDF pages: {len(reader.pages)}")
    pages: List[PageData] = []

    # Extract page-by-page
    for i in range(len(reader.pages)):
        status_cb(f"Extracting page {i+1}/{len(reader.pages)} ...")
        pd = extract_page_data(reader, i)
        pages.append(pd)

    # Group by order_id (critical constraint)
    status_cb("Grouping pages by order_id ...")
    orders = group_pages_into_orders(pages)

    # Assign aisle to each order
    status_cb(f"Assigning aisles to {len(orders)} orders/groups ...")
    for idx, order in enumerate(orders, start=1):
        assign_aisle_to_order(cfg, order)
        if idx % 20 == 0 or idx == len(orders):
            status_cb(f"Assigned aisles: {idx}/{len(orders)}")

    # Sort orders
    status_cb("Sorting orders by route rank then order_id ...")
    sorted_orders = sort_orders(cfg, orders)

    # Output paths
    in_dir = os.path.dirname(os.path.abspath(pdf_path))
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_pdf_path = os.path.join(in_dir, f"SORTED_{base}.pdf")
    output_csv_path = os.path.join(in_dir, f"SORTED_{base}_report.csv")

    # Write outputs
    status_cb(f"Writing output PDF: {output_pdf_path}")
    write_pdf(reader, sorted_orders, output_pdf_path)

    status_cb(f"Writing CSV report: {output_csv_path}")
    write_report_csv(sorted_orders, output_csv_path)

    status_cb("Done.")
    return output_pdf_path, output_csv_path


# =========================
# Tkinter UI
# =========================

class OrderSorterUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PDF Order Sorter (Aisle Route)")

        self.pdf_path_var = tk.StringVar(value="")
        self.config_path_var = tk.StringVar(value=os.path.join(os.getcwd(), "config.json"))

        # Top controls
        frm = tk.Frame(root, padx=10, pady=10)
        frm.pack(fill="x")

        btn_pdf = tk.Button(frm, text="Select PDF", command=self.select_pdf, width=14)
        btn_pdf.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.lbl_pdf = tk.Label(frm, textvariable=self.pdf_path_var, anchor="w")
        self.lbl_pdf.grid(row=0, column=1, padx=5, pady=5, sticky="we")

        btn_cfg = tk.Button(frm, text="Select Config", command=self.select_config, width=14)
        btn_cfg.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.lbl_cfg = tk.Label(frm, textvariable=self.config_path_var, anchor="w")
        self.lbl_cfg.grid(row=1, column=1, padx=5, pady=5, sticky="we")

        self.btn_run = tk.Button(frm, text="Run Sort", command=self.run_sort, width=14)
        self.btn_run.grid(row=2, column=0, padx=5, pady=10, sticky="w")

        frm.columnconfigure(1, weight=1)

        # Status/progress text
        self.status_box = ScrolledText(root, height=16, padx=10, pady=10)
        self.status_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log("Ready. Select a PDF and (optionally) a config.json, then click Run Sort.")

        # Thread-safe UI updates
        self._lock = threading.Lock()
        self._running = False

    def log(self, msg: str):
        self.status_box.insert("end", msg + "\n")
        self.status_box.see("end")
        self.root.update_idletasks()

    def select_pdf(self):
        path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if path:
            self.pdf_path_var.set(path)
            self.log(f"Selected PDF: {path}")

    def select_config(self):
        path = filedialog.askopenfilename(
            title="Select config.json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if path:
            self.config_path_var.set(path)
            self.log(f"Selected Config: {path}")

    def set_running(self, running: bool):
        with self._lock:
            self._running = running
        self.btn_run.config(state=("disabled" if running else "normal"))

    def run_sort(self):
        with self._lock:
            if self._running:
                return

        pdf_path = self.pdf_path_var.get().strip()
        config_path = self.config_path_var.get().strip()

        if not pdf_path:
            messagebox.showwarning("Missing PDF", "Please select a PDF file first.")
            return

        # If user didn't pick a config, default to ./config.json
        if not config_path:
            config_path = os.path.join(os.getcwd(), "config.json")
            self.config_path_var.set(config_path)

        self.set_running(True)
        self.log("Starting sort...")

        def worker():
            try:
                out_pdf, out_csv = run_sort_pipeline(
                    pdf_path=pdf_path,
                    config_path=config_path,
                    status_cb=lambda m: self.root.after(0, self.log, m),
                )
                def done_ok():
                    self.log(f"SUCCESS!\nOutput PDF: {out_pdf}\nReport CSV: {out_csv}")
                    messagebox.showinfo("Success", f"Sorted PDF created:\n{out_pdf}\n\nReport CSV:\n{out_csv}")
                    self.set_running(False)

                self.root.after(0, done_ok)

            except Exception as e:
                tb = traceback.format_exc()

                def done_err():
                    self.log("ERROR:\n" + str(e))
                    self.log(tb)
                    messagebox.showerror("Error", f"{e}")
                    self.set_running(False)

                self.root.after(0, done_err)

        threading.Thread(target=worker, daemon=True).start()


def run_ui():
    root = tk.Tk()
    app = OrderSorterUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_ui()
