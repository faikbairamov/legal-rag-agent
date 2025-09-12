import fitz
import re

# Input and output paths
input_pdf = "./data/raw/matsne-31702-134.pdf"
cleaned_txt = "./data/processed/matsne-31702-134.txt"

# ---------- Step 1: Extract text from PDF ----------
doc = fitz.open(input_pdf)
all_text = ""
for page in doc:
    all_text += page.get_text("text") + "\n"

# ---------- Step 2: Clean text ----------
# Convert superscript numbers into dotted notation (e.g. 49¹ -> 49.1)
SUPERSCRIPT_MAP = str.maketrans({
    "¹": ".1", "²": ".2", "³": ".3", "⁴": ".4", "⁵": ".5",
    "⁶": ".6", "⁷": ".7", "⁸": ".8", "⁹": ".9", "⁰": ".0",
})
all_text = all_text.translate(SUPERSCRIPT_MAP)

# Remove repetitive Matsne footer lines
MATSNE_RE = re.compile(
    r"http://www\.matsne\.gov\.ge\s*040\.000\.000\.05\.001\.000\.223",
    re.MULTILINE
)
all_text = MATSNE_RE.sub("", all_text)

# Optional: Remove soft hyphens so words don’t break
SOFT_HYPHENS = ("\u00ad", "\u2010")
for sh in SOFT_HYPHENS:
    all_text = all_text.replace(sh, "")

# Save cleaned version
with open(cleaned_txt, "w", encoding="utf-8") as f:
    f.write(all_text)

print("Cleaned text saved to:", cleaned_txt)
