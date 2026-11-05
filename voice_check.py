import re
import subprocess
import sys

BANNED = ["delve", "crucial", "pivotal", "vital", "tapestry", "landscape", "realm", "multifaceted", "nuanced",
          "robust", "seamless", "leverage", "foster", "underscore", "shed light", "testament", "navigate",
          "ever-evolving", "in today's world", "important to note", "worth noting", "in conclusion", "in summary",
          "plays a key role", "a wide range of", "not only", "moreover", "furthermore", "additionally",
          "utilize", "demonstrate", "facilitate"]

pdf = sys.argv[1]
text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
body = text.split("References")[0]
body = re.sub(r"-\n\s*", "", body)
flat = re.sub(r"\s+", " ", body)

problems = []
for ch, name in [("—", "em dash"), ("–", "en dash"), (";", "semicolon"), ("…", "ellipsis"), ("...", "ellipsis"), ("!", "exclamation")]:
    for m in re.finditer(re.escape(ch), flat):
        problems.append(f"{name}: ...{flat[max(0, m.start() - 40):m.end() + 40]}...")
for m in re.finditer(r"[A-Za-z]: [A-Za-z]", flat):
    problems.append(f"colon: ...{flat[max(0, m.start() - 40):m.end() + 40]}...")
for m in re.finditer(r"\([^)]*[A-Za-z]{3,}[^)]*\)", flat):
    problems.append(f"parentheses: {m.group(0)[:80]}")
low = flat.lower()
for w in BANNED:
    for m in re.finditer(r"\b" + re.escape(w) + r"\b", low):
        problems.append(f"banned '{w}': ...{flat[max(0, m.start() - 40):m.end() + 40]}...")
for m in re.finditer(r"\?", flat):
    problems.append(f"question mark: ...{flat[max(0, m.start() - 60):m.end() + 10]}...")

print(f"{len(flat.split())} words before references")
print("\n".join(problems) if problems else "no problems found")
