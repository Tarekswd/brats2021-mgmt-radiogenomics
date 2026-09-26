"""
Convert Task 2 research report to PDF using headless Chrome/Edge.
Author: Tarek Al Ahmadieh (Sole Author)
"""
import subprocess
from pathlib import Path
import markdown

def convert_task2_pdf():
    t2_root = Path("c:/Users/Admin/Desktop/brats2021/brats2021_research/brats2021_task2")
    md_path = t2_root / "research_report.md"
    html_path = t2_root / "research_report.html"
    pdf_path = t2_root / "research_report.pdf"
    desktop_pdf = Path("C:/Users/Admin/Desktop/BraTS2021_Task2_MGMT_Research_Report_Tarek_Al_Ahmadieh.pdf")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc'])
    
    # Fix image links
    import re
    def fix_img(match):
        rel = match.group(1)
        abs_uri = (t2_root / rel).resolve().as_uri()
        return f'src="{abs_uri}"'
        
    html_body = re.sub(r'src="(results/figures/[^"]+)"', fix_img, html_body)
    
    styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{ size: A4; margin: 20mm 15mm; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; color: #1e293b; line-height: 1.55; font-size: 10.5pt; }}
h1 {{ color: #0f172a; font-size: 20pt; border-bottom: 2px solid #2563eb; padding-bottom: 6px; margin-bottom: 8px; }}
h2 {{ color: #1e3a8a; font-size: 13pt; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; margin-top: 22px; margin-bottom: 8px; }}
h3 {{ color: #1e40af; font-size: 11pt; margin-top: 14px; margin-bottom: 4px; }}
p {{ margin-bottom: 8px; text-align: justify; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 9pt; }}
th, td {{ padding: 6px 8px; border: 1px solid #cbd5e1; text-align: left; }}
th {{ background: #f1f5f9; color: #0f172a; font-weight: 600; }}
tr:nth-child(even) {{ background: #f8fafc; }}
img {{ max-width: 90%; height: auto; display: block; margin: 12px auto 4px auto; border: 1px solid #e2e8f0; border-radius: 4px; }}
em {{ display: block; text-align: center; font-size: 8.5pt; color: #64748b; margin-bottom: 12px; }}
code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 3px; font-size: 8.5pt; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
        
    chrome_path = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if not chrome_path.exists():
        chrome_path = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        
    cmd = [
        str(chrome_path),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        html_path.as_uri()
    ]
    subprocess.run(cmd, capture_output=True)
    
    if pdf_path.exists():
        import shutil
        shutil.copy2(pdf_path, desktop_pdf)
        print(f"Task 2 PDF report successfully created at: {pdf_path}")
        print(f"Desktop copy created at: {desktop_pdf}")

if __name__ == "__main__":
    convert_task2_pdf()
