import os
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#4a5568"))
        
        # Draw header on all pages except the first page
        if self._pageNumber > 1:
            self.drawString(54, 750, "Rapport de Recherche : Re-ID & Suivi Persistant de Vehicules")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Draw footer on all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(54, 36, "Projet Persistent Vehicle Tracking & Re-ID")
        self.drawRightString(558, 36, f"Page {self._pageNumber} / {page_count}")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_pdf():
    # Setup directories
    workspace = Path("/home/youssef/Desktop/vehicles-tracking-id")
    docs_dir = workspace / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = docs_dir / "rapport_technique_pipeline.pdf"
    
    # Read the markdown report from workspace docs directory
    md_path = docs_dir / "rapport_technique_pipeline.md"
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define Styles
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1a365d")
    secondary_color = colors.HexColor("#2b6cb0")
    text_color = colors.HexColor("#2d3748")
    bg_color = colors.HexColor("#f7fafc")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=15,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=text_color,
        spaceAfter=7
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )
    
    math_style = ParagraphStyle(
        'MathEq',
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1a202c"),
        alignment=1,
        spaceBefore=6,
        spaceAfter=6,
        backColor=bg_color,
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=0.5,
        borderPadding=5
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        parent=body_style,
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#2d3748"),
        backColor=bg_color,
        borderColor=colors.HexColor("#cbd5e0"),
        borderWidth=0.5,
        borderPadding=5,
        spaceAfter=8
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    story = []
    
    # Pre-parse sections
    lines = content.split('\n')
    i = 0
    in_code_block = False
    code_content = []
    
    # Embeddable images
    results_img_path = workspace / "outputs/results/results.png"
    pr_img_path = workspace / "outputs/results/BoxPR_curve.png"
    matrix_img_path = workspace / "outputs/results/confusion_matrix.png"
    labels_img_path = workspace / "outputs/results/labels.jpg"

    story.append(Paragraph("Rapport de Recherche Technique : Pipeline de Re-Identification Persistante de Vehicules", title_style))
    story.append(Paragraph("Systeme Modulaire de Re-identification Persistante et Suivi Multi-Objets en Entrepot et Autoroute", ParagraphStyle('Sub', parent=body_style, fontSize=10, leading=14, textColor=secondary_color, alignment=1)))
    story.append(Spacer(1, 15))
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Code block handling
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_content)
                story.append(Paragraph(code_text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"), code_style))
                code_content = []
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            code_content.append(lines[i])
            i += 1
            continue
            
        # Ignore empty spacer lines
        if not line:
            i += 1
            continue
            
        # Headers
        if line.startswith("# "):
            # We skip the main title since we drew a custom header
            i += 1
            continue
        elif line.startswith("## "):
            text = line[3:]
            story.append(Paragraph(text, h1_style))
        elif line.startswith("### "):
            text = line[4:]
            story.append(Paragraph(text, h2_style))
        elif line.startswith("#### "):
            text = line[5:]
            story.append(Paragraph(text, h2_style))
            
        # Math blocks (centered and styled)
        elif line.startswith("$$") and line.endswith("$$"):
            eq = line[2:-2].strip()
            # Clean display for pdf:
            eq_clean = eq.replace("\\parallel", " || ").replace("\\|", " || ").replace("\\dots", "...").replace("\\_\\text{stripe}\\_", "_stripe").replace("\\_\\text{nouveau}\\_", "_new").replace("\\mathbf{g}", "g").replace("\\alpha", "α").replace("\\hat{f}", "f_hat").replace("\\|f_{final}\\|_2", "||f_{final}||_2").replace("\\ge", ">=").replace("\\le", "<=").replace("\\sigma^*", "sigma*").replace("\\arg\\min", "argmin").replace("_{new}", "_new").replace("_{id}", "_id").replace("_{final}", "_final").replace("_{stripe_1}", "_stripe1").replace("_{stripe_2}", "_stripe2").replace("_{stripe_K}", "_stripeK").replace("_{ij}", "_ij").replace("_{i,\\sigma(i)}", "_i,sigma(i)").replace("_{new}", "_new")
            story.append(Paragraph(eq_clean, math_style))
            
        # Inline math conversion for standard lines
        elif "$$" in line:
            eq_matches = re.findall(r"\$\$(.*?)\$\$", line)
            for eq in eq_matches:
                eq_clean = eq.replace("\\parallel", " || ").replace("\\|", " || ")
                line = line.replace(f"$${eq}$$", f"<b>{eq_clean}</b>")
            story.append(Paragraph(line, body_style))
            
        # Standard List item
        elif line.startswith("- ") or line.startswith("* "):
            text = line[2:]
            # Clean inline maths in list items
            text = re.sub(r"\$(.*?)\$", r"<i>\1</i>", text)
            text = text.replace("\\text", "")
            story.append(Paragraph(f"&bull; {text}", bullet_style))
        elif re.match(r"^\d+\.\s", line):
            m = re.match(r"^(\d+)\.\s(.*)", line)
            num = m.group(1)
            text = m.group(2)
            text = re.sub(r"\$(.*?)\$", r"<i>\1</i>", text)
            story.append(Paragraph(f"{num}. {text}", bullet_style))
            
        # Embed plots at exact placeholder spots
        elif "results.png" in line:
            if results_img_path.exists():
                story.append(Image(str(results_img_path), width=5.5*inch, height=2.75*inch))
                story.append(Spacer(1, 3))
                story.append(Paragraph("<i>Figure 1 : Evolution des fonctions de perte et progression du mAP.</i>", ParagraphStyle('Cap', parent=body_style, alignment=1, fontSize=7.5)))
        elif "BoxPR_curve.png" in line:
            if pr_img_path.exists():
                story.append(Image(str(pr_img_path), width=4.0*inch, height=3.0*inch))
                story.append(Spacer(1, 3))
                story.append(Paragraph("<i>Figure 2 : Courbes de precision par rapport au rappel.</i>", ParagraphStyle('Cap', parent=body_style, alignment=1, fontSize=7.5)))
        elif "confusion_matrix.png" in line:
            if matrix_img_path.exists():
                story.append(Image(str(matrix_img_path), width=4.0*inch, height=3.0*inch))
                story.append(Spacer(1, 3))
                story.append(Paragraph("<i>Figure 3 : Matrice de confusion normalisee.</i>", ParagraphStyle('Cap', parent=body_style, alignment=1, fontSize=7.5)))
        elif "labels.jpg" in line:
            if labels_img_path.exists():
                story.append(Image(str(labels_img_path), width=4.5*inch, height=3.4*inch))
                story.append(Spacer(1, 3))
                story.append(Paragraph("<i>Figure 4 : Distribution spatiale des annotations du jeu de donnees.</i>", ParagraphStyle('Cap', parent=body_style, alignment=1, fontSize=7.5)))
                
        # Skip table placeholders and bibliography header lines since we build a customized clean flowable
        elif line.startswith("|") or line.startswith("---") or "mermaids" in line:
            i += 1
            continue
            
        # Standard paragraph
        else:
            # Inline math substitutions
            line = re.sub(r"\$(.*?)\$", r"<i>\1</i>", line)
            line = line.replace("\\text", "")
            story.append(Paragraph(line, body_style))
            
        i += 1
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF SUCCESSFULLY GENERATED at docs/rapport_technique_pipeline.pdf")

if __name__ == "__main__":
    build_pdf()
