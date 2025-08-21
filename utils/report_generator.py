import datetime
import io
import os
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF


# Comprehensive tumor information database
TUMOR_INFO = {
    "Glioma Tumor": {
        "description": "Gliomas are tumors that arise from glial cells in the brain and spinal cord. They are the most common type of primary brain tumor in adults.",
        "types": [
            "Astrocytoma (Grade I-IV)",
            "Oligodendroglioma",
            "Ependymoma",
            "Mixed glioma"
        ],
        "symptoms": [
            "Headaches that gradually worsen",
            "Seizures (especially in adults with no prior history)",
            "Personality or memory changes",
            "Nausea and vomiting",
            "Weakness or paralysis in parts of the body",
            "Vision or speech problems"
        ],
        "treatment_options": [
            "Surgical resection",
            "Radiation therapy",
            "Chemotherapy",
            "Targeted therapy",
            "Clinical trials"
        ],
        "prognosis": "Varies significantly based on grade, location, and molecular characteristics. Grade I gliomas have better prognosis than Grade IV (Glioblastoma).",
        "prevalence": "Approximately 3-5 cases per 100,000 people annually"
    },
    
    "Meningioma Tumor": {
        "description": "Meningiomas are tumors that arise from the meninges, the protective layers surrounding the brain and spinal cord. Most are benign (non-cancerous).",
        "types": [
            "Grade I (Benign) - 90% of cases",
            "Grade II (Atypical) - 7-8% of cases", 
            "Grade III (Malignant) - 1-3% of cases"
        ],
        "symptoms": [
            "Headaches",
            "Vision problems",
            "Hearing loss or ringing in ears",
            "Memory loss",
            "Weakness in arms or legs",
            "Seizures",
            "Changes in smell"
        ],
        "treatment_options": [
            "Observation (for small, asymptomatic tumors)",
            "Surgical removal",
            "Stereotactic radiosurgery",
            "Conventional radiation therapy",
            "Hormone therapy (in some cases)"
        ],
        "prognosis": "Generally excellent for Grade I meningiomas. 5-year survival rate is over 95% for benign meningiomas.",
        "prevalence": "Most common primary brain tumor, representing about 36% of all brain tumors"
    },
    
    "Pituitary Tumor": {
        "description": "Pituitary tumors are growths in the pituitary gland, a small organ that controls several other hormone-producing glands. Most are benign adenomas.",
        "types": [
            "Functional adenomas (hormone-producing)",
            "Non-functional adenomas",
            "Microadenomas (<10mm)",
            "Macroadenomas (>10mm)",
            "Craniopharyngioma",
            "Rathke's cleft cyst"
        ],
        "symptoms": [
            "Vision problems (peripheral vision loss)",
            "Hormonal imbalances",
            "Headaches",
            "Unexplained fatigue",
            "Mood changes or depression",
            "Changes in menstrual periods",
            "Erectile dysfunction",
            "Growth abnormalities"
        ],
        "treatment_options": [
            "Transsphenoidal surgery",
            "Medication (dopamine agonists, somatostatin analogs)",
            "Radiation therapy",
            "Stereotactic radiosurgery",
            "Hormone replacement therapy"
        ],
        "prognosis": "Generally excellent with appropriate treatment. Most pituitary adenomas are curable.",
        "prevalence": "About 15% of all brain tumors. Affects approximately 1 in 1,000 people"
    },
    
    "No Tumor": {
        "description": "No tumor detected in the MRI scan. The brain tissue appears normal based on AI analysis.",
        "normal_findings": [
            "Healthy brain tissue structure",
            "Normal ventricle size and shape",
            "No abnormal masses or lesions",
            "Appropriate gray and white matter contrast",
            "Normal cerebrospinal fluid spaces"
        ],
        "recommendations": [
            "Continue routine medical check-ups",
            "Maintain healthy lifestyle",
            "Monitor for any new symptoms",
            "Follow up with healthcare provider as scheduled"
        ],
        "note": "While no tumor is detected, this AI analysis should be confirmed by a qualified radiologist and medical professional."
    }
}


def create_custom_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.Color(0.2, 0.3, 0.7),
        alignment=1  # Center alignment
    )
    
    # Custom heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.Color(0.3, 0.4, 0.8),
        borderWidth=1,
        borderColor=colors.Color(0.8, 0.8, 0.9),
        borderPadding=8
    )
    
    # Custom body style
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leading=14
    )
    
    return {
        'title': title_style,
        'heading': heading_style,
        'body': body_style,
        'normal': styles['Normal']
    }


def create_logo_section():
    """Create a professional logo section for the PDF header."""
    # Create a drawing for the header with logo placeholder
    drawing = Drawing(400, 80)
    
    # Background rectangle
    drawing.add(Rect(0, 0, 400, 80, fillColor=colors.Color(0.95, 0.95, 0.98), strokeColor=None))
    
    # You can add actual logo here when available
    # For now, we'll create a text-based logo
    return drawing


def generate_pdf_report(original_path, overlay_path, pred_class, confidence):
    """
    Generate comprehensive PDF report containing MRI images, AI prediction, and medical information.
    
    Args:
        original_path (str): Path to original MRI image.
        overlay_path (str): Path to overlay heatmap image.
        pred_class (str): Predicted tumor type.
        confidence (float): Prediction confidence (0-1).
    
    Returns:
        BytesIO: In-memory PDF file buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    story = []
    custom_styles = create_custom_styles()

    # --- Header with Logo Section ---
    story.append(Paragraph("TUMORX", custom_styles['title']))
    story.append(Paragraph("AI-Powered Brain Tumor Detection & Analysis", custom_styles['normal']))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.Color(0.3, 0.4, 0.8)))
    story.append(Spacer(1, 20))

    # --- Report Information ---
    report_info = [
        ["Report Generated:", datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')],
        ["Analysis Method:", "Deep Learning Neural Networks"],
        ["Model Version:", "TumorX v2.1.0"],
        ["Report ID:", f"TX-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"]
    ]
    #["Report ID:", f"TX-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"]

    info_table = Table(report_info, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 30))

    # --- MRI Images Section ---
    story.append(Paragraph("MRI SCAN ANALYSIS", custom_styles['heading']))
    
    # Create image table
    image_data = []
    image_row = []
    
    if original_path and os.path.exists(original_path):
        image_row.append([
            Paragraph("<b>Original MRI Scan</b>", custom_styles['body']),
            RLImage(original_path, width=2.5*inch, height=2.5*inch)
        ])
    
    if overlay_path and os.path.exists(overlay_path):
        image_row.append([
            Paragraph("<b>AI Segmentation Analysis</b>", custom_styles['body']),
            RLImage(overlay_path, width=2.5*inch, height=2.5*inch)
        ])
    
    if image_row:
        image_table = Table([image_row], colWidths=[2.5*inch, 2.5*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 30))

    # --- AI Prediction Results ---
    story.append(Paragraph("AI DIAGNOSTIC RESULTS", custom_styles['heading']))
    
    # Main result table
    confidence_percentage = confidence * 1 if confidence else 0
    result_color = colors.red if pred_class != "No Tumor" else colors.green
    
    result_data = [
        ["Classification Result", pred_class],
        ["Confidence Level", f"{confidence_percentage:.2f}%"],
        ["Risk Assessment", "HIGH PRIORITY - Requires medical attention" if pred_class != "No Tumor" else "NORMAL - No tumor detected"]
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 3.5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.95)),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), "LEFT"),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (1, 2), (1, 2), result_color),
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 30))

    # --- Detailed Medical Information ---
    if pred_class in TUMOR_INFO:
        tumor_data = TUMOR_INFO[pred_class]
        
        story.append(Paragraph("DETAILED MEDICAL INFORMATION", custom_styles['heading']))
        story.append(Paragraph(f"<b>About {pred_class}:</b>", custom_styles['body']))
        story.append(Paragraph(tumor_data['description'], custom_styles['body']))
        story.append(Spacer(1, 15))
        
        if pred_class != "No Tumor":
            # Types/Subtypes
            if 'types' in tumor_data:
                story.append(Paragraph("<b>Common Types/Subtypes:</b>", custom_styles['body']))
                for tumor_type in tumor_data['types']:
                    story.append(Paragraph(f"• {tumor_type}", custom_styles['body']))
                story.append(Spacer(1, 10))
            
            # Symptoms
            if 'symptoms' in tumor_data:
                story.append(Paragraph("<b>Common Symptoms:</b>", custom_styles['body']))
                for symptom in tumor_data['symptoms']:
                    story.append(Paragraph(f"• {symptom}", custom_styles['body']))
                story.append(Spacer(1, 10))
            
            # Treatment Options
            if 'treatment_options' in tumor_data:
                story.append(Paragraph("<b>Treatment Options:</b>", custom_styles['body']))
                for treatment in tumor_data['treatment_options']:
                    story.append(Paragraph(f"• {treatment}", custom_styles['body']))
                story.append(Spacer(1, 10))
            
            # Prognosis
            if 'prognosis' in tumor_data:
                story.append(Paragraph("<b>Prognosis:</b>", custom_styles['body']))
                story.append(Paragraph(tumor_data['prognosis'], custom_styles['body']))
                story.append(Spacer(1, 10))
            
            # Prevalence
            if 'prevalence' in tumor_data:
                story.append(Paragraph("<b>Prevalence:</b>", custom_styles['body']))
                story.append(Paragraph(tumor_data['prevalence'], custom_styles['body']))
                story.append(Spacer(1, 20))
        
        else:  # No Tumor case
            if 'normal_findings' in tumor_data:
                story.append(Paragraph("<b>Normal Findings Detected:</b>", custom_styles['body']))
                for finding in tumor_data['normal_findings']:
                    story.append(Paragraph(f"• {finding}", custom_styles['body']))
                story.append(Spacer(1, 10))
            
            if 'recommendations' in tumor_data:
                story.append(Paragraph("<b>Recommendations:</b>", custom_styles['body']))
                for rec in tumor_data['recommendations']:
                    story.append(Paragraph(f"• {rec}", custom_styles['body']))
                story.append(Spacer(1, 10))

    # --- All Tumor Types Reference ---
    story.append(PageBreak())
    story.append(Paragraph("BRAIN TUMOR REFERENCE GUIDE", custom_styles['heading']))
    story.append(Paragraph("Complete overview of brain tumor types analyzed by TumorX AI system:", custom_styles['body']))
    story.append(Spacer(1, 15))
    
    for tumor_name, tumor_info in TUMOR_INFO.items():
        if tumor_name != "No Tumor":
            story.append(Paragraph(f"<b>{tumor_name}</b>", custom_styles['body']))
            story.append(Paragraph(tumor_info['description'], custom_styles['body']))
            if 'prevalence' in tumor_info:
                story.append(Paragraph(f"<i>Prevalence: {tumor_info['prevalence']}</i>", custom_styles['body']))
            story.append(Spacer(1, 15))

    # --- Medical Disclaimers ---
    story.append(PageBreak())
    story.append(Paragraph("MEDICAL DISCLAIMERS & IMPORTANT INFORMATION", custom_styles['heading']))
    
    disclaimers = [
        "<b>AI Technology Limitations:</b> This analysis is performed by artificial intelligence and machine learning algorithms. While highly accurate, AI systems can make errors and should never replace professional medical judgment.",
        
        "<b>Not a Medical Diagnosis:</b> This report provides AI-assisted analysis for informational purposes only. It does not constitute a medical diagnosis, treatment recommendation, or medical advice.",
        
        "<b>Professional Medical Consultation Required:</b> Any abnormal findings require immediate consultation with qualified medical professionals including radiologists, neurologists, or neurosurgeons.",
        
        "<b>Imaging Limitations:</b> MRI interpretation depends on image quality, patient positioning, contrast usage, and scanning parameters. Some conditions may not be visible on MRI.",
        
        "<b>Emergency Situations:</b> If experiencing severe headaches, seizures, vision changes, or neurological symptoms, seek immediate medical attention regardless of this AI analysis.",
        
        "<b>Second Opinion Recommended:</b> For any positive findings, obtain a second opinion from qualified medical professionals and consider additional diagnostic tests.",
        
        "<b>Data Privacy:</b> Medical imaging data processed by this system is handled according to healthcare privacy regulations. No personal health information is stored permanently.",
        
        "<b>Regulatory Status:</b> This AI system is for research and educational purposes. It is not FDA-approved for clinical diagnosis."
    ]
    
    for disclaimer in disclaimers:
        story.append(Paragraph(disclaimer, custom_styles['body']))
        story.append(Spacer(1, 10))

    # --- Footer Information ---
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))
    
    footer_text = """
    <b>TumorX AI System</b><br/>
    Advanced Brain Tumor Detection Platform<br/>
    Powered by Deep Learning & Computer Vision<br/>
    For research and educational use only<br/>
    <i>Generated on {}</i>
    """.format(datetime.datetime.now().strftime('%B %d, %Y'))
    
    story.append(Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=custom_styles['normal'],
        fontSize=9,
        alignment=1,  # Center alignment
        textColor=colors.grey
    )))

    # --- Build PDF ---
    doc.build(story)
    buffer.seek(0)
    return buffer