"""
WEBSITE EDUKASI DAN DETEKSI STUNTING
Streamlit Application - Full Implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Import custom modules
from z_score_calculator import WHOZScoreCalculator
from knn_model_trainer import StuntingKNNModel

# =====================================================
# CONFIGURATION
# =====================================================

# Page config
st.set_page_config(
    page_title="Edukasi & Deteksi Stunting",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Palettes
COLORS_LIGHT = {
    'primary': '#8FC0A9',
    'secondary': '#68B0AB',
    'accent': "#F0F0F0",  # Light grey for sidebar
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#DC2626',
    'text_dark': '#1A202C',
    'text_light': '#2D3748',
    'background': '#F8FAFB',
    'card_bg': '#FFFFFF',
    'border': '#E2E8F0'
}

COLORS_DARK = {
    'primary': '#4FD1C5',
    'secondary': '#38B2AC',
    'accent': '#2D3748',
    'success': '#48BB78',
    'warning': '#ED8936',
    'danger': '#F56565',
    'text_dark': '#F7FAFC',
    'text_light': '#E2E8F0',
    'background': '#1A202C',
    'card_bg': '#2D3748',
    'border': '#4A5568'
}

# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Get active color palette
COLORS = COLORS_LIGHT if st.session_state.theme == 'light' else COLORS_DARK

# Custom CSS
st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap')
    
    /* Global Styles */
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Main Background */
    .main {{
        background-color: {COLORS['background']} !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }}
    
    .stApp {{
        background-color: {COLORS['background']} !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }}
    
    /* Remove default bottom spacing */
    .block-container {{
        padding-bottom: 1rem !important;
    }}
    
    /* Ensure footer is at bottom with no space */
    .app-footer {{
        margin-bottom: 0 !important;
        padding-bottom: 1rem !important;
    }}
    
    /* Hero Section - MUST BE BEFORE general text rules */
    .hero-section {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white !important;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }}
    
    .hero-section, .hero-section * {{
        color: white !important;
    }}
    
    .hero-section p, .hero-section h1, .hero-section h2, .hero-section span {{
        color: white !important;
    }}
    
    .hero-title {{
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        color: white !important;
    }}
    
    .hero-subtitle {{
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.95;
        color: white !important;
    }}
    
    /* Force text colors - General rules */
    .main p, .main li, .main span, .main div:not(.hero-section):not(.hero-section *), .main label, .main a {{
        color: {COLORS['text_light']} !important;
    }}
    
    .main h1:not(.hero-title), .main h2, .main h3, .main h4, .main h5, .main h6 {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Override Streamlit default text colors */
    .stMarkdown:not(.hero-section *), .stMarkdown p:not(.hero-section *), .stMarkdown span:not(.hero-section *) {{
        color: {COLORS['text_light']} !important;
    }}
    
    .stMarkdown h1:not(.hero-title), .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Info Cards */
    .info-card {{
        background: {COLORS['card_bg']} !important;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
        border-left: 5px solid {COLORS['primary']};
        height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }}
    
    .info-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }}
    
    .card-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
    }}
    
    .card-title {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {COLORS['text_dark']} !important;
        margin-bottom: 1rem;
    }}
    
    .card-text {{
        font-size: 1rem;
        line-height: 1.6;
        color: {COLORS['text_light']} !important;
    }}
    
    .info-card p, .info-card span, .info-card li, .info-card strong {{
        color: {COLORS['text_light']} !important;
    }}
    
    .info-card h1, .info-card h2, .info-card h3, .info-card h4, .info-card h5, .info-card h6 {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Timeline */
    .timeline-container {{
        background: {COLORS['card_bg']} !important;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 2rem 0;
    }}
    
    .timeline-item {{
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid {COLORS['primary']};
        background-color: {COLORS['card_bg']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 10px;
    }}
    
    .timeline-title {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {COLORS['text_dark']} !important;
        margin-bottom: 0.5rem;
    }}
    
    .timeline-content {{
        font-size: 1rem;
        line-height: 1.6;
        color: {COLORS['text_light']} !important;
    }}
    
    .timeline-item p, 
    .timeline-item li, 
    .timeline-item span, 
    .timeline-item strong,
    .timeline-item ul, 
    .timeline-item ol {{
        color: {COLORS['text_light']} !important;
    }}
    
    .timeline-container p, .timeline-container span {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Alert Boxes */
    .alert-box {{
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }}
    
    .alert-box p, .alert-box span, .alert-box li, .alert-box h4, .alert-box strong {{
        color: inherit !important;
    }}
    
    /* Result card - pastikan text terbaca di dark mode */
    .result-card p, .result-card span, .result-card li, .result-card strong {{
        color: {COLORS['text_light']} !important;
    }}
    
    .result-card h1, .result-card h2, .result-card h3, .result-card h4 {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Kesimpulan/Conclusion section */
    .alert-box.alert-warning p,
    .alert-box.alert-warning span,
    .alert-box.alert-warning strong {{
        color: #92400E !important;
    }}
    
    .alert-success {{
        background-color: #D1FAE5;
        border-left: 5px solid {COLORS['success']};
        color: #065F46;
    }}
    
    .alert-warning {{
        background-color: #FEF3C7;
        border-left: 5px solid {COLORS['warning']};
        color: #92400E;
    }}
    
    .alert-danger {{
        background-color: #FEE2E2;
        border-left: 5px solid {COLORS['danger']};
        color: #991B1B;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white !important;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }}
    
    /* Sidebar buttons - force white text */
    [data-testid="stSidebar"] .stButton > button {{
        color: white !important;
    }}
    
    [data-testid="stSidebar"] .stButton > button p {{
        color: white !important;
    }}
    
    /* Form Container */
    .form-container {{
        background: {COLORS['card_bg']} !important;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }}
    
    /* Result Card */
    .result-card {{
        background: {COLORS['card_bg']} !important;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }}
    
    /* Metric Cards */
    .metric-container {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 2rem 0;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['accent']};
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        font-size: 1.1rem;
        padding: 1rem 2rem;
        background-color: {COLORS['card_bg']} !important;
        border-radius: 10px 10px 0 0;
        color: {COLORS['text_dark']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
        color: white !important;
    }}
    
    /* Detection page header - always white text */
    div[style*="background: linear-gradient"] h1,
    div[style*="background: linear-gradient"] p,
    div[style*="background: linear-gradient"] h1 *,
    div[style*="background: linear-gradient"] p * {{
        color: white !important;
    }}
    
    /* Force white text in gradient backgrounds */
    [style*="linear-gradient"] h1,
    [style*="linear-gradient"] p {{
        color: white !important;
    }}
    
    /* Detection header - always white */
    .detection-header {{
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }}
    
    .detection-header h1,
    .detection-header p,
    .detection-header span,
    .detection-header * {{
        color: white !important;
    }}
    
    /* Override any conflicting rules for detection header */
    .main .detection-header h1,
    .main .detection-header p,
    .stMarkdown .detection-header h1,
    .stMarkdown .detection-header p {{
        color: white !important;
    }}
    
    /* Input fields - dynamic based on theme */
    .stTextInput input, .stNumberInput input, .stSelectbox select {{
        background-color: {COLORS['card_bg']} !important;
        color: {COLORS['text_dark']} !important;
        border: 2px solid {COLORS['border']} !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }}
    
    .stTextInput input::placeholder {{
        color: {COLORS['text_light']} !important;
        opacity: 0.5 !important;
    }}
    
    /* Form labels - use text_dark for visibility in both themes */
    .stTextInput label,
    .stNumberInput label,
    .stDateInput label,
    .stSelectbox label {{
        color: {COLORS['text_dark']} !important;
        font-weight: 500 !important;
    }}
    
    /* Radio buttons - use text_dark for visibility */
    .stRadio label,
    .stRadio > label,
    .stRadio div[role="radiogroup"] label {{
        color: {COLORS['text_dark']} !important;
        font-weight: 500 !important;
    }}
    
    .stRadio label span,
    .stRadio div[role="radiogroup"] label span {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Radio button text override */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] p {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Date input */
    .stDateInput input {{
        background-color: {COLORS['card_bg']} !important;
        color: {COLORS['text_light']} !important;
        border: 2px solid {COLORS['border']} !important;
    }}
    
    /* Sidebar override */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['accent']} !important;
    }}
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Info boxes and expanders */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {{
        background-color: {COLORS['card_bg']} !important;
        color: {COLORS['text_light']} !important;
    }}
    
    .stAlert *, .stInfo *, .stSuccess *, .stWarning *, .stError * {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* Streamlit info/success/warning/error divs */
    div[data-testid="stNotificationContentInfo"],
    div[data-testid="stNotificationContentSuccess"],
    div[data-testid="stNotificationContentWarning"],
    div[data-testid="stNotificationContentError"] {{
        background-color: {COLORS['card_bg']} !important;
    }}
    
    div[data-testid="stNotificationContentInfo"] *,
    div[data-testid="stNotificationContentSuccess"] *,
    div[data-testid="stNotificationContentWarning"] *,
    div[data-testid="stNotificationContentError"] * {{
        color: {COLORS['text_light']} !important;
    }}
    
    /* =============================================
       FIX: EXPANDER STYLE (SAFE MODE)
       ============================================= */
    
    /* 1. Container Utama Expander */
    .stExpander {{
        background-color: {COLORS['card_bg']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        margin-bottom: 1rem !important;
        overflow: hidden !important; /* Mencegah konten bocor keluar radius */
    }}

    /* 2. Header (Bagian Judul yang Diklik) */
    .stExpander > details > summary {{
        background-color: {COLORS['card_bg']} !important;
        border: none !important;
        color: {COLORS['text_dark']} !important;
        padding-left: 1rem !important;
        transition: color 0.3s ease !important;
    }}

    /* Efek Hover pada Header */
    .stExpander > details > summary:hover {{
        color: {COLORS['primary']} !important;
        background-color: rgba(0,0,0,0.02) !important; /* Sedikit gelap saat hover */
    }}

    /* 3. Memaksa Warna Teks Judul Expander */
    .stExpander > details > summary p,
    .stExpander > details > summary span {{
        color: inherit !important; /* Mengikuti warna parent (summary) */
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}

    /* 4. Memaksa Warna Ikon Panah (Chevron) */
    .stExpander > details > summary svg {{
        fill: {COLORS['text_dark']} !important;
        color: {COLORS['text_dark']} !important;
    }}
    
    .stExpander > details > summary:hover svg {{
        fill: {COLORS['primary']} !important;
        color: {COLORS['primary']} !important;
    }}

    /* 5. Konten di Dalam Expander */
    div[data-testid="stExpanderDetails"] {{
        background-color: {COLORS['card_bg']} !important;
        border-top: 1px solid {COLORS['border']} !important;
        padding: 1.5rem !important;
    }}

    /* Memastikan teks di dalam konten terbaca */
    div[data-testid="stExpanderDetails"] p,
    div[data-testid="stExpanderDetails"] li,
    div[data-testid="stExpanderDetails"] span {{
        color: {COLORS['text_light']} !important;
    }}

    /* ============================================= */
    
    /* Custom Theme Toggle Button */
    button[kind="secondary"][data-testid="baseButton-secondary"]#theme_toggle {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
        border: none !important;
        border-radius: 30px !important;
        height: 40px !important;
        font-size: 20px !important;
        padding: 0 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    }}
    
    button[kind="secondary"][data-testid="baseButton-secondary"]#theme_toggle:hover {{
        transform: scale(1.05) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
    }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():
    """Load WHO Z-Score Calculator and KNN Model"""
    zscore_calc = WHOZScoreCalculator()
    
    # Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data_balita.csv")
    model_path = os.path.join(current_dir, "models")
    
    knn_model = StuntingKNNModel(data_path)
    try:
        knn_model.load_model(model_path)
        # Model loaded successfully (silently)
    except Exception as e:
        st.warning(f"⚠️ Model KNN tidak tersedia. Pastikan model sudah di-training terlebih dahulu.")
        return zscore_calc, None
    
    return zscore_calc, knn_model

zscore_calculator, knn_model = load_models()

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def create_gauge_chart(value, title, color):
    """Create gauge chart for visualization"""
    # Use theme-appropriate text colors
    text_color = COLORS['text_dark'] if st.session_state.theme == 'light' else COLORS['text_light']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'weight': 600, 'color': text_color}},
        number = {'font': {'color': text_color}},
        gauge = {
            'axis': {'range': [-4, 4], 'tickwidth': 1, 'tickcolor': text_color, 'tickfont': {'color': text_color}},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-4, -3], 'color': '#FEE2E2'},
                {'range': [-3, -2], 'color': '#FEF3C7'},
                {'range': [-2, 4], 'color': '#D1FAE5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color}
    )
    
    return fig

def create_probability_chart(probabilities):
    """Create bar chart for KNN probabilities"""
    # Use theme-appropriate text colors
    text_color = COLORS['text_dark'] if st.session_state.theme == 'light' else COLORS['text_light']
    
    df_prob = pd.DataFrame({
        'Status': list(probabilities.keys()),
        'Probabilitas': [v * 100 for v in probabilities.values()]
    })
    
    # Sort by probability
    df_prob = df_prob.sort_values('Probabilitas', ascending=True)
    
    fig = px.bar(
        df_prob,
        y='Status',
        x='Probabilitas',
        orientation='h',
        title='Distribusi Probabilitas Model KNN',
        color='Probabilitas',
        color_continuous_scale=['#FEE2E2', '#FEF3C7', '#D1FAE5'],
        text='Probabilitas'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont={'color': text_color})
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Probabilitas (%)",
        yaxis_title="Status Gizi",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color},
        title_font={'color': text_color},
        xaxis={'tickfont': {'color': text_color}, 'title': {'font': {'color': text_color}}},
        yaxis={'tickfont': {'color': text_color}, 'title': {'font': {'color': text_color}}}
    )
    
    return fig

def generate_pdf_report(result):
    """Generate PDF report for detection results"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#4A7C59'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#68B0AB'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Title
    elements.append(Paragraph("LAPORAN HASIL DETEKSI STUNTING", title_style))
    elements.append(Spacer(1, 0.5*cm))
    
    # Timestamp
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M")
    elements.append(Paragraph(f"<i>Tanggal Pemeriksaan: {timestamp}</i>", normal_style))
    elements.append(Spacer(1, 0.5*cm))
    
    # Data Anak
    elements.append(Paragraph("1. DATA ANAK", heading_style))
    
    data_anak = [
        ['Nama', result['child_name']],
        ['Umur', f"{result['age_months']} bulan ({result['age_months'] // 12} tahun {result['age_months'] % 12} bulan)"],
        ['Jenis Kelamin', result['gender'].title()],
        ['Tinggi Badan', f"{result['height_cm']} cm"],
        ['Berat Badan', f"{result['weight_kg']} kg"]
    ]
    
    table_data = Table(data_anak, colWidths=[5*cm, 10*cm])
    table_data.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#8FC0A9')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    elements.append(table_data)
    elements.append(Spacer(1, 0.7*cm))
    
    # WHO Z-Score Analysis
    elements.append(Paragraph("2. HASIL ANALISIS WHO Z-SCORE", heading_style))
    
    zscore_data = [
        ['Status Gizi', result['who_status']],
        ['Z-Score', f"{result['zscore']:.2f}"],
        ['Kategori', result['who_recommendation']['title']]
    ]
    
    table_zscore = Table(zscore_data, colWidths=[5*cm, 10*cm])
    table_zscore.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#8FC0A9')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    elements.append(table_zscore)
    elements.append(Spacer(1, 0.3*cm))
    
    # Interpretation
    elements.append(Paragraph("<b>Interpretasi:</b>", normal_style))
    elements.append(Paragraph(result['who_interpretation'], normal_style))
    elements.append(Spacer(1, 0.5*cm))
    
    # KNN Analysis (if available)
    if result['knn_result']:
        elements.append(Paragraph("3. HASIL ANALISIS MODEL MACHINE LEARNING (KNN)", heading_style))
        
        knn_data = result['knn_result']
        risk_info = result['risk_interpretation']
        
        knn_table_data = [
            ['Prediksi Status', knn_data['prediction'].title()],
            ['Tingkat Risiko', risk_info['level']],
            ['Persentase Risiko', f"{knn_data['risk_percentage']}%"]
        ]
        
        table_knn = Table(knn_table_data, colWidths=[5*cm, 10*cm])
        table_knn.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#8FC0A9')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        elements.append(table_knn)
        elements.append(Spacer(1, 0.3*cm))
        elements.append(Paragraph("<b>Interpretasi Model KNN:</b>", normal_style))
        elements.append(Paragraph(risk_info['message'], normal_style))
        elements.append(Spacer(1, 0.5*cm))
    
    # Recommendations
    reco = result['who_recommendation']
    elements.append(Paragraph("4. REKOMENDASI TINDAKAN", heading_style))
    elements.append(Paragraph(f"<b>{reco['title']}</b>", normal_style))
    elements.append(Spacer(1, 0.3*cm))
    
    elements.append(Paragraph("<b>Langkah-langkah yang perlu dilakukan:</b>", normal_style))
    
    for idx, action in enumerate(reco['actions'], 1):
        elements.append(Paragraph(f"{idx}. {action}", normal_style))
    
    elements.append(Spacer(1, 0.5*cm))
    
    # Disclaimer
    elements.append(Spacer(1, 1*cm))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_JUSTIFY
    )
    
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> Hasil analisis ini merupakan screening awal dan bukan diagnosis medis. "
        "Untuk diagnosis yang akurat dan penanganan yang tepat, konsultasikan dengan tenaga kesehatan profesional "
        "seperti dokter anak, bidan, atau petugas kesehatan di Puskesmas/Posyandu terdekat.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# PAGE: DASHBOARD (EDUKASI)
# =====================================================

def render_dashboard():
    """Render halaman dashboard edukasi"""
    
    # Hero Section
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">👶 Kenali, Cegah, dan Atasi Stunting Sejak Dini</h1>
        <p class="hero-subtitle">Desa Sehat, Anak Cerdas, Masa Depan Gemilang</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Cek Status Gizi Anak Sekarang", use_container_width=True, type="primary"):
            st.session_state.page = 'detection'
            st.rerun()
    
    st.markdown("---")
    
    # Section: Apa itu Stunting?
    st.markdown("## 📚 Apa Itu Stunting?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <p class="card-text">
            <strong>Stunting</strong> adalah kondisi gagal tumbuh pada anak balita akibat kekurangan gizi kronis 
            dan infeksi berulang, terutama pada <strong>1.000 Hari Pertama Kehidupan (HPK)</strong>. 
            <br><br>
            Anak tergolong stunting jika panjang atau tinggi badannya berada di bawah standar usianya 
            berdasarkan kurva pertumbuhan WHO.
            <br><br>
            <strong>Periode Kritis:</strong> Dari masa kehamilan hingga anak berusia 2 tahun (730 hari)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "image", "apaitustunting.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.image("https://mangusada.badungkab.go.id/promosi/read/102/stunting", 
                     use_container_width=True)
        st.caption("Ilustrasi perbandingan anak dengan pertumbuhan normal dan stunting")
        st.caption("Sumber: [mangusada.badungkab.go.id](https://mangusada.badungkab.go.id/promosi/read/102/stunting)")
    
    st.markdown("---")
    
    # Section: Ciri-Ciri & Gejala
    st.markdown("## 🔍 Ciri-Ciri & Gejala Stunting")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">📏</div>
            <h3 class="card-title">Pertumbuhan Melambat</h3>
            <p class="card-text">Tinggi badan lebih pendek dari anak seusianya dan pertumbuhan gigi terlambat</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">👦</div>
            <h3 class="card-title">Tampak Lebih Muda</h3>
            <p class="card-text">Wajah terlihat lebih muda dari anak seusianya, namun tubuh tidak berkembang</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">🧠</div>
            <h3 class="card-title">Sulit Fokus</h3>
            <p class="card-text">Kemampuan memori dan belajar kurang baik, serta kurang aktif bermain</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">🤒</div>
            <h3 class="card-title">Mudah Sakit</h3>
            <p class="card-text">Sistem kekebalan tubuh rendah sehingga rentan terkena penyakit infeksi</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section: Dampak Stunting
    st.markdown("## ⚠️ Dampak Stunting")
    
    tab1, tab2 = st.tabs(["📉 Jangka Pendek", "📊 Jangka Panjang"])
    
    with tab1:
        st.markdown("""
        <div class="alert-box alert-warning">
            <h4>Dampak Segera yang Terlihat:</h4>
            <ul>
                <li>🧠 <strong>Terganggunya perkembangan otak</strong>: Perkembangan kognitif tidak optimal</li>
                <li>📚 <strong>Kecerdasan berkurang</strong>: Kemampuan belajar dan daya ingat terhambat</li>
                <li>📏 <strong>Gangguan pertumbuhan fisik</strong>: Tinggi badan tidak sesuai standar usia</li>
                <li>⚡ <strong>Gangguan metabolisme</strong>: Sistem pencernaan dan penyerapan nutrisi terganggu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="alert-box alert-danger">
            <h4>Dampak yang Berkelanjutan hingga Dewasa:</h4>
            <ul>
                <li>🎓 <strong>Menurunnya kemampuan kognitif dan prestasi belajar</strong>: Performa akademik rendah</li>
                <li>💔 <strong>Risiko penyakit degeneratif</strong>: Rentan terkena jantung, diabetes, stroke di usia tua</li>
                <li>⚖️ <strong>Risiko obesitas</strong>: Metabolisme yang terganggu meningkatkan risiko kegemukan</li>
                <li>💼 <strong>Produktivitas ekonomi menurun</strong>: Kemampuan kerja dan penghasilan terbatas</li>
                <li>🔄 <strong>Siklus kemiskinan</strong>: Stunting dapat diturunkan ke generasi berikutnya</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section: Pencegahan (General)
    st.markdown("## 🛡️ Pencegahan Stunting")
    
    col1, col2, col3, col4 = st.columns([1, 1.1, 0.8, 1.1])
    
    prevention_steps = [
        {
            'icon': '🚰',
            'title': 'Sanitasi Bersih',
            'description': 'Akses air bersih dan jamban sehat. Stop BABS (Buang Air Besar Sembarangan)'
        },
        {
            'icon': '🍽️',
            'title': 'Nutrisi Seimbang',
            'description': 'Pola makan gizi seimbang dengan prinsip "Isi Piringku" dan protein hewani      '
        },
        {
            'icon': '👨‍👩‍👧',
            'title': 'Pola Asuh',
            'description': 'Orang tua yang paham kesehatan, gizi, dan stimulasi tumbuh kembang anak'
        },
        {
            'icon': '💉',
            'title': 'Imunisasi Lengkap',
            'description': 'Melengkapi imunisasi dasar di Posyandu sesuai jadwal pemerintah          '
        }
    ]
    
    for idx, (col, step) in enumerate(zip([col1, col2, col3, col4], prevention_steps)):
        with col:
            st.markdown(f"""
            <div class="info-card">
                <div class="card-icon">{step['icon']}</div>
                <h3 class="card-title">{step['title']}</h3>
                <p class="card-text">{step['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section: 1000 HPK (HIGHLIGHT)
    st.markdown("## ⭐ 1000 Hari Pertama Kehidupan (HPK)")
    st.markdown("### Periode EMAS untuk Cegah Stunting")
    
    st.markdown("""
    <div class="timeline-container">
        <p style="font-size: 1.1rem; text-align: center; margin-bottom: 2rem; color: #555;">
            <strong>1000 HPK</strong> adalah periode paling kritis dalam kehidupan anak yang dimulai sejak 
            <strong>masa kehamilan (270 hari)</strong> hingga anak berusia <strong>2 tahun (730 hari)</strong>.
            <br>Pada periode ini, otak dan tubuh anak berkembang sangat pesat!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Timeline visualization
    timeline_data = [
        {
            'phase': '🤰 Masa Kehamilan (270 Hari)',
            'duration': '9 Bulan',
            'actions': [
                'Ibu mengonsumsi makanan bergizi lengkap (4 sehat 5 sempurna)',
                'Minum Tablet Tambah Darah (TTD) minimal 90 tablet selama kehamilan',
                'Periksa kehamilan (ANC) minimal 6 kali ke Bidan/Puskesmas',
                'Hindari merokok, alkohol, dan stres berlebihan'
            ]
        },
        {
            'phase': '👶 Lahir s.d. 6 Bulan (180 Hari)',
            'duration': '0-6 Bulan',
            'actions': [
                'Lakukan Inisiasi Menyusu Dini (IMD) segera setelah lahir',
                'Berikan ASI Eksklusif tanpa tambahan apapun (termasuk air putih)',
                'Pastikan bayi menyusu 8-12 kali sehari',
                'Pantau tumbuh kembang di Posyandu setiap bulan',
                'Berikan imunisasi dasar lengkap sesuai jadwal'
            ]
        },
        {
            'phase': '🍼 Usia 6-24 Bulan (550 Hari)',
            'duration': '6 Bulan - 2 Tahun',
            'actions': [
                'Teruskan pemberian ASI hingga anak berusia 2 tahun',
                'Berikan MPASI bertahap: mulai makanan lumat → lunak → padat',
                'Pastikan ada protein hewani SETIAP HARI (telur/ikan/ayam/daging)',
                'Berikan makanan beragam dengan gizi seimbang',
                'Stimulasi perkembangan: ajak bicara, bermain, dan bersosialisasi',
                'Pantau pertumbuhan rutin di Posyandu'
            ]
        }
    ]
    
    for item in timeline_data:
        with st.expander(f"**{item['phase']}** - Durasi: {item['duration']}", expanded=True):
            st.markdown(f"""
            <div class="timeline-item">
                <p class="timeline-title">Hal yang Harus Dilakukan:</p>
                <div class="timeline-content">
                    <ul>
                        {''.join([f'<li>✅ {action}</li>' for action in item['actions']])}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section: Pengobatan & Penanganan
    st.markdown("## 🏥 Jika Anak Terindikasi Stunting")
    
    st.markdown("""
    <div class="alert-box alert-warning">
        <h4>⚠️ Langkah-Langkah Penanganan:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Intervensi Gizi Segera
        - Pemberian Makanan Tambahan (PMT) tinggi protein dan kalori
        - Dilakukan di bawah pengawasan Puskesmas/Bidan
        - Fokus pada protein hewani: telur, ikan, daging
        - Pantau progress setiap minggu
        
        ### 2️⃣ Stimulasi Intensif
        - Melatih motorik kasar (merangkak, berjalan, berlari)
        - Melatih motorik halus (memegang, menulis, menggambar)
        - Bermain edukatif sesuai usia
        - Interaksi sosial dengan teman sebaya
        """)
    
    with col2:
        st.markdown("""
        ### 3️⃣ Periksa Penyakit Penyerta
        - Obati TBC, cacingan, atau infeksi lain
        - Atasi diare kronis yang menghambat penyerapan gizi
        - Pemeriksaan kesehatan menyeluruh
        - Suplemen vitamin jika diperlukan
        
        ### 4️⃣ Rujukan Medis
        - Segera konsultasi ke Dokter Spesialis Anak
        - Bawa hasil pengukuran rutin dari Posyandu
        - Follow-up treatment secara berkala
        - Jangan tunda penanganan medis
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("## 📞 Informasi & Kontak")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🏥 Puskesmas Terdekat**\n\nKonsultasikan kondisi anak Anda secara berkala")
    
    with col2:
        st.info("**👶 Posyandu**\n\nTimbang dan ukur anak setiap bulan")
    
    with col3:
        st.info("**📱 Hotline Kesehatan**\n\nHubungi (Halo Kemenkes): \nTelepon: (Kode Lokal) 1500-567 \nSMS: 0812-8156-2620")
    
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #888; font-size: 0.9rem;">
        💡 <strong>Website ini adalah alat screening awal.</strong> Untuk diagnosis dan penanganan yang tepat, 
        selalu konsultasikan dengan tenaga medis profesional.
    </p>
    """, unsafe_allow_html=True)

# =====================================================
# PAGE: DETEKSI STUNTING
# =====================================================

def render_detection():
    """Render halaman deteksi stunting"""
    
    # Header
    header_gradient = f"linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%)"
    st.markdown(f"""
    <div class="detection-header" style="background: {header_gradient};">
        <h1 style="margin: 0;">🔍 Status Gizi & Deteksi Dini Stunting</h1>
        <p style="opacity: 0.9; margin-top: 0.5rem;">
            Analisis status gizi anak berdasarkan standar WHO dan Model Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Masukkan Data Anak")
    
    col1, col2 = st.columns(2)
    
    with col1:
        child_name = st.text_input("Nama Anak (Opsional)", placeholder="Contoh: Ahmad")
        gender = st.radio(
            "Jenis Kelamin",
            options=['laki-laki', 'perempuan'],
            horizontal=True,
            help="Pilih jenis kelamin anak"
        )
        
        # Date picker for birth date
        today = datetime.now()
        min_date = today - timedelta(days=25*365)  # Maksimal 25 tahun ke belakang
        max_date = today
        
        birth_date = st.date_input(
            "Tanggal Lahir",
            value=today - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            help="Pilih tanggal lahir"
        )
        
        # Calculate age in months
        age_months = int((today - datetime.combine(birth_date, datetime.min.time())).days / 30)
        st.info(f"📅 Umur anak: **{age_months} bulan** ({age_months // 12} tahun {age_months % 12} bulan)")
    
    with col2:
        height_cm = st.number_input(
            "Tinggi Badan (cm)",
            min_value=40.0,
            max_value=200.0,
            value=75.0,
            step=0.1,
            help="Masukkan tinggi badan dalam centimeter"
        )
        
        weight_kg = st.number_input(
            "Berat Badan (kg) - Opsional",
            min_value=2.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help="Masukkan berat badan anak (tidak digunakan dalam analisis, hanya untuk catatan)"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔬 Analisis Sekarang",
            use_container_width=True,
            type="primary"
        )
    
    # Disclaimer
    st.markdown("""
    <div class="alert-box alert-warning" style="margin-top: 2rem;">
        <strong>⚠️ Disclaimer:</strong> Website ini adalah alat screening awal dan bukan keputusan final. 
        Hasil analisis harus dikonsultasikan dengan tenaga medis terpercaya (Bidan/Dokter/Puskesmas) 
        untuk diagnosis dan penanganan yang tepat.
    </div>
    """, unsafe_allow_html=True)
    
    # =====================================================
    # ANALYSIS & RESULTS
    # =====================================================
    
    if analyze_button:
        # Validation
        if age_months < 0:
            st.error("❌ Umur tidak valid. Tanggal lahir tidak boleh di masa depan.")
            return
        
        if height_cm < 40 or height_cm > 200:
            st.error("❌ Tinggi badan tidak valid. Harap masukkan nilai antara 40-200 cm.")
            return
        
        # Perform analysis
        with st.spinner('🔄 Menganalisis data...'):
            
            # 1. WHO Z-Score Analysis
            zscore, is_adult = zscore_calculator.calculate_zscore(age_months, height_cm, gender)
            who_status = zscore_calculator.classify_nutrition_status(zscore, is_adult)
            who_interpretation = zscore_calculator.get_zscore_interpretation(zscore, is_adult)
            who_recommendation = zscore_calculator.get_recommendation(zscore, who_status)
            
            # Warning untuk usia dewasa
            if is_adult:
                st.warning("⚠️ **Catatan**: Usia di atas 19 tahun. Standar WHO Z-Score dirancang untuk anak dan remaja (0-19 tahun). Hasil ini menggunakan referensi standar akhir (19 tahun) sebagai perbandingan. Stunting yang terdeteksi pada dewasa menunjukkan kemungkinan gangguan pertumbuhan di masa kecil.")
            
            # 2. KNN Model Prediction (if available)
            knn_result = None
            if knn_model is not None:
                try:
                    knn_result = knn_model.predict(age_months, gender, height_cm)
                    risk_interpretation = knn_model.get_risk_interpretation(knn_result['risk_percentage'])
                except Exception as e:
                    st.warning(f"⚠️ Model KNN error: {str(e)}")
            
            # Save to session state
            st.session_state.detection_result = {
                'child_name': child_name if child_name else "Anak",
                'age_months': age_months,
                'gender': gender,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'zscore': zscore,
                'is_adult': is_adult,
                'who_status': who_status,
                'who_interpretation': who_interpretation,
                'who_recommendation': who_recommendation,
                'knn_result': knn_result,
                'risk_interpretation': risk_interpretation if knn_result else None
            }
        
        st.success("✅ Analisis selesai!")
    
    # Display results
    if st.session_state.detection_result:
        result = st.session_state.detection_result
        
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis")
        
        # Summary Card
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); color: white;">
            <h3 style="margin-top: 0;">Data Anak</h3>
            <p><strong>Nama:</strong> {result['child_name']}</p>
            <p><strong>Umur:</strong> {result['age_months']} bulan ({result['age_months'] // 12} tahun {result['age_months'] % 12} bulan)</p>
            <p><strong>Jenis Kelamin:</strong> {result['gender']}</p>
            <p><strong>Tinggi Badan:</strong> {result['height_cm']} cm</p>
            <p><strong>Berat Badan:</strong> {result['weight_kg']} kg</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Analisis WHO Z-Score", "🤖 Analisis Model KNN", "💡 Rekomendasi"])
        
        # TAB 1: WHO Z-Score Analysis
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Hasil Z-Score WHO")
                
                # Status badge
                status_color = result['who_recommendation']['color']
                st.markdown(f"""
                <div style="background-color: {status_color}20; padding: 1.5rem; border-radius: 10px; 
                            border-left: 5px solid {status_color}; margin: 1rem 0;">
                    <h2 style="margin: 0; color: {status_color};">{result['who_status']}</h2>
                    <p style="font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0;">
                        Z-Score: {result['zscore']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation
                st.markdown(f"""
                <div class="alert-box" style="background-color: {status_color}20; border-left: 5px solid {status_color};">
                    <h4>Interpretasi:</h4>
                    <p>{result['who_interpretation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Reference ranges
                with st.expander("📚 Referensi Kategori Z-Score WHO"):
                    st.markdown("""
                    - **Severely Stunted (Sangat Pendek):** Z-score < -3 SD
                    - **Stunted (Pendek):** -3 SD ≤ Z-score < -2 SD
                    - **Normal:** -2 SD ≤ Z-score ≤ +3 SD
                    - **Tall (Tinggi):** Z-score > +3 SD
                    
                    *SD = Standard Deviation (Simpangan Baku)*
                    """)
            
            with col2:
                # Gauge chart
                gauge_color = status_color
                fig_gauge = create_gauge_chart(result['zscore'], "Z-Score Height-for-Age", gauge_color)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Explanation
                st.info("""
                **Apa itu Z-Score?**
                
                Z-Score adalah nilai statistik yang menunjukkan seberapa jauh tinggi badan anak 
                dari nilai rata-rata (median) untuk umur dan jenis kelaminnya berdasarkan standar WHO.
                
                - Z-Score **0** = Tepat di nilai rata-rata
                - Z-Score **positif** = Di atas rata-rata
                - Z-Score **negatif** = Di bawah rata-rata
                """)
        
        # TAB 2: KNN Model Analysis
        with tab2:
            if result['knn_result']:
                knn_data = result['knn_result']
                risk_info = result['risk_interpretation']
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Prediksi Model Machine Learning (KNN)")
                    
                    # Risk percentage display
                    risk_color = risk_info['color']
                    st.markdown(f"""
                    <div style="background-color: {risk_color}20; padding: 1.5rem; border-radius: 10px; 
                                border-left: 5px solid {risk_color}; margin: 1rem 0;">
                        <h3 style="margin: 0;">Tingkat Risiko: {risk_info['level']}</h3>
                        <p style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0; color: {risk_color};">
                            {knn_data['risk_percentage']}%
                        </p>
                        <p style="margin: 0; font-size: 0.9rem;">Persentase Risiko Stunting</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction
                    st.markdown(f"""
                    <div class="alert-box" style="background-color: {risk_color}20; border-left: 5px solid {risk_color};">
                        <h4>Prediksi Status:</h4>
                        <p style="font-size: 1.2rem; font-weight: 600;">{knn_data['prediction'].title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interpretation
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>💬 Interpretasi Model KNN:</h4>
                        <p>{risk_info['message']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Probability distribution chart
                    fig_prob = create_probability_chart(knn_data['probabilities'])
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Model explanation
                    st.info("""
                    **Bagaimana Model KNN Bekerja?**
                    
                    Model K-Nearest Neighbors (KNN) membandingkan data anak Anda dengan ribuan data 
                    anak lain dalam database training. Model mencari anak-anak dengan karakteristik 
                    serupa (umur, tinggi, jenis kelamin) dan menghitung persentase risiko berdasarkan 
                    status gizi mereka.
                    
                    **Persentase risiko** menunjukkan seberapa mirip karakteristik anak Anda dengan 
                    anak-anak yang mengalami stunting dalam dataset.
                    """)
                
                # Detailed probabilities
                with st.expander("📊 Detail Probabilitas Setiap Kategori"):
                    prob_df = pd.DataFrame({
                        'Status Gizi': list(knn_data['probabilities'].keys()),
                        'Probabilitas (%)': [f"{v*100:.2f}%" for v in knn_data['probabilities'].values()]
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            else:
                st.warning("⚠️ Model KNN tidak tersedia. Pastikan model sudah di-training terlebih dahulu.")
        
        # TAB 3: Recommendations
        with tab3:
            st.markdown("### 💡 Rekomendasi Tindakan")
            
            reco = result['who_recommendation']
            
            # Title with color
            st.markdown(f"""
            <div style="background-color: {reco['color']}20; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid {reco['color']}; margin: 1rem 0;">
                <h3 style="margin: 0; color: {reco['color']};">{reco['title']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Actions list
            st.markdown("#### 📋 Langkah-Langkah yang Perlu Dilakukan:")
            
            for idx, action in enumerate(reco['actions'], 1):
                st.markdown(f"""
                <div class="info-card" style="margin: 0.5rem 0;">
                    <p><strong>{idx}.</strong> {action}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional info based on status
            if "Stunted" in result['who_status']:
                st.markdown("---")
                st.markdown("#### 🏥 Rujukan dan Bantuan")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Tempat Konsultasi:**
                    - 🏥 Puskesmas terdekat
                    - 👶 Posyandu desa
                    - 🩺 Dokter Spesialis Anak
                    - 👩‍⚕️ Bidan Desa
                    """)
                
                with col2:
                    st.markdown("""
                    **Program Bantuan:**
                    - 🍲 PMT (Pemberian Makanan Tambahan)
                    - 💊 Suplementasi vitamin dan mineral
                    - 📚 Edukasi gizi keluarga
                    - 🏃 Stimulasi tumbuh kembang
                    """)
            
            # Print/Download button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                pdf_buffer = generate_pdf_report(result)
                st.download_button(
                    label="📄 Download Laporan PDF",
                    data=pdf_buffer,
                    file_name=f"Laporan_Stunting_{result['child_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # Additional comparison with KNN
        if result['knn_result']:
            st.markdown("---")
            st.markdown("## 🔄 Perbandingan Hasil Analisis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid {result['who_recommendation']['color']};">
                    <h4>📈 Metode WHO Z-Score</h4>
                    <p><strong>Status:</strong> {result['who_status']}</p>
                    <p><strong>Z-Score:</strong> {result['zscore']}</p>
                    <p style="font-size: 0.9rem; color: #666;">
                        Berdasarkan standar antropometri WHO yang membandingkan tinggi badan 
                        anak dengan median populasi referensi global.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid {result['risk_interpretation']['color']};">
                    <h4>🤖 Model Machine Learning KNN</h4>
                    <p><strong>Prediksi:</strong> {result['knn_result']['prediction'].title()}</p>
                    <p><strong>Risiko:</strong> {result['knn_result']['risk_percentage']}% ({result['risk_interpretation']['level']})</p>
                    <p style="font-size: 0.9rem; color: #666;">
                        Berdasarkan perbandingan dengan ribuan data anak dalam dataset training 
                        yang memiliki karakteristik serupa.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Conclusion
            st.markdown("""
            <div class="alert-box alert-warning">
                <h4>🎯 Catatan:</h4>
                <p>
                Kedua metode analisis digunakan untuk memberikan gambaran yang lebih komprehensif. 
                <strong>Metode WHO Z-Score</strong> adalah standar internasional yang diakui secara medis, 
                sedangkan <strong>Model KNN</strong> memberikan perspektif tambahan berdasarkan data populasi lokal.
                </p>
                <p>
                <strong>Penting:</strong> Hasil ini adalah screening awal. Untuk diagnosis pasti dan 
                penanganan yang tepat, selalu konsultasikan dengan tenaga kesehatan profesional.
                </p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# MAIN APPLICATION
# =====================================================

def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        # Logo with center alignment
        # Get the directory where app.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, "image", "logo.jpg")
        if os.path.exists(logo_path):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image(logo_path, use_container_width=True)
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image("https://via.placeholder.com/200x100/8FC0A9/FFFFFF?text=Logo+PKM", use_container_width=True)
                
        # Theme Toggle Button (styled as toggle switch)
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            # Button dengan icon yang berubah sesuai theme
            current_icon = '🌙' if st.session_state.theme == 'dark' else '☀️'
            if st.button(current_icon, key='theme_toggle', use_container_width=True, type='secondary'):
                st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
                st.rerun()
                
        st.markdown("### 📍 Navigasi")
        
        if st.button("🏠 Dashboard", use_container_width=True, 
                     type="primary" if st.session_state.page == 'dashboard' else "secondary"):
            st.session_state.page = 'dashboard'
            st.rerun()
        
        if st.button("🔍 Deteksi Stunting", use_container_width=True,
                     type="primary" if st.session_state.page == 'detection' else "secondary"):
            st.session_state.page = 'detection'
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### ℹ️ Tentang")
        st.markdown("""
        Website ini dibuat untuk membantu deteksi dini stunting pada balita menggunakan:
        
        - **Standar WHO** (Z-Score)
        - **Machine Learning** (KNN Model)
        
        **Versi:** 1.0  
        **Tahun:** 2025
        """)
        
        st.markdown("---")
        
        st.markdown("### 📞 Kontak")
        st.markdown("""
        **Puskesmas:**  
        📍 Puskesmas Bojong Nangka\n
        ☎️ (021) 5476423
        
        **Halo Kemenkes:**  
        ☎️ Telepon: (Kode Lokal) 1500-567\n
        ✉️ SMS: 0812-8156-2620
        """)
    
    # Route to appropriate page
    if st.session_state.page == 'dashboard':
        render_dashboard()
    elif st.session_state.page == 'detection':
        render_detection()

# Run the application
if __name__ == "__main__":
    main()