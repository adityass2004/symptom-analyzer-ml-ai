import streamlit as st
import requests
import json
import datetime
import os
import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ==================== CONFIGURATION ====================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
HISTORY_FILE = "analysis_history.json"
MODEL_NAME = "llama3"

# File paths
MODEL_PATH = "disease_prediction_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
SYMPTOM_INDEX_PATH = "symptom_index.pkl"
MODEL_SUMMARY_PATH = "optimized_model_summary.json"

# Chart paths
CHARTS_DIR = "Charts"
CONFUSION_MATRIX_PATH = os.path.join(CHARTS_DIR, "confusion_matrix_final.png")
FEATURE_IMPORTANCE_PATH = os.path.join(CHARTS_DIR, "top_features.png")
MODEL_PERFORMANCE_PATH = os.path.join(CHARTS_DIR, "disease_distribution.png")

# Dataset paths
DATASET_DIR = "dataset"
TRAINING_DATA_PATH = os.path.join(DATASET_DIR, "Training.csv")
TESTING_DATA_PATH = os.path.join(DATASET_DIR, "Testing.csv")

# ==================== CUSTOM CSS ====================
def inject_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
    * { font-family: 'Outfit', sans-serif; }
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .custom-card, .glass-card, .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; font-size: 3.5em; font-weight: 800;
        margin-bottom: 5px; letter-spacing: -1px;
    }
    .sub-header {
        text-align: center; color: #94a3b8; font-size: 1.2em; font-weight: 300;
        margin-bottom: 40px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white; border: none;
        padding: 12px 30px; border-radius: 12px; font-weight: 600;
        transition: all 0.3s ease; box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(14, 165, 233, 0.4);
    }
    .symptom-tag, .symptom-card {
        background: rgba(14, 165, 233, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 8px 16px; border-radius: 10px; display: inline-block; margin: 4px;
        font-size: 0.9em;
    }
    .result-box, .prediction-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        padding: 30px; border-radius: 20px; margin: 20px 0;
    }
    .confidence-meter {
        height: 10px; background: #1e293b; border-radius: 10px; overflow: hidden; margin: 15px 0;
    }
    .confidence-fill {
        height: 100%; background: linear-gradient(to right, #0ea5e9, #6366f1);
        transition: width 1s ease-in-out;
    }
    [data-testid="stSidebar"] { background-color: #020617 !important; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #94a3b8; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; background: #1e293b; padding: 6px; border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important; color: #94a3b8 !important; border-radius: 10px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #0f172a !important; color: #38bdf8 !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .info-box, .success-box {
        border-left: 4px solid #0ea5e9; background: rgba(14, 165, 233, 0.05);
        padding: 15px; border-radius: 8px; margin: 10px 0;
    }
    .warning-box {
        border-left: 4px solid #f43f5e; background: rgba(244, 63, 94, 0.05);
        padding: 15px; border-radius: 8px; margin: 10px 0;
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
    .stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"] {
        background: #1e293b !important; border: 1px solid #334155 !important; color: #f1f5f9 !important;
    }
    .stMultiSelect [data-baseweb="tag"] { background: #0ea5e9 !important; color: white !important; }
    .stats-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease;
        margin-bottom: 10px;
    }
    .stats-card:hover { transform: translateY(-5px); border-color: rgba(56, 189, 248, 0.4); }
    .stats-value { font-size: 1.8em; font-weight: 800; color: #38bdf8; margin-bottom: 5px; }
    .stats-label { color: #94a3b8; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .model-info-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def stream_mistral_response(prompt):
    """Stream response from Ollama"""
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }
    response_text = ""
    try:
        with requests.post(OLLAMA_API_URL, json=data, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = json.loads(line.decode("utf-8"))
                    response_text += decoded_line.get("response", "")
                    yield decoded_line.get("response", "")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error communicating with Ollama: {e}")
        yield None

def extract_json_from_text(text):
    """Extract JSON object from text"""
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # Try code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except:
            pass
    
    return None

def save_history_to_file():
    """Save analysis history to file"""
    with open(HISTORY_FILE, "w") as f:
        json.dump(st.session_state.analysis_history, f, indent=2)

def load_history_from_file():
    """Load analysis history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def format_list_html(items):
    """Format list items as HTML"""
    if not items:
        return "<p>No information available</p>"
    if isinstance(items, str):
        items = [item.strip() for item in items.split(",") if item.strip()]
    return "<ul style='margin-left: 20px;'>" + "".join(f"<li style='margin: 8px 0;'>{i}</li>" for i in items) + "</ul>"

def load_model_summary():
    """Load model summary from JSON"""
    if os.path.exists(MODEL_SUMMARY_PATH):
        with open(MODEL_SUMMARY_PATH, 'r') as f:
            return json.load(f)
    return None

# ==================== MODEL SETUP ====================
@st.cache_resource
def load_pretrained_model():
    """Load pre-trained model and artifacts"""
    try:
        # Check if model files exist
        if not all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, FEATURE_NAMES_PATH, SYMPTOM_INDEX_PATH]):
            st.error("❌ Model files not found. Please run the training script first.")
            st.stop()
        
        # Load model artifacts
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        symptom_index = joblib.load(SYMPTOM_INDEX_PATH)
        
        # Load core profiles for validation
        core_profiles = {}
        if os.path.exists("core_profiles.pkl"):
            core_profiles = joblib.load("core_profiles.pkl")
        
        # Load model summary
        model_summary = load_model_summary()
        
        return model, encoder, feature_names, symptom_index, model_summary, core_profiles
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def predict_disease(symptoms, model, encoder, symptom_index, core_profiles, feature_names=None):
    """Predict disease from symptoms with rule-based validation"""
    symptoms_list = symptoms.split(",")
    input_data = [0] * len(symptom_index)
    
    # Normalize input symptoms to match keys in symptom_index
    normalized_input = []
    for symptom in symptoms_list:
        if symptom in symptom_index:
            index = symptom_index[symptom]
            input_data[index] = 1
            # Get the internal name (e.g. 'high_fever')
            internal_name = feature_names[index] if feature_names is not None else ""
            normalized_input.append(internal_name)
    
    if feature_names is not None:
        input_data = pd.DataFrame([input_data], columns=feature_names)
    else:
        input_data = np.array(input_data).reshape(1, -1)
        
    prediction = encoder.classes_[model.predict(input_data)[0]]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100
    
    # --- Rule-Based Validation ---
    is_medically_consistent = True
    validation_message = ""
    
    if prediction in core_profiles:
        cores = core_profiles[prediction]
        # Check how many core symptoms are present in user input
        matching_cores = [s for s in normalized_input if s in cores]
        
        # Specific strict rules for localized diseases
        localized_diseases = ['Varicose veins', 'Acne', 'Arthritis', 'Psoriasis']
        if prediction in localized_diseases and not matching_cores:
            is_medically_consistent = False
            confidence = confidence * 0.1 # Heavily penalize confidence
            validation_message = f"⚠️ Low medical consistency: Predicted {prediction} but missing core localized symptoms."
        
        # General mismatch check
        elif len(matching_cores) == 0 and len(normalized_input) > 0:
            is_medically_consistent = False
            confidence = confidence * 0.5 # Penalize confidence
            validation_message = f"⚠️ Prediction might be inconsistent with reported symptoms."

    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_predictions = [
        (encoder.classes_[idx], probabilities[idx] * 100)
        for idx in top_3_indices
    ]
    
    return prediction, confidence, top_3_predictions, is_medically_consistent, validation_message

# ==================== ANALYSIS FUNCTION ====================
def analyze_symptoms(symptoms, disease):
    """Analyze symptoms and get recommendations"""
    prompt = (
        f"You are an expert medical assistant. Based on these symptoms: {', '.join(symptoms)}, "
        f"and the predicted disease: {disease}.\n\n"
        "Provide a comprehensive medical analysis in ONLY valid JSON format:\n"
        '{\n'
        '  "medicines": "medicine1, medicine2, medicine3 (with brief dosage info if applicable)",\n'
        '  "precautions": "precaution1, precaution2, precaution3 (detailed)",\n'
        '  "advice": "advice1, advice2, advice3 (lifestyle and care tips)",\n'
        '  "severity": "mild/moderate/severe",\n'
        '  "when_to_see_doctor": "specific warning signs and when immediate medical attention is needed",\n'
        '  "diet_recommendations": "dietary suggestions",\n'
        '  "things_to_avoid": "what to avoid (activities, foods, etc.)"\n'
        '}\n\n'
        "Return ONLY the JSON object with no additional text before or after."
    )
    
    full_text = ""
    for chunk in stream_mistral_response(prompt):
        if chunk is None:
            yield None
            return
        full_text += chunk
        yield full_text

# ==================== DISPLAY RESULTS ====================
def display_analysis_results(analysis_data):
    """Display analysis results"""
    st.markdown("---")
    st.markdown("## 📊 Comprehensive Analysis Report")
    
    # Disease prediction with confidence
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="result-box">
            <h3>🔬 Predicted Disease</h3>
            <h2 style="margin: 15px 0;">{analysis_data['predicted_disease']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>🎯 Confidence</h3>
            <h2>{analysis_data['confidence']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        severity = analysis_data['analysis'].get('severity', 'moderate').upper()
        color_map = {"MILD": "#30cfd0", "MODERATE": "#fa709a", "SEVERE": "#f5576c", "UNKNOWN": "#666"}
        color = color_map.get(severity, "#4facfe")
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>⚠️ Severity</h3>
            <h2>{severity}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Reported symptoms
    st.markdown("### 🤒 Reported Symptoms")
    cols = st.columns(4)
    for i, symptom in enumerate(analysis_data['symptoms']):
        cols[i % 4].markdown(f'<div class="symptom-card">{symptom}</div>', unsafe_allow_html=True)
    
    # Alternative predictions
    if 'top_3_predictions' in analysis_data and len(analysis_data['top_3_predictions']) > 1:
        st.markdown("### 🎯 Alternative Diagnoses")
        cols = st.columns(3)
        for i, (disease, prob) in enumerate(analysis_data['top_3_predictions'][:3]):
            with cols[i]:
                st.markdown(f"""
                <div class="model-info-card">
                    <h4>{i+1}. {disease}</h4>
                    <p style="font-size: 1.2em; color: #667eea; font-weight: 600;">{prob:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💊 Medicines", 
        "⚠️ Precautions", 
        "🏥 Medical Advice",
        "🍎 Diet & Lifestyle",
        "🚨 Warning Signs"
    ])
    
    with tab1:
        st.markdown("### 💊 Recommended Medicines")
        st.markdown(f'<div class="info-box">{format_list_html(analysis_data["analysis"].get("medicines", "N/A"))}</div>', 
                    unsafe_allow_html=True)
        st.info("⚠️ **Disclaimer:** These are general recommendations. Always consult a healthcare professional before taking any medication.")
    
    with tab2:
        st.markdown("### ⚠️ Precautionary Measures")
        st.markdown(f'<div class="warning-box">{format_list_html(analysis_data["analysis"].get("precautions", "N/A"))}</div>', 
                    unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🏥 Medical Advice & Care")
        st.markdown(f'<div class="success-box">{format_list_html(analysis_data["analysis"].get("advice", "N/A"))}</div>', 
                    unsafe_allow_html=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🍎 Diet Recommendations")
            diet = analysis_data["analysis"].get("diet_recommendations", "Maintain a balanced diet")
            st.markdown(f'<div class="info-box">{format_list_html(diet)}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🚫 Things to Avoid")
            avoid = analysis_data["analysis"].get("things_to_avoid", "Consult your doctor")
            st.markdown(f'<div class="warning-box">{format_list_html(avoid)}</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown("### 🚨 When to See a Doctor Immediately")
        when_to_see = analysis_data["analysis"].get("when_to_see_doctor", "If symptoms worsen or persist, consult a healthcare professional immediately.")
        st.error(when_to_see)
    
    # Download report button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        report_data = {
            "timestamp": analysis_data["timestamp"],
            "predicted_disease": analysis_data["predicted_disease"],
            "confidence": f"{analysis_data['confidence']:.1f}%",
            "symptoms": analysis_data["symptoms"],
            "severity": analysis_data["analysis"].get("severity", "N/A"),
            "recommendations": analysis_data["analysis"]
        }
        
        st.download_button(
            label="📄 Download Full Report (JSON)",
            data=json.dumps(report_data, indent=2),
            file_name=f"medical_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width='stretch'
        )

# ==================== CHAT INTERFACE ====================
def display_chat_interface(analysis_data):
    """Display chat interface for follow-up questions"""
    st.markdown("---")
    st.markdown("## 💬 Ask Follow-up Questions")
    st.markdown("*Have questions about your diagnosis? Ask our AI medical assistant!*")
    
    # Clear chat button
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role.lower()):
            st.markdown(message)
    
    # Chat input
    user_input = st.chat_input("Ask anything about your diagnosis, treatment, or precautions...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append(("User", user_input))
        
        # Generate response
        chat_prompt = (
            f"You are an expert medical assistant helping a patient.\n\n"
            f"**Context:**\n"
            f"- Patient symptoms: {', '.join(analysis_data['symptoms'])}\n"
            f"- Diagnosed disease: {analysis_data['predicted_disease']} (Confidence: {analysis_data['confidence']:.1f}%)\n"
            f"- Severity: {analysis_data['analysis'].get('severity', 'N/A')}\n"
            f"- Recommended medicines: {analysis_data['analysis'].get('medicines', 'N/A')}\n"
            f"- Precautions: {analysis_data['analysis'].get('precautions', 'N/A')}\n\n"
            f"**Patient Question:** {user_input}\n\n"
            f"Provide a helpful, empathetic, and medically accurate answer. Be concise but thorough. "
            f"If the question is about medication dosage or serious medical decisions, remind them to consult their doctor. "
            f"Format your response in a clear, easy-to-read manner with bullet points if needed."
        )
        
        response_text = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for chunk in stream_mistral_response(chat_prompt):
                if chunk:
                    response_text += chunk
                    message_placeholder.markdown(response_text + "▌")
                else:
                    message_placeholder.error("Failed to get response from AI")
                    break
            
            if response_text:
                message_placeholder.markdown(response_text)
        
        if response_text:
            st.session_state.chat_history.append(("Assistant", response_text))
            
            # Update history file
            if st.session_state.analysis_history:
                # Find the current analysis in history (usually the last one added)
                # Note: We should update the LAST entry, assuming it was just performed.
                # A more robust solution might use a unique ID.
                st.session_state.analysis_history[-1]["chat"] = st.session_state.chat_history
                save_history_to_file()
            
            st.rerun()

# ==================== HISTORY VIEW ====================
def display_history_view(entry):
    """Display historical analysis with full details"""
    st.markdown("## 📄 Detailed Medical Report")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📅 Session Date: {entry['timestamp']}")
    with col2:
        if st.button("🔙 Back to Dashboard", width='stretch'):
            st.session_state.viewing_history = False
            st.session_state.viewed_entry = None
            st.rerun()
    
    st.markdown("---")
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="result-box">
            <h3>🔬 Disease</h3>
            <h3>{entry['predicted_disease']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3>🎯 Confidence</h3>
            <h3>{entry.get('confidence', 0):.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        severity = entry['analysis'].get('severity', 'moderate').upper()
        color_map = {"MILD": "#30cfd0", "MODERATE": "#fa709a", "SEVERE": "#f5576c", "UNKNOWN": "#666"}
        color = color_map.get(severity, "#4facfe")
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 20px; border-radius: 12px; text-align: center;">
            <h3>⚠️ Severity</h3>
            <h3>{severity}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Display full analysis
    display_analysis_results(entry)
    
    # Chat history
    if entry.get("chat"):
        st.markdown("---")
        st.markdown("### 💬 Conversation History")
        st.info(f"Total messages: {len(entry['chat'])}")
        
        for role, message in entry["chat"]:
            with st.chat_message(role.lower()):
                st.markdown(message)
    else:
        st.info("No conversation history for this session")

# ==================== MODEL INFO VIEW ====================
def display_model_info(model_summary):
    """Display model information and charts"""
    st.markdown("## 🤖 Model Information")
    
    if model_summary:
        # Model stats in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{model_summary.get('testing_accuracy', 0):.2%}</div>
                <div class="stats-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{model_summary.get('testing_precision', 0):.2%}</div>
                <div class="stats-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{model_summary.get('testing_recall', 0):.2%}</div>
                <div class="stats-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{model_summary.get('testing_f1', 0):.2%}</div>
                <div class="stats-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model details
        with st.expander("📋 Model Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Model Type:** {model_summary.get('model_type', 'N/A')}  
                **Training Accuracy:** {model_summary.get('training_accuracy', 0):.2%}  
                **CV Mean Accuracy:** {model_summary.get('cv_mean', 0):.2%} ± {model_summary.get('cv_std', 0):.4f}  
                **Number of Features:** {model_summary.get('num_features', 0)}
                """)
            
            with col2:
                st.markdown(f"""
                **Number of Classes:** {model_summary.get('num_classes', 0)}  
                **Best Parameters:** """)
                for key, value in model_summary.get('best_parameters', {}).items():
                    st.markdown(f"- {key}: `{value}`")
        
        # Display charts
        st.markdown("---")
        st.markdown("## 📊 Model Performance Charts")
        
        tab1, tab2, tab3 = st.tabs(["📈 Performance Summary", "🎯 Feature Importance", "🔄 Confusion Matrix"])
        
        with tab1:
            if os.path.exists(MODEL_PERFORMANCE_PATH):
                try:
                    image = Image.open(MODEL_PERFORMANCE_PATH)
                    st.image(image, width='stretch')
                except:
                    st.error("Could not load performance chart")
            else:
                st.warning("Performance chart not found")
        
        with tab2:
            if os.path.exists(FEATURE_IMPORTANCE_PATH):
                try:
                    image = Image.open(FEATURE_IMPORTANCE_PATH)
                    st.image(image, width='stretch')
                except:
                    st.error("Could not load feature importance chart")
            else:
                st.warning("Feature importance chart not found")
        
        with tab3:
            if os.path.exists(CONFUSION_MATRIX_PATH):
                try:
                    image = Image.open(CONFUSION_MATRIX_PATH)
                    st.image(image, width='stretch')
                except:
                    st.error("Could not load confusion matrix")
            else:
                st.warning("Confusion matrix not found")
    else:
        st.warning("Model summary file not found. Please run the training script.")

# ==================== MAIN VIEW ====================
def display_main_view(model, encoder, feature_names, symptom_index, core_profiles, ollama_status):
    """Display the main symptom input and analysis view"""
    
    # Available symptoms
    available_symptoms = sorted(list(symptom_index.keys()))
    
    # Info message if Ollama is offline
    if not ollama_status:
        st.warning("⚠️ AI Assistant is offline. Disease prediction will work, but AI recommendations won't be available. Make sure Ollama is running with llama3 model.")
    
    with st.expander("📋 **View All Available Symptoms** (Click to expand)", expanded=False):
        st.markdown(f"**Total Symptoms Available:** {len(available_symptoms)}")
        cols = st.columns(4)
        for i, symptom in enumerate(available_symptoms):
            cols[i % 4].markdown(f'<div class="symptom-card">{symptom}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Symptom selection
    st.markdown("### 🔍 Select Your Symptoms")
    st.markdown("*Please select at least 3 symptoms for accurate prediction. You can select up to 8 symptoms.*")
    
    col1, col2 = st.columns(2)
    
    symptoms = []
    symptom_options = ["None"] + available_symptoms
    
    with col1:
        s1 = st.selectbox("Symptom 1 *", symptom_options, key="s1")
        s2 = st.selectbox("Symptom 2 *", symptom_options, key="s2")
        s3 = st.selectbox("Symptom 3 *", symptom_options, key="s3")
        s4 = st.selectbox("Symptom 4", symptom_options, key="s4")
    
    with col2:
        s5 = st.selectbox("Symptom 5", symptom_options, key="s5")
        s6 = st.selectbox("Symptom 6", symptom_options, key="s6")
        s7 = st.selectbox("Symptom 7", symptom_options, key="s7")
        s8 = st.selectbox("Symptom 8", symptom_options, key="s8")
    
    for s in [s1, s2, s3, s4, s5, s6, s7, s8]:
        if s and s != "None" and s not in symptoms:
            symptoms.append(s)
    
    # Display selected symptoms
    if symptoms:
        st.markdown("#### 📝 Selected Symptoms:")
        cols = st.columns(4)
        for i, symptom in enumerate(symptoms):
            cols[i % 4].markdown(f'<div class="symptom-card">✓ {symptom}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analyze button
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        analyze_btn = st.button("🔬 **Analyze Symptoms**", width='stretch', type="primary")
    
    if analyze_btn:
        if len(symptoms) < 3:
            st.error("⚠️ Please select at least 3 symptoms for accurate prediction.")
        else:
            perform_analysis(symptoms, model, encoder, feature_names, symptom_index, core_profiles, ollama_status)
    
    # Display current analysis
    if st.session_state.current_analysis:
        display_analysis_results(st.session_state.current_analysis)
        if ollama_status:
            display_chat_interface(st.session_state.current_analysis)

# ==================== ANALYSIS EXECUTION ====================
# ==================== ANALYSIS EXECUTION ====================
def perform_analysis(symptoms, model, encoder, feature_names, symptom_index, core_profiles, ollama_status):
    """Perform disease prediction and AI analysis"""
    
    try: 
        with st.spinner("🔄 Analyzing your symptoms..."):
            # Predict disease with validation
            predicted_disease, confidence, top_3, is_consistent, val_msg = predict_disease(
                ",".join(symptoms), model, encoder, symptom_index, core_profiles, feature_names
            )
            
            if not is_consistent:
                st.warning(val_msg)
            
            st.success(f"✅ Analysis complete! Predicted: **{predicted_disease}** (Confidence: {confidence:.1f}%)")
            
            # Show top 3 predictions
            with st.expander("🎯 View Alternative Predictions"):
                for i, (disease, prob) in enumerate(top_3, 1):
                    st.markdown(f"{i}. **{disease}** - {prob:.1f}% confidence")
            
            analysis_data = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symptoms": symptoms,
                "predicted_disease": predicted_disease,
                "confidence": confidence,
                "top_3_predictions": top_3,
                "analysis": {},
                "chat": []
            }
            
            # Get AI recommendations if Ollama is available
            if ollama_status:
                full_response = ""
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🤖 AI is generating comprehensive recommendations...")
                
                # Use the analyze_symptoms generator to stream the response
                for i, updated_text in enumerate(analyze_symptoms(symptoms, predicted_disease)):
                    if updated_text:
                        full_response = updated_text
                        # Update progress roughly
                        progress_bar.progress(min(0.1 + (i * 0.05), 0.9))
                
                progress_bar.progress(1.0)
                progress_bar.empty()
                status_text.empty()
                
                # Parse response
                parsed = extract_json_from_text(full_response)
                
                if parsed:
                    analysis_data["analysis"] = parsed
                else:
                    st.warning("⚠️ AI response received but could not be parsed properly. Check Ollama server logs.")
                    analysis_data["analysis"] = {
                        "medicines": "Please consult a healthcare professional",
                        "precautions": "Seek medical attention",
                        "advice": "Unable to parse AI recommendations. Raw response: " + full_response[:100] + "...",
                        "severity": "unknown"
                    }
            else:
                analysis_data["analysis"] = {
                    "medicines": "AI Assistant offline - Please consult a healthcare professional",
                    "precautions": "AI Assistant offline - Seek medical attention",
                    "advice": "AI Assistant offline - Start Ollama for AI recommendations",
                    "severity": "unknown"
                }
            
            # Final state updates inside the 'try' block
            st.session_state.current_analysis = analysis_data
            st.session_state.analysis_history.append(analysis_data)
            save_history_to_file()
            st.rerun() # Rerun to display the analysis results
            
    except Exception as e: # <--- NOW CORRECTLY ALIGNED with 'try'
        st.error(f"❌ Error during analysis: {e}")
        import traceback
        st.error(traceback.format_exc())
# ==================== MAIN APP ====================
def main():
    # Page config
    st.set_page_config(
        page_title="AI Disease Predictor Pro",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject CSS
    inject_custom_css()
    
    # Initialize session state
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = load_history_from_file()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "viewing_history" not in st.session_state:
        st.session_state.viewing_history = False
    if "viewed_entry" not in st.session_state:
        st.session_state.viewed_entry = None
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "show_model_info" not in st.session_state:
        st.session_state.show_model_info = False
    
    # Load model
    model, encoder, feature_names, symptom_index, model_summary, core_profiles = load_pretrained_model()
    
    # Check Ollama connection
    ollama_status = check_ollama_connection()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Disease Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML-Powered Medical Analysis System</p>', unsafe_allow_html=True)
    
    # Status bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Model:** {model_summary.get('model_type', 'Random Forest') if model_summary else 'Random Forest'}")
    with col2:
        st.markdown(f"**Diseases:** {len(encoder.classes_)}")
    with col3:
        status_color = "🟢" if ollama_status else "🔴"
        st.markdown(f"{status_color} **AI:** {'Online' if ollama_status else 'Offline'}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Dashboard")
        
        # Model info button
        if st.button("ℹ️ Model Information", width='stretch'):
            st.session_state.show_model_info = True # Navigate to Model Info
            st.session_state.viewing_history = False
            st.rerun()
        
        # Statistics
        total_analyses = len(st.session_state.analysis_history)
        total_diseases = len(encoder.classes_)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_analyses}</div>
                <div class="metric-label">Analyses</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_diseases}</div>
                <div class="metric-label">Diseases</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🎯 Navigation")
        
        if st.button("🏠 New Analysis", width='stretch', type="primary"):
            st.session_state.viewing_history = False
            st.session_state.viewed_entry = None
            st.session_state.current_analysis = None # Reset for new analysis
            st.session_state.chat_history = []
            st.session_state.show_model_info = False # Reset model info view
            st.rerun()
                
        if st.button("📈 View Model Charts", width='stretch'):
            st.session_state.show_model_info = True
            st.session_state.viewing_history = False
            st.rerun()
            
        st.markdown("---")
        st.markdown("## 📚 Recent Sessions")
        
        if st.session_state.analysis_history:
            # Displaying last 10 entries
            for i, entry in enumerate(reversed(st.session_state.analysis_history[-10:])):
                # Use a unique key for the button (e.g., index in history list)
                original_index = len(st.session_state.analysis_history) - 1 - i 
                timestamp = entry['timestamp']
                disease = entry['predicted_disease']
                confidence = entry.get('confidence', 0)
                
                if st.button(
                    f"📅 {timestamp[:16]}\n🔬 {disease[:20]}{'...' if len(disease) > 20 else ''}\n🎯 {confidence:.1f}%",
                    key=f"hist_{original_index}",
                    width='stretch'
                ):
                    st.session_state.viewing_history = True
                    st.session_state.viewed_entry = entry
                    st.session_state.show_model_info = False
                    st.rerun()
        else:
            st.info("No previous analyses")
        
        # Export history button
        if st.session_state.analysis_history:
            st.markdown("---")
            # We use a download button which does not trigger a rerun automatically
            st.download_button(
                label="💾 Export History",
                data=json.dumps(st.session_state.analysis_history, indent=2),
                file_name=f"medical_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width='stretch'
            )
    
    # Main content routing
    if st.session_state.show_model_info:
        display_model_info(model_summary)
    elif st.session_state.viewing_history and st.session_state.viewed_entry:
        display_history_view(st.session_state.viewed_entry)
    else:
        display_main_view(model, encoder, feature_names, symptom_index, core_profiles, ollama_status)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()