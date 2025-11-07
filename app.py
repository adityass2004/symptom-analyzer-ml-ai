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
MODEL_SUMMARY_PATH = "model_summary.json"

# Chart paths
CHARTS_DIR = "Charts"
CONFUSION_MATRIX_PATH = os.path.join(CHARTS_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = os.path.join(CHARTS_DIR, "feature_importance.png")
MODEL_PERFORMANCE_PATH = os.path.join(CHARTS_DIR, "model_performance_summary.png")

# Dataset paths
DATASET_DIR = "dataset"
TRAINING_DATA_PATH = os.path.join(DATASET_DIR, "Training.csv")
TESTING_DATA_PATH = os.path.join(DATASET_DIR, "Testing.csv")

# ==================== CUSTOM CSS ====================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        font-size: 3.5em;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    /* Symptom card */
    .symptom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
        border-radius: 10px;
        margin: 5px;
        font-size: 14px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .symptom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Result box styling */
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .result-box h3 {
        margin-top: 0;
        font-size: 1.5em;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    /* Model info card */
    .model-info-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stats-value {
        font-size: 2em;
        font-weight: 700;
        color: #667eea;
    }
    
    .stats-label {
        font-size: 0.9em;
        color: #666;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 1em;
        color: #666;
        margin-top: 5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
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
        
        # Load model summary
        model_summary = load_model_summary()
        
        return model, encoder, feature_names, symptom_index, model_summary
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def predict_disease(symptoms, model, encoder, symptom_index):
    """Predict disease from symptoms"""
    symptoms_list = symptoms.split(",")
    input_data = [0] * len(symptom_index)
    
    for symptom in symptoms_list:
        if symptom in symptom_index:
            index = symptom_index[symptom]
            input_data[index] = 1
    
    input_data = np.array(input_data).reshape(1, -1)
    prediction = encoder.classes_[model.predict(input_data)[0]]
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100
    
    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_predictions = [
        (encoder.classes_[idx], probabilities[idx] * 100)
        for idx in top_3_indices
    ]
    
    return prediction, confidence, top_3_predictions

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
        st.markdown("""
        #### Emergency Signs:
        - Severe chest pain or difficulty breathing
        - Sudden confusion or loss of consciousness
        - Severe bleeding or trauma
        - High fever (above 103°F/39.4°C) that doesn't respond to medication
        - Sudden severe headache or vision changes
        """)
    
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
            use_container_width=True
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
        if st.button("🔙 Back to Dashboard", use_container_width=True):
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
                    st.image(image, use_container_width=True)
                except:
                    st.error("Could not load performance chart")
            else:
                st.warning("Performance chart not found")
        
        with tab2:
            if os.path.exists(FEATURE_IMPORTANCE_PATH):
                try:
                    image = Image.open(FEATURE_IMPORTANCE_PATH)
                    st.image(image, use_container_width=True)
                except:
                    st.error("Could not load feature importance chart")
            else:
                st.warning("Feature importance chart not found")
        
        with tab3:
            if os.path.exists(CONFUSION_MATRIX_PATH):
                try:
                    image = Image.open(CONFUSION_MATRIX_PATH)
                    st.image(image, use_container_width=True)
                except:
                    st.error("Could not load confusion matrix")
            else:
                st.warning("Confusion matrix not found")
    else:
        st.warning("Model summary file not found. Please run the training script.")

# ==================== MAIN VIEW ====================
def display_main_view(model, encoder, feature_names, symptom_index, ollama_status):
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
        analyze_btn = st.button("🔬 **Analyze Symptoms**", use_container_width=True, type="primary")
    
    if analyze_btn:
        if len(symptoms) < 3:
            st.error("⚠️ Please select at least 3 symptoms for accurate prediction.")
        else:
            perform_analysis(symptoms, model, encoder, symptom_index, ollama_status)
    
    # Display current analysis
    if st.session_state.current_analysis:
        display_analysis_results(st.session_state.current_analysis)
        if ollama_status:
            display_chat_interface(st.session_state.current_analysis)

# ==================== ANALYSIS EXECUTION ====================
# ==================== ANALYSIS EXECUTION ====================
def perform_analysis(symptoms, model, encoder, symptom_index, ollama_status):
    """Perform disease prediction and AI analysis"""
    
    # The 'try' block must be aligned with 'except'
    # In this case, both must be aligned with 'with st.spinner(...)'
    try: 
        with st.spinner("🔄 Analyzing your symptoms..."):
            # Predict disease (This is the start of the try block content)
            predicted_disease, confidence, top_3 = predict_disease(
                ",".join(symptoms), model, encoder, symptom_index
            )
            
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
                
                # ... (rest of the streaming logic) ...
                
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
    model, encoder, feature_names, symptom_index, model_summary = load_pretrained_model()
    
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
        if st.button("ℹ️ Model Information", use_container_width=True):
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
        
        # FIX 1 & 2: Corrected "New Analysis" button logic
        if st.button("🏠 New Analysis", use_container_width=True, type="primary"):
            st.session_state.viewing_history = False
            st.session_state.viewed_entry = None
            st.session_state.current_analysis = None # Reset for new analysis
            st.session_state.chat_history = []
            st.session_state.show_model_info = False # Reset model info view
            st.rerun()
                
        if st.button("📈 View Model Charts", use_container_width=True):
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
                    use_container_width=True
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
                use_container_width=True
            )
    
    # Main content routing
    if st.session_state.show_model_info:
        display_model_info(model_summary)
    elif st.session_state.viewing_history and st.session_state.viewed_entry:
        display_history_view(st.session_state.viewed_entry)
    else:
        display_main_view(model, encoder, feature_names, symptom_index, ollama_status)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()