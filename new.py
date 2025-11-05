import streamlit as st
import requests
import json
import datetime
import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ====================================================
# CONFIGURATION
# ====================================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
HISTORY_FILE = "analysis_history.json"
MODEL_FILE = "disease_prediction_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
SYMPTOM_INDEX_FILE = "symptom_index.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"

# ====================================================
# LOAD TRAINED MODEL AND ARTIFACTS
# ====================================================
@st.cache_resource
def load_model_artifacts():
    """Load all saved model artifacts"""
    try:
        model = joblib.load(MODEL_FILE)
        encoder = joblib.load(ENCODER_FILE)
        symptom_index = joblib.load(SYMPTOM_INDEX_FILE)
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        
        return model, encoder, symptom_index, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please run the training script first to generate model files.")
        st.stop()

# Load model
model, encoder, symptom_index, feature_names = load_model_artifacts()

# ====================================================
# OLLAMA LLM FUNCTIONS
# ====================================================
def stream_mistral_response(prompt):
    """Stream response from Ollama LLM"""
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": True
    }
    response_text = ""
    try:
        with requests.post(OLLAMA_API_URL, json=data, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = json.loads(line.decode("utf-8"))
                    response_text += decoded_line.get("response", "")
                    yield decoded_line.get("response", "")
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        yield None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding Ollama response: {e}")
        yield None

def extract_json_from_text(text):
    """Extract JSON object from text that may contain extra content"""
    import re
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON block between curly braces
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # Try to find JSON in code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except:
            pass
    
    return None

def analyze_symptoms(symptoms, disease, confidence):
    """Run symptom analysis using LLM"""
    prompt = (
        f"Based on the symptoms: {', '.join(symptoms)}, "
        f"the predicted disease is {disease} with {confidence:.2f}% confidence.\n\n"
        "Return ONLY valid JSON like this:\n"
        '{\n'
        '  "medicines": "List of suggested medicines separated by commas",\n'
        '  "precautions": "List of precautions separated by commas",\n'
        '  "advice": "Important medical advice and next steps"\n'
        '}\n'
        "No explanation. Only JSON. Do not add any text before or after the JSON."
    )

    full_text = ""
    for chunk in stream_mistral_response(prompt):
        if chunk is None:
            yield None
            return
        full_text += chunk
        yield full_text

# ====================================================
# PREDICTION FUNCTION
# ====================================================
def predict_disease(symptoms_list):
    """Predict disease using trained model"""
    # Create input vector
    input_data = [0] * len(feature_names)
    
    for symptom in symptoms_list:
        if symptom in symptom_index:
            index = symptom_index[symptom]
            input_data[index] = 1
    
    # Reshape for prediction
    input_data = np.array(input_data).reshape(1, -1)
    
    # Get prediction and probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Get disease name and confidence
    disease_name = encoder.classes_[prediction]
    confidence = probabilities[prediction] * 100
    
    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = [(encoder.classes_[idx], probabilities[idx] * 100) 
                      for idx in top_3_indices]
    
    return disease_name, confidence, top_3_diseases

# ====================================================
# HISTORY MANAGEMENT
# ====================================================
def save_history_to_file():
    """Save analysis history to file"""
    with open(HISTORY_FILE, "w") as f:
        json.dump(st.session_state.analysis_history, f, indent=2)

def load_history_from_file():
    """Load analysis history from file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# ====================================================
# UI HELPER FUNCTIONS
# ====================================================
def dark_box(content, color="#2b2b2b"):
    """Create dark themed box"""
    return f"<div style='background-color:{color};padding:15px;border-radius:10px;color:white;'>{content}</div>"

def format_list(items):
    """Format list items for display"""
    if not items:
        return "<ul><li>N/A</li></ul>"
    if isinstance(items, str):
        items = [item.strip() for item in items.split(",") if item.strip()]
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"

def confidence_badge(confidence):
    """Create colored confidence badge"""
    if confidence >= 80:
        color = "#28a745"  # Green
        label = "High Confidence"
    elif confidence >= 60:
        color = "#ffc107"  # Yellow
        label = "Medium Confidence"
    else:
        color = "#dc3545"  # Red
        label = "Low Confidence"
    
    return f"<span style='background-color:{color};color:white;padding:5px 15px;border-radius:20px;font-weight:bold;'>{label}: {confidence:.1f}%</span>"

# ====================================================
# STREAMLIT APP SETUP
# ====================================================
st.set_page_config(page_title="AI Disease Predictor", layout="wide", page_icon="🏥")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1c1c1c;
        color: white;
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #2c2c2c;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #444;
    }
    .symptom-chip {
        display: inline-block;
        background-color: #667eea;
        color: white;
        padding: 8px 15px;
        margin: 5px;
        border-radius: 20px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 AI-Powered Disease Prediction System</h1>
    <p>Advanced Medical Diagnosis using Machine Learning & LLM Analysis</p>
</div>
""", unsafe_allow_html=True)

# ====================================================
# SESSION STATE INITIALIZATION
# ====================================================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = load_history_from_file()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = False
if "viewed_entry" not in st.session_state:
    st.session_state.viewed_entry = None

# ====================================================
# SIDEBAR - HISTORY & MODEL INFO
# ====================================================
with st.sidebar:
    st.title("📊 Model Information")
    
    # Model stats
    st.markdown("### 🤖 Model Details")
    st.info(f"""
    **Algorithm:** Random Forest  
    **Features:** {len(feature_names)}  
    **Diseases:** {len(encoder.classes_)}  
    **Status:** ✅ Loaded
    """)
    
    st.markdown("---")
    
    # History section
    st.title("📚 Analysis History")
    if st.session_state.analysis_history:
        st.success(f"Total Sessions: {len(st.session_state.analysis_history)}")
        for i, entry in enumerate(reversed(st.session_state.analysis_history)):
            label = f"📅 {entry['timestamp'][:16]}"
            disease = entry.get('predicted_disease', 'N/A')
            if st.button(f"{label}\n🩺 {disease}", key=f"history_{i}", use_container_width=True):
                st.session_state.viewing_history = True
                st.session_state.viewed_entry = entry
                st.rerun()
    else:
        st.info("No history available yet.")
    
    if st.session_state.viewing_history:
        if st.button("🔙 Back to New Analysis", use_container_width=True):
            st.session_state.viewing_history = False
            st.session_state.viewed_entry = None
            st.rerun()

# ====================================================
# MAIN CONTENT - NEW ANALYSIS
# ====================================================
if not st.session_state.get("viewing_history"):
    
    # Display available symptoms
    with st.expander("📋 View All Available Symptoms", expanded=False):
        st.markdown("**Select from these symptoms:**")
        available_symptoms = list(symptom_index.keys())
        
        cols_per_row = 3
        for i in range(0, len(available_symptoms), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(available_symptoms):
                    cols[j].markdown(
                        f"<div style='background-color:#1e1e1e;color:white;padding:10px;margin:5px;border-radius:8px;'>"
                        f"{i+j+1}. {available_symptoms[i+j]}</div>",
                        unsafe_allow_html=True
                    )
    
    st.markdown("---")
    st.markdown("### 🩺 Enter Your Symptoms")
    
    # Symptom selection
    symptoms = []
    symptom_options = ["None"] + sorted(list(symptom_index.keys()))
    
    cols = st.columns(2)
    for i in range(5):
        col = cols[i % 2]
        with col:
            s = st.selectbox(
                f"Symptom {i+1}",
                symptom_options,
                key=f"symptom_{i}"
            )
            if s and s != "None":
                symptoms.append(s)
    
    # Display selected symptoms
    if symptoms:
        st.markdown("### ✅ Selected Symptoms:")
        symptoms_html = "".join([f"<span class='symptom-chip'>{s}</span>" for s in symptoms])
        st.markdown(symptoms_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analyze Symptoms", use_container_width=True, type="primary")
    
    if analyze_btn:
        if len(symptoms) < 2:
            st.error("⚠️ Please select at least 2 symptoms for accurate prediction.")
        else:
            st.session_state.chat_history = []
            st.session_state.symptom_context = ", ".join(symptoms)
            st.session_state.analysis_json = None
            
            with st.spinner("🔬 Analyzing symptoms with AI..."):
                try:
                    # Get prediction
                    predicted_disease, confidence, top_3 = predict_disease(symptoms)
                    st.session_state.predicted_disease = predicted_disease
                    st.session_state.confidence = confidence
                    st.session_state.top_3_predictions = top_3
                    
                    # Display prediction
                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🩺 Predicted Disease</h3>
                            <h2 style='color:#667eea;'>{predicted_disease}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊 Confidence Level</h3>
                            <h2>{confidence_badge(confidence)}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show top 3 predictions
                    st.markdown("### 📈 Top 3 Probable Diseases")
                    top_cols = st.columns(3)
                    for idx, (disease, prob) in enumerate(top_3):
                        with top_cols[idx]:
                            st.markdown(f"""
                            <div style='background-color:#2c2c2c;padding:15px;border-radius:10px;text-align:center;'>
                                <h4>{idx+1}. {disease}</h4>
                                <p style='color:#667eea;font-size:20px;'>{prob:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Get LLM analysis
                    st.markdown("## 💊 Medical Recommendations")
                    with st.spinner("Getting detailed medical recommendations..."):
                        full_response = ""
                        error_occurred = False
                        
                        for chunk in analyze_symptoms(symptoms, predicted_disease, confidence):
                            if chunk is None:
                                error_occurred = True
                                break
                            full_response = chunk
                        
                        if not error_occurred:
                            parsed = extract_json_from_text(full_response)
                            
                            if parsed:
                                st.session_state.analysis_json = parsed
                                
                                # Save to history
                                new_entry = {
                                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "symptoms": symptoms,
                                    "predicted_disease": predicted_disease,
                                    "confidence": confidence,
                                    "top_3": top_3,
                                    "analysis": parsed,
                                    "chat": []
                                }
                                st.session_state.analysis_history.append(new_entry)
                                st.session_state.viewed_entry = new_entry
                                save_history_to_file()
                                
                                st.success("✅ Analysis completed successfully!")
                            else:
                                st.error("❌ Could not parse AI response.")
                                st.code(full_response)
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

# ====================================================
# DISPLAY ANALYSIS RESULTS
# ====================================================
if st.session_state.get("analysis_json") and not st.session_state.get("viewing_history"):
    parsed = st.session_state.analysis_json
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💊 Suggested Medicines")
        st.markdown(dark_box(format_list(parsed.get('medicines', 'N/A')), "#2c2c2c"), unsafe_allow_html=True)
        
        st.markdown("### ⚠️ Precautions")
        st.markdown(dark_box(format_list(parsed.get('precautions', 'N/A')), "#333"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏥 Medical Advice")
        st.markdown(dark_box(parsed.get('advice', 'N/A'), "#2b2b2b"), unsafe_allow_html=True)
    
    st.warning("⚠️ **Disclaimer:** This is an AI-based prediction tool. Always consult with a qualified healthcare professional for accurate diagnosis and treatment.")

# ====================================================
# DISPLAY HISTORY VIEW
# ====================================================
if st.session_state.get("viewing_history") and st.session_state.get("viewed_entry"):
    entry = st.session_state.viewed_entry
    
    st.markdown("## 📄 Detailed Analysis Report")
    
    # Header info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📅 Date & Time</h4>
            <p>{entry['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🩺 Disease</h4>
            <h3 style='color:#667eea;'>{entry['predicted_disease']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = entry.get('confidence', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Confidence</h4>
            <p>{confidence_badge(confidence)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Symptoms
    st.markdown("### 🤒 Reported Symptoms")
    symptoms_html = "".join([f"<span class='symptom-chip'>{s}</span>" for s in entry.get('symptoms', [])])
    st.markdown(symptoms_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💊 Suggested Medicines")
        st.markdown(dark_box(format_list(entry['analysis'].get('medicines', 'N/A')), "#2c2c2c"), unsafe_allow_html=True)
        
        st.markdown("### ⚠️ Precautions")
        st.markdown(dark_box(format_list(entry['analysis'].get('precautions', 'N/A')), "#333"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏥 Medical Advice")
        st.markdown(dark_box(entry['analysis'].get('advice', 'N/A'), "#2b2b2b"), unsafe_allow_html=True)
    
    # Chat history
    if entry.get("chat"):
        st.markdown("---")
        st.markdown("### 💬 Chat History")
        for speaker, msg in entry["chat"]:
            with st.chat_message(speaker):
                st.markdown(msg)

# ====================================================
# CHAT SECTION
# ====================================================
if st.session_state.get("symptom_context") and not st.session_state.get("viewing_history"):
    st.markdown("---")
    st.markdown("### 💬 Ask Your Medical Questions")
    
    # Clear chat button
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
    
    # Chat input
    chat_input = st.chat_input("Ask a question about your symptoms or diagnosis...")
    
    if chat_input:
        # Add user message
        st.session_state.chat_history.append(("user", chat_input))
        
        # Prepare prompt
        analysis = st.session_state.get('analysis_json', {})
        chat_prompt = (
            f"Patient symptoms: {st.session_state.symptom_context}\n"
            f"Predicted disease: {st.session_state.get('predicted_disease', 'N/A')}\n"
            f"Confidence: {st.session_state.get('confidence', 0):.1f}%\n"
            f"Suggested medicines: {analysis.get('medicines', 'N/A')}\n"
            f"Precautions: {analysis.get('precautions', 'N/A')}\n\n"
            f"Patient question: {chat_input}\n\n"
            "Provide a clear, concise, and helpful medical response. "
            "Be professional but friendly. Give practical advice."
        )
        
        # Get AI response
        with st.spinner("AI Doctor is thinking..."):
            response_text = ""
            for token in stream_mistral_response(chat_prompt):
                if token is None:
                    break
                response_text += token
        
        # Add AI response
        st.session_state.chat_history.append(("assistant", response_text))
        
        # Update history
        if st.session_state.get("analysis_history"):
            st.session_state.analysis_history[-1]["chat"] = st.session_state.chat_history
            save_history_to_file()
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:gray;padding:20px;'>
    <p>🏥 AI Disease Prediction System | Powered by Machine Learning & LLM</p>
    <p>⚠️ For educational purposes only. Always consult healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)