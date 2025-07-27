import streamlit as st
import pandas as pd
from utils import extract_text_from_image, clean_ocr_text, speech_to_text_whisper
from ml_model import predict_common_diseases
from rag_helper import build_knowledge_base, get_medical_guidance_with_llm

st.set_page_config(page_title="💊 GenAI Drug Finder", layout="centered")
st.title("💊 GenAI Drug Finder & AI Health Guide")

# ------------------------------
# Load data and Knowledge Base
# ------------------------------
medicine_df = pd.read_csv("data/medicine_data.csv")
billing_data_path = "data/billing_data.csv"
kb_index, kb_texts, kb_sources = build_knowledge_base("data/medical_guides")

# ------------------------------
# Medicine Search Section
# ------------------------------
st.header("🔎 Search for Medicine")
method = st.radio("Choose Input Method", ["Text Search", "Upload Prescription", "Voice Message"])
query = ""

if method == "Text Search":
    query = st.text_input("Enter medicine name")

elif method == "Upload Prescription":
    file = st.file_uploader("Upload prescription image", type=["png", "jpg", "jpeg"])
    if file:
        raw_text = extract_text_from_image(file)
        query = clean_ocr_text(raw_text)
        st.success(f"OCR Extracted: {raw_text}")
        st.info(f"Cleaned Text Used for Search: {query}")

elif method == "Voice Message":
    audio_file = st.file_uploader("Upload voice message (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
    if audio_file:
        st.info("Transcribing with Whisper-Tiny model...")
        query = speech_to_text_whisper(audio_file)
        st.success(f"You said: {query}")

# Search medicine
if query:
    results = medicine_df[medicine_df['medicine_name'].str.contains(query, case=False, na=False, regex=False)]
    if not results.empty:
        st.subheader("💊 Matching Medicines")
        st.dataframe(results)
    else:
        st.warning("Medicine not found in inventory.")

# ------------------------------
# Predict Common Diseases
# ------------------------------
st.header("📊 Predict Common Diseases in Area")
if st.button(":chart_with_upwards_trend: Analyze Billing Data"):
    top_diseases = predict_common_diseases(billing_data_path)
    st.success("Top Diseases:")
    for disease in top_diseases:
        st.markdown(f"- {disease}")

# ------------------------------
# AI Health Assistant (RAG + LLM)
# ------------------------------
st.header("🧠 AI Health Assistant")
disease_input = st.selectbox("Select or type a disease", ["cancer", "diabetes", "hypertension", "obesity", "asthma", "other"])
custom_input = ""

if disease_input == "other":
    custom_input = st.text_input("Enter disease name")

selected_disease = custom_input.strip() if disease_input == "other" else disease_input

if selected_disease:
    # Check if selected_disease is present in any of the knowledge base texts
    found = any(selected_disease.lower() in text.lower() for text in kb_texts)

    if found:
        guidance = get_medical_guidance_with_llm(selected_disease, kb_index, kb_texts)
    else:
        st.warning("No specific guidance found in knowledge base, using LLM to generate guidance.")
        guidance = get_medical_guidance_with_llm(selected_disease, None, None, use_fallback=True)

    st.subheader("📘 Medical Guidance")
    st.write(guidance)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
---
App built for GenAI Hackathon | Powered by open-source LLMs, Whisper, and Corrective RAG
""")