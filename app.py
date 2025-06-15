import streamlit as st
import time
from PIL import Image
import numpy as np
import time
from gradcam import *
from RAGService import analyze_retinal_image_and_heatmap

st.title("Diabetic Retinopathy Screening Tool")
"This application demonstrates how an AI-powered screening tool might work."
st.markdown('## AI-Assisted Screening for Early Detection')

with st.sidebar:
    st.header("About")
    script = """<div id = 'chat_outer'></div>"""
    st.markdown(script, unsafe_allow_html=True)
    with st.container():
        script = """<div id = 'chat_inner'></div>"""
        st.markdown(script, unsafe_allow_html=True)
        st.text("This is a prototype application for detecting diabetic retinopathy\nfrom retinal images.")

col1, col2 = st.columns(2)
with col1:
    st.subheader('Benefits')
    st.markdown('- Early detection of retinopathy')
    st.markdown('- Accessible screening in primary care settings')
    st.markdown('- Reduced burden on specialists')

with col2:
    st.subheader('How It Works')
    st.markdown('1. Upload a retinal image')
    st.markdown('2. AI model analyzes the image')
    st.markdown('3. View results with explanations')
    st.markdown('4. Make informed clinical decisions')

st.header('Upload Retinal Image')
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Retinal Image', width=300)
    st.write("Image successfully uploaded!")

if st.button("process"):
        with st.spinner("In progress"):
            image = Image.open(uploaded_file).convert('RGB')
            model =  load_model("/home/charles/project_dr/Retinal_blindness_detection_Pytorch/classifier.pt")
            st.spinner("In progress")
            results = test(uploaded_file)
            tab1, tab2, tab3 = st.tabs(["Results", "Heatmap", "AI Explanation"])

        with tab1:
            "Here are your results:"
            "Class: " + str(results["class"])
            "Severity: " + str(results["value"])
            "Probability: " + str(results["probability"])   

        with tab2:
            "Here is your heatmap:"
            st.image('heatmap.png')

        with tab3:

            with st.spinner("In progress"):
                if st.button("Generate explanation"):
                    try:
                        heatmap = Image.open('heatmap.png').convert('RGB')
                        text = analyze_retinal_image_and_heatmap(image, heatmap, results)
                        st.write(text)

                    except Exception as e:
                        st.error(f"❌ Error getting AI response: {str(e)}")
             

st.button("reset")


chat_plh_style = """
        <style>
            div[data-testid='stVerticalBlock']:has(div#chat_inner):not(:has(div#chat_outer)) {
                background-color: #2d425d;
                border-radius: 10px;
                padding: 10px;
            };s
        </style>
        """

st.markdown(chat_plh_style, unsafe_allow_html=True)
