import streamlit as st
import time
from PIL import Image
import numpy as np
import time
import torchvision
from torchvision import datasets, transforms, models
from services.visualizationservice import VisualizationService
from services.ragservice import RAGService, analyze_retinal_image_and_heatmap
from gradcam import load_model, inference

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
        model = load_model("/home/charles/project_dr/Retinal_blindness_detection_Pytorch/classifier.pt")
        st.spinner("In progress")

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        results = inference(model, uploaded_file, test_transforms, classes)

        tab1, tab2, tab3 = st.tabs(["Results", "Heatmap", "AI Explanation"])

    with tab1:
        "Here are your results:"
        "Class: " + str(results["class"])
        "Severity: " + str(results["value"])
        "Probability: " + str(results["probability"])   

    with tab2:
        vs = VisualizationService()
        image = vs.generate_gradcam_visualization(model, uploaded_file).savefig("heatmap.png", format='png')
        "Here is your heatmap:"
        st.image('heatmap.png')

    with tab3:
        if st.button("Generate explanation"):
            with st.spinner("In progress"):
                try:
                    # print("Trying to generate AI explanation")
                    rag = RAGService()
                    # print(rag)
                    rag.initialize_knowledge_base()
                    heatmap = Image.open('heatmap.png').convert('RGB')
                    text = rag.analyze_retinal_image_and_heatmap(image, heatmap, results)
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
