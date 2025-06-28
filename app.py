import streamlit as st
import time
from PIL import Image
import numpy as np
import time
import torchvision
from torchvision import datasets, transforms, models
from services.visualizationservice import VisualizationService
from services.ragservice import RAGService, analyze_retinal_image_and_heatmap
from PIL import Image
import matplotlib.pyplot as plt
from gradcam import load_model, inference

st.title("Diabetic Retinopathy Screening Tool")
"This application demonstrates how an AI-powered screening tool might work."
st.markdown('## AI-Assisted Screening for Early Detection')

# initialize session state for patient information
if 'patient_age' not in st.session_state:
    st.session_state.patient_age = None
if 'diabetes_duration' not in st.session_state:
    st.session_state.diabetes_duration = None

with st.sidebar:
    st.header("📋 Patient Information")
    st.markdown("*Optional: Provide patient details for personalized analysis*")
    
    patient_age = st.number_input(
        "Patient Age", 
        min_value=18, 
        max_value=100, 
        value=st.session_state.patient_age if st.session_state.patient_age else 50,
        help="Patient's current age"
    )
    
    diabetes_duration = st.number_input(
        "Diabetes Duration (years)", 
        min_value=0, 
        max_value=50, 
        value=st.session_state.diabetes_duration if st.session_state.diabetes_duration else 5,
        help="How long the patient has had diabetes"
    )

    st.session_state.patient_age = patient_age
    st.session_state.diabetes_duration = diabetes_duration

# description
col1, col2 = st.columns(2)
with col1:
    st.subheader('Benefits')
    st.markdown('1. **Early detection** of retinopathy')
    st.markdown('2. **Accessible** screening in various settings')
    st.markdown('3. Reduced burden on specialists')
    st.markdown('4. Evidence-based patient **education**')

with col2:
    st.subheader('How It Works')
    st.markdown('1. Upload a retinal image')
    st.markdown('2. AI model analyzes the image')
    st.markdown('3. View results with explanations')
    st.markdown('4. Make informed clinical decisions')

# upload image
st.header('Upload Retinal Image')
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Retinal Image', width=300)
    st.write("Image successfully uploaded!")

# store results from expensive tasks to speed up reloads
# st.session_state["results"] = None
# st.session_state["heatmap"] = None

# manage nested buttons
if "process" not in st.session_state:
    st.session_state["process"] = False

if st.button("Process"):
    st.session_state["process"] = True
    st.session_state["results"] = None
    st.session_state["heatmap"] = None
    print("Processing")

if st.session_state["process"]:
    # generate inference and heatmap results
    with st.spinner("In progress..."):
        image = Image.open(uploaded_file).convert('RGB')
        vs = VisualizationService()
        model = load_model("/home/charles/project_dr/Retinal_blindness_detection_Pytorch/classifier.pt")

        if (st.session_state["heatmap"] is None):
            print("Heatmap was None. Generating heatmap")
            st.session_state["heatmap"] = vs.generate_gradcam_visualization(model, uploaded_file)
            st.session_state["heatmap"].savefig("heatmap.png", format='png', bbox_inches='tight')

        heatmap = st.session_state["heatmap"]

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

        if (st.session_state["results"] is None):
            print("Results were None. Generating results")
            st.session_state["results"] = inference(model, uploaded_file, transforms, classes)

        tab1, tab2, tab3 = st.tabs(["Results", "Heatmap", "AI Explanation"])

    # results
    with tab1:
        results = st.session_state["results"]
        "Here are your results:"
        "Class: " + str(results["class"])
        "Severity: " + str(results["value"]) + "/4"
        "Probability: " + str(results["probability"])   

    # heatmap
    with tab2:
        "Here is your heatmap:"
        st.image('heatmap.png', caption='GradCAM Heatmap')
        
    # AI explanation
    with tab3:
        rag = st.checkbox("Use RAG")
        age_and_diabetes = st.checkbox("Use Patient Age and Diabetes Duration", value=(st.session_state.patient_age != 50 or st.session_state.diabetes_duration != 5))

        if st.button("Generate AI Explanation", type="secondary"):
            with st.spinner("Generating..."):
                try:
                    print("Trying to generate AI explanation")

                    if (rag):
                        rag = RAGService()
                        rag.initialize_knowledge_base()
                        text = rag.analyze_retinal_image_and_heatmap(image, heatmap, results, patient_age=patient_age, diabetes_duration=diabetes_duration)
                    else:
                        text = analyze_retinal_image_and_heatmap(image, heatmap, results, useRag=False, patient_age=patient_age, diabetes_duration=diabetes_duration)

                    st.write(text)

                except Exception as e:
                    st.error(f"❌ Error getting AI response: {str(e)}")

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
