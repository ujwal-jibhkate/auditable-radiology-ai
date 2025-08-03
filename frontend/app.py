import streamlit as st
import requests
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="Auditable Radiology AI",
    page_icon="🩺",
    layout="wide"
)

# --- Helper function to render Mermaid charts ---
def render_mermaid(mermaid_code: str):
    components.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <div class="mermaid" style="text-align: center;">
            {mermaid_code}
        </div>
        """,
        height=550 # Increased height slightly for the new chart
    )


# --- Flowchart Definitions---
architecture_flowchart = """
graph TD
    A[Input Image] --> VE[Vision Encoder]
    VE --> PVF[Projected Visual Features]
    B[Input Report Text] --> TD[Text Decoder]
    PVF --> TD
    TD --> SR[Shared Representation]
    SR --> TH[Text Head]
    SR --> CH[Topic Prediction Head]
    TH --> TL[Generated Report Text]
    CH --> CL[Predicted Clinical Topics]
"""
system_flowchart = """
graph TD
    User[👤 User] --> |1. Uploads Image| App[Streamlit Frontend]
    App --> |2. Sends HTTP Request| API[FastAPI Backend]
    subgraph "Backend Server"
        API --> Model[PyTorch Model]
        Model --> Auditors[Audit Agents]
        Auditors --> FinalOutput[JSON Response]
    end
    API --> |3. Sends Response| App
    App --> |4. Displays Results| User
"""


# --- Function to render the "Project Details" page content ---
def render_project_details_page():
    st.title("📖 Project Details & Design Decisions")
    st.markdown("---")

    st.markdown("### The Goal: Responsible and Transparent AI")
    st.info("""
    This is an end-to-end portfolio project demonstrating a modern MLOps workflow. It features a sophisticated vision-language model that generates radiology reports from X-ray images. Crucially, the model is then audited for fairness and logical consistency to ensure its outputs are reliable. The entire system is deployed as a web application with a separate frontend and backend.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["🏛️ System Design", "🧠 Model Architecture", "🔬 Audit Process", "🛠️ Tech Stack"])

    with tab1:
        st.markdown("#### Core Design Philosophy: Separation of Concerns")
        st.write("""
        A key goal for this portfolio project was to demonstrate professional engineering practices. We chose a **decoupled, client-server architecture** over a single, monolithic script.
        - **Frontend (Streamlit):** Its only job is to provide a user interface. It is lightweight and focused on presentation. We chose it for its ability to rapidly build beautiful data apps in pure Python.
        - **Backend (FastAPI):** Its only job is to handle computation. It loads the model and exposes a clean API endpoint. We chose it for its high performance and automatic, interactive documentation, which are industry standards for serving ML models.
        
        This separation makes the system more scalable, maintainable, and robust—key traits for a production-ready application.
        """)
        render_mermaid(system_flowchart)

    with tab2:
        st.markdown("#### The Hierarchical Model Architecture")
        st.write("""
        Early model iterations failed by generating generic, clinically useless text. The solution was a **hierarchical, knowledge-grounded architecture** that mimics a more logical reasoning process. It first identifies clinical topics from the image, then uses those topics to explicitly guide the text generation.
        - **Why a Swin Transformer for the Vision Encoder?** Traditional CNNs have a local bias. For a task like radiology that requires a holistic understanding of the entire image, a Transformer is a better fit. Its **self-attention mechanism** can learn **global, long-range relationships** between distant parts of an X-ray (e.g., connecting heart size to lung findings).
        - **Why BioBERT for the Text Decoder?** Initializing the decoder with random weights would require it to learn the entire English language and complex medical terminology from our small dataset. By using **pre-trained BioBERT weights**, we give the model a massive head-start in understanding clinical language, leading to more fluent and accurate reports.
        """)
        render_mermaid(architecture_flowchart)

    with tab3:
        st.markdown("#### The Audit Process & Final Performance")
        st.write("""
        After training, the model was subjected to a rigorous two-part audit to move beyond simple accuracy metrics and test for trustworthiness. The goal is to provide a transparent view of the model's capabilities and limitations.
        - **Bias & Fairness Audit:** We tested if the model's text generation quality was consistent across different types of diseases. The model showed **no significant performance bias**, maintaining high semantic fluency (BERTScore F1 > 0.84) even for rare findings.
        - **Internal Consistency Audit:** We programmatically scanned each generated report for logical self-contradictions. The final model achieved a **100% consistency rate**, indicating its outputs are logically sound.
        """)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("BERTScore F1 (Semantic Similarity)", "0.88", help="A high score indicates high fluency and relevance.")
        col_b.metric("Micro-Avg ROC-AUC (Clinical Accuracy)", "0.94", help="Shows a strong underlying ability to distinguish between findings.")
        col_c.metric("Consistency Rate", "100%", help="The percentage of reports with no internal logical contradictions.")

    with tab4:
        st.markdown("#### Technology Stack")
        st.markdown("""
        - **Modeling:** PyTorch, Hugging Face Transformers, `timm`
        - **Data Processing:** pandas, NumPy, spaCy
        - **Backend API:** FastAPI
        - **Frontend UI:** Streamlit
        - **Evaluation:** Scikit-learn, `evaluate` (Hugging Face)
        - **Deployment:** GitHub, Hugging Face Spaces (planned)
        """)
# --- Function to render the "Live App" page content ---
# In your frontend/app.py file...

def render_main_app_page():
    st.title("🩺 Auditable Radiology AI Assistant")
    st.write("Upload a chest X-ray to generate a preliminary report or try a pre-computed sample below.")

    API_URL = "http://127.0.0.1:8000/predict"
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded X-Ray', use_container_width=True)
        with col2:
            if st.button('Generate Report', use_container_width=True):
                # (This section for user uploads remains the same)
                with st.spinner('Analyzing the image... This may take a moment on CPU.'):
                    # ... (full API call and results display logic) ...
                    image_bytes = uploaded_file.getvalue()
                    files = {'image_file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                    try:
                        response = requests.post(API_URL, files=files, timeout=300)
                        if response.status_code == 200:
                            results = response.json()
                            st.header("Results")
                            tab1, tab2, tab3 = st.tabs(["📄 Generated Report", "📊 Clinical Predictions", "✅ Audit"])
                            with tab1:
                                st.markdown(results['generated_report'])
                            with tab2:
                                labels_df = pd.DataFrame(results['predicted_labels'])
                                significant_labels = labels_df[labels_df['probability'] > 0.05].sort_values("probability", ascending=False)
                                st.markdown("**Key Findings (Probability > 5%)**")
                                if not significant_labels.empty:
                                    st.dataframe(significant_labels.style.format({"probability": "{:.2%}"}), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No significant clinical findings detected.")
                            with tab3:
                                st.write("This audit checks if the generated report contains logical contradictions.")
                                if results['audit']['is_consistent']:
                                    st.success("The generated report is internally consistent.")
                                else:
                                    st.error("The generated report may contain internal contradictions.")
                        else:
                            st.error(f"Error: Received status code {response.status_code}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    # --- NEW: Sample Example Section ---
    st.markdown("---")
    st.header("Sample Example")
    if st.button("Analyze a Sample X-Ray", use_container_width=True):

        st.subheader("The displayed results are pre-computed!")
        sample_report = "the heart is normal in size. the mediastinum is stable. left - sided chest is again visualized with tip at cavoatrial junction. there is no pneumothorax. numerous bilateral pulmonary nodules have increased in size and number compared to prior study. the dominant nodules have increased in size and number compared to prior study. the dominant nodule / mass in the right midlung is also mildly increased. there is no pleural effusion. interval increase in size and number of innumerable bilateral pulmonary nodules consistent with worsening metastatic disease."
        sample_labels_data = {
            'label': ['Lung Opacity', 'No Finding', 'Lung Lesion', 'Pneumonia'],
            'probability': [0.3660, 0.2888, 0.2436, 0.2220]
        }
        sample_labels_df = pd.DataFrame(sample_labels_data)
        
        # Display the sample image and results in the same layout
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image("xray_chest.png", caption='Sample X-Ray', use_container_width=True)
        with col2:
            st.header("Sample Results")
            tab1, tab2, tab3 = st.tabs(["📄 Generated Report", "📊 Clinical Predictions", "✅ Audit"])
            with tab1:
                st.markdown(sample_report)
            with tab2:
                st.markdown("**Key Findings (Probability > 5%)**")
                st.dataframe(
                    sample_labels_df.style.format({"probability": "{:.2%}"}),
                    use_container_width=True, hide_index=True
                )
            with tab3:
                st.write("This audit checks if the generated report contains logical contradictions.")
                st.success("The generated report is internally consistent.")


# --- Main App Execution with Navigation & Signature ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Live App", "Project Details"])

# --- NEW: Add your name and links to the sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Ujwal Jibhkate**")
st.sidebar.markdown("[Portfolio](https://ujwal.technology/) | [GitHub](https://github.com/ujwal-jibhkate) | [LinkedIn](http://linkedin.com/in/ujwal-jibhkate/)")


if page_selection == "Live App":
    render_main_app_page()
else:
    render_project_details_page()