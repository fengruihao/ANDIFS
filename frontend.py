import json
import time
import csv
import os
from PIL import Image
import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from torchvision import transforms
import streamlit as st
import cv2
import pandas as pd
from model.ANDIFS import ghostnetv2

# Load class indices
def load_class_indices(json_path):
    assert os.path.exists(json_path), f"File: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    return class_indict


# Load methods data
def load_methods_data(csv_path):
    assert os.path.exists(csv_path), f"File: '{csv_path}' does not exist."
    methods_data = {}
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            class_name = row[1]
            characteristics = row[2]
            treatment_methods = row[3]
            methods_data[class_name] = (class_name, characteristics, treatment_methods)
    return methods_data


# Transform input image
def transform_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    image = data_transform(image)
    image = torch.unsqueeze(image, dim=0)
    return image


# Predict image class
def predict_image_class(model, image, class_indict, device):
    start_time = time.time()
    with torch.no_grad():
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        top3_prob, top3_indices = torch.topk(predict, 3)
    end_time = time.time()

    time_taken = end_time - start_time

    top3_classes = [class_indict[str(idx.item())] for idx in top3_indices]
    top3_probabilities = [float(prob * 100) for prob in top3_prob.numpy()]

    return top3_classes, top3_probabilities, time_taken


# Save result image
def save_result_image(image, top3_classes, top3_probabilities, time_taken, output_path):
    font_path = "C:\\Windows\\Fonts\\times.ttf"
    prop = FontProperties(fname=font_path, size=20)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(12, 6))

    ax_img.imshow(image)
    ax_img.axis('off')
    fig.text(0.28, 0.08, "Uploaded Image", fontproperties=prop, ha='center', va='top')

    for i, (cla, prob) in enumerate(zip(top3_classes, top3_probabilities)):
        ax.barh(i, prob, color=colors[i % len(colors)], height=0.8, edgecolor='red', label="Probability (%)")
    ax.barh(len(top3_classes), -time_taken * 1000, color='lightblue', height=0.8, edgecolor='blue',
            label="Time_taken (ms)")

    ax.set_xlim(-time_taken * 1000 * 1.2, max(top3_probabilities) * 1.2)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.yaxis.set_visible(False)

    ax.tick_params(axis='x', labelsize=12)
    xticks = ax.get_xticks()
    new_xticks = [abs(tick) if tick < 0 else tick for tick in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(tick)}' for tick in new_xticks], fontproperties=prop)

    for i, cla in enumerate(top3_classes):
        ax.text(-time_taken * 1000 * 0.05, i, cla, va='center', ha='right', fontproperties=prop, rotation=90,
                color='red')
    ax.text(time_taken * 1000 * 0.05, len(top3_classes), 'Processing Time', va='center', ha='left', fontproperties=prop,
            rotation=90, color='blue')

    ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

    ax.text(-time_taken * 1000 * 0.3, len(top3_classes), f'{time_taken * 1000:.1f} ms', va='center', ha='right',
            fontproperties=prop, color='blue')
    for i, prob in enumerate(top3_probabilities):
        ax.text(prob * 0.5, i, f'{prob:.3f}%', va='center', ha='left', fontproperties=prop, color='red')

    ax.axhline(y=2.5, color='black', linewidth=1, linestyle='-')
    fig.text(0.62, 0.22, "Top 3 Predictions", fontproperties=prop, color='red', rotation='vertical')
    fig.text(0.55, 0.03, "Processing Time (ms)", fontproperties=prop, color='blue')
    fig.text(0.8, 0.03, "Probability (%)", fontproperties=prop, color='red')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# Save captured image
def save_captured_image(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


# User management
def load_users():
    if not os.path.exists("./csv/users.csv"):
        with open("/csv/users.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password", "login_time"])
    users = {}
    with open("./csv/users.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            users[row[0]] = (row[1], row[2])
    return users


def save_user(username, password):
    with open("./csv/users.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, password, time.strftime("%Y-%m-%d %H:%M:%S")])


# Load history
def load_history():
    history = []
    if os.path.exists("./csv/history.csv"):
        with open("./csv/history.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[0] == st.session_state.username:
                    history.append(row)
    return history


def save_to_history(username, image_name, classes, probs, time_taken):
    with open("./csv/history.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            username,
            image_name,
            classes[0], probs[0],
            classes[1], probs[1],
            classes[2], probs[2],
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])


def perform_analysis(image):
    with st.spinner("AI is analyzing..."):
        try:
            # Transform image
            image_tensor = transform_image(image)

            # Perform prediction
            top3_classes, top3_probabilities, time_taken = predict_image_class(
                st.session_state.model,
                image_tensor,
                st.session_state.class_indict,
                st.session_state.device
            )

            # Get class_indict
            class_indict = st.session_state.class_indict
            # Convert English classes to disease names
            methods_data = st.session_state.methods_data
            top3_classes_en = [class_indict[str(idx.item())] for idx in torch.topk(torch.softmax(torch.squeeze(st.session_state.model(image_tensor.to(st.session_state.device))).cpu(), dim=0), 3)[1]]
            top3_classes_names = [methods_data.get(en_class, (en_class, '', ''))[0] for en_class in top3_classes_en]

            # Save English and class names to session_state
            st.session_state.class_name_en = top3_classes_en[0]
            st.session_state.class_name = top3_classes_names[0]
            st.session_state.top3_classes = top3_classes_names
            st.session_state.top3_probabilities = top3_probabilities
            st.session_state.time_taken = time_taken

            # Mark analysis as complete
            st.session_state.analysis_done = True

            # Get image name
            if st.session_state.captured_image:
                images_folder = './images'
                image_name = f'captured_{int(time.time())}.jpg'
            else:
                image_name = st.session_state.uploaded_file.name

            # Save to history
            save_to_history(st.session_state.username, image_name, top3_classes_names, top3_probabilities, time_taken)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")


def show_analysis_result():
    """Display analysis results"""
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: -3rem;">
        <h3 style="color: #2c3e50; margin-bottom: 1rem;">Disease Identification Results</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 18px; color: #3498db;">Most Likely</div>
                <div style="font-size: 18px; font-weight: bold; margin: 0.5rem 0;">{0}</div>
                <div style="color: #27ae60; font-size: 18px;">{1:.2f}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 18px; color: #f39c12;">Second</div>
                <div style="font-size: 18px; font-weight: bold; margin: 0.5rem 0;">{2}</div>
                <div style="color: #27ae60; font-size: 18px;">{3:.2f}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 18px; color: #e74c3c;">Possible</div>
                <div style="font-size: 18px; font-weight: bold; margin: 0.5rem 0;">{4}</div>
                <div style="color: #27ae60; font-size: 18px;">{5:.2f}%</div>
            </div>
        </div>
        <div style="margin-top: 1rem; color: #7f8c8d; font-size: 18px;">
            Processing Time: {6:.2f}ms
        </div>
    </div>
    """.format(
        st.session_state.top3_classes[0], st.session_state.top3_probabilities[0],
        st.session_state.top3_classes[1], st.session_state.top3_probabilities[1],
        st.session_state.top3_classes[2], st.session_state.top3_probabilities[2],
        st.session_state.time_taken * 1000
    ), unsafe_allow_html=True)

    # Add disease guide button
    if st.button("Disease Treatment Guide", key="disease_guide_btn", use_container_width=True):
        st.session_state.show_guide = True
        st.session_state.analysis_done = False  # Hide analysis results

def show_disease_guide():
    """Display disease treatment guide"""
    if st.session_state.class_name_en in st.session_state.methods_data:
        disease_info = st.session_state.methods_data[st.session_state.class_name_en]
        class_name, characteristics, treatment_methods = disease_info

        class_name_list = class_name.split('\n')
        for cat in class_name_list:
            if cat.strip():
                st.markdown(f"**Disease Class**: {cat.strip()}", unsafe_allow_html=True)

        characteristics_list = characteristics.split('\n')
        for idx, char in enumerate(characteristics_list):
            if char.strip():
                if idx == len(characteristics_list) - 1:
                    st.markdown(f"**Identification Features**: {char.strip()} (Qin, 2020 and Zheng, 2023)", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {char.strip()}")

        st.markdown(f"**Recommended Treatment**<br>", unsafe_allow_html=True)
        treatment_methods_list = treatment_methods.split('\n')
        for idx, method in enumerate(treatment_methods_list):
            if method.strip():
                if idx == len(treatment_methods_list) - 1:
                    st.markdown(f"{method.strip()} (Qin, 2020 and Zheng, 2023)")
                else:
                    st.markdown(f"{method.strip()}")

        st.markdown(f"**References**<br>Qin, Y. (2020) <i>Pitaya Cultivation Techniques</i>. Guangzhou: Guangdong Science Press.<br>Zheng, W. (2023) <i>Pitaya Disease Control Atlas</i>. Beijing: China Agriculture Press.", unsafe_allow_html=True)

    # Add back button
    if st.button("Back to Results", key="back_to_result_btn", use_container_width=True):
        st.session_state.show_guide = False
        st.session_state.analysis_done = True  # Show analysis results again


def main():
    # Custom CSS styles
    st.markdown("""
        <style>
        * {
            font-family: "Times New Roman", sans-serif !important;
        }
        h1, h2 {
            font-size: 18pt !important;
            font-family: "Times New Roman", sans-serif !important;
        }
        h3, h4 {
            font-size: 16pt !important;
            font-family: "Times New Roman", sans-serif !important;
        }
        h5, h6 {
            font-size: 14pt !important;
            font-family: "Times New Roman", sans-serif !important;
        }
        body {
            font-size: 15pt !important;
        }
      .main {padding: 2rem;}
      .title {color: #2c3e50; text-align: center; margin-bottom: 2rem;}
      .section-title {color: #3498db; margin: 1.5rem 0 1rem;}
        /* Button styles with light gray border */
      .stButton>button {
            background: white;
            color: black;
            border: 1px solid #d3d3d3;
            border-radius: 8px;
            padding: 0.8rem 2rem;
        }
      .stButton>button:hover {
            background: #f0f0f0;
        }
      .result-card {padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0;}
      .history-table {margin-top: 1rem;}
      .camera-box {border: 2px dashed #bdc3c7; border-radius: 10px; padding: 1rem; margin: 1rem 0;}
      .header {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 2rem;
        }
      .image-result-container {
            display: flex;
            align-items: flex-start;
        }
      .references {
            color: #7f8c8d;
        }   
    </style>
    """, unsafe_allow_html=True)

    # Initialize session_state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'class_name' not in st.session_state:
        st.session_state.class_name = None
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if'show_history' not in st.session_state:
        st.session_state.show_history = False
    if 'upload_button_clicked' not in st.session_state:
        st.session_state.upload_button_clicked = False
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if'show_guide' not in st.session_state:
        st.session_state.show_guide = False
    if'show_camera' not in st.session_state:
        st.session_state.show_camera = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'

    # Initialize model, class indices, device and methods data
    if'model' not in st.session_state:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        st.session_state.model = ghostnetv2(num_classes=13, width=1.0, dropout=0.2, args=None).to(device)
        st.session_state.model.eval()
        if torch.cuda.is_available():
            st.session_state.model.load_state_dict(torch.load('./weights/ANDIFS-99.pth'))
        else:
            st.session_state.model.load_state_dict(
                torch.load('./weights/ANDIFS-99.pth', map_location=torch.device('cpu')))

    if 'class_indict' not in st.session_state:
        st.session_state.class_indict = load_class_indices('./class_indices.json')
    if 'device' not in st.session_state:
        st.session_state.device = torch.device("cpu")
        st.session_state.model.to(st.session_state.device)
    if'methods_data' not in st.session_state:
        st.session_state.methods_data = load_methods_data('./csv/Methods.csv')

    # User login interface
    if not st.session_state.logged_in:
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Dragon Fruit Stem Disease Identification System</h1>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        with st.form("auth_form"):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                cols = st.columns(2)
                with cols[0]:
                    if st.form_submit_button("Login", use_container_width=True):
                        if username and password:
                            users = load_users()
                            if username in users and users[username][0] == password:
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.success(f"Welcome back, {username}!")
                                st.rerun()
                            else:
                                st.error("Invalid username or password")
                with cols[1]:
                    if st.form_submit_button("Register", use_container_width=True):
                        if username and password:
                            users = load_users()
                            if username in users:
                                st.error("Username already exists")
                            else:
                                save_user(username, password)
                                st.success("Registration successful! Please login")

    # Main interface
    if st.session_state.logged_in:
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Dragon Fruit Stem Disease Identification System</h1>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sidebar layout for navigation
        sidebar = st.sidebar
        sidebar.markdown('<h3>Navigation</h3>', unsafe_allow_html=True)
        if sidebar.button("Home", use_container_width=True):
            st.session_state.current_page = 'Home'
            st.session_state.captured_image = None
            st.session_state.uploaded_file = None
            st.session_state.class_name = None
            st.session_state.show_history = False
            st.session_state.upload_button_clicked = False
            st.session_state.file_uploaded = False
            st.session_state.analysis_done = False
            st.session_state.show_guide = False
            st.session_state.show_camera = False

        if sidebar.button("Image Import", use_container_width=True):
            st.session_state.current_page = 'Image Import'
            st.session_state.captured_image = None
            st.session_state.uploaded_file = None
            st.session_state.upload_button_clicked = False
            st.session_state.file_uploaded = False
            st.session_state.analysis_done = False
            st.session_state.show_guide = False
            st.session_state.show_camera = False

        if sidebar.button("Disease Identification", use_container_width=True):
            st.session_state.current_page = 'Disease Identification'
            if not st.session_state.captured_image and not st.session_state.uploaded_file:
                st.warning("Please upload or capture an image first.")

        if sidebar.button("Disease Treatment", use_container_width=True):
            if st.session_state.analysis_done:
                st.session_state.show_guide = True
                st.session_state.analysis_done = False
            else:
                st.warning("Please complete disease identification first.")

        if sidebar.button("History", use_container_width=True):
            st.session_state.current_page = 'History'
            st.session_state.show_history = True

        if sidebar.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

        if st.session_state.current_page == 'Home':
            st.markdown(
                '<p style="text-align: center;">Welcome to the Dragon Fruit Stem Disease Identification System! Please select a function from the navigation sidebar.</p>',
                unsafe_allow_html=True
            )

        elif st.session_state.current_page == 'Image Import':
            st.markdown("""
                            <style>
                            /* Adjust label text size */
                            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                                font-size: 18px !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["File Upload", "Live Capture"])
            with tab1:
                if st.button('Click to Upload File'):
                    # Click upload button to show upload component
                    st.session_state.upload_button_clicked = True
                    st.session_state.file_uploaded = False

                if st.session_state.upload_button_clicked and not st.session_state.file_uploaded:
                    # Create a placeholder for the upload component
                    upload_placeholder = st.empty()
                    # Add custom English prompt
                    upload_placeholder.markdown("Please select or drag and drop image files in the gray box below. Maximum file size: 200MB. Supported formats: JPG/PNG")
                    uploaded_file = upload_placeholder.file_uploader(
                        "",  # Clear default English prompt
                        type=["jpg", "jpeg", "png"],
                        key="file_uploader",
                        label_visibility="collapsed"
                    )
                    if uploaded_file is not None:
                        # Clear previously captured image
                        st.session_state.captured_image = None
                        # Store uploaded file
                        st.session_state.uploaded_file = uploaded_file
                        # Upload successful, hide upload component
                        st.session_state.file_uploaded = True
                        st.session_state.upload_button_clicked = False
                        # Clear placeholder to close upload component
                        upload_placeholder.empty()

                if st.session_state.file_uploaded:
                    st.markdown('<h3>Uploaded Image</h3>', unsafe_allow_html=True)
                    image = Image.open(st.session_state.uploaded_file)
                    st.image(image, use_container_width=False, width=200)  # Uniform display width

            with tab2:
                st.markdown("Click the button below to open the camera for capture")
                if st.button("Open Camera", key="open_camera_btn"):
                    st.session_state.show_camera = True
                if st.session_state.show_camera:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Unable to open camera. Please check camera device or permissions.")
                    else:
                        stframe = st.empty()
                        # Create two-column layout
                        col1, col2 = st.columns(2)
                        with col1:
                            capture_button = st.button("Capture Image", key="capture_image_btn")
                        with col2:
                            close_button = st.button("Close Camera", key="close_camera_btn")
                        while True:
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                stframe.image(frame, width=int(frame.shape[1] * 0.5), use_container_width=False)
                                if capture_button:
                                    captured_image = Image.fromarray(frame)
                                    images_folder = './images'
                                    os.makedirs(images_folder, exist_ok=True)
                                    image_path = os.path.join(images_folder, f'captured_{int(time.time())}.jpg')
                                    save_captured_image(captured_image, image_path)
                                    st.session_state.captured_image = captured_image
                                    st.success("Image captured successfully!")
                                    cap.release()
                                    break
                                if close_button:
                                    cap.release()
                                    st.session_state.show_camera = False
                                    break
                            else:
                                st.error("Unable to read camera frame. Please check camera device.")
                                cap.release()
                                break

        elif st.session_state.current_page == 'Disease Identification':
            if st.session_state.captured_image or st.session_state.uploaded_file:
                image = st.session_state.captured_image if st.session_state.captured_image else Image.open(
                    st.session_state.uploaded_file)
                # Right side interface divided into two columns
                col1, col2 = st.columns([1, 2.5])
                st.markdown("""
                                    <style>
                                       /* New styles to eliminate gaps */
                                       .compact-container {
                                            margin: 0;
                                            padding: 0;
                                        }
                                       .no-spacing {
                                            margin-top: 0 !important;
                                            margin-bottom: 0 !important;
                                            padding: 0 !important;
                                        }
                                       .tight-button {
                                            margin-bottom: -1rem !important;
                                        }
                                    </style>
                                """, unsafe_allow_html=True)
                with col1:
                    st.markdown("\n\n\n\n\n")
                    st.markdown('<h3 class="image-container">Current Image for Identification</h3>', unsafe_allow_html=True)
                    st.image(image, width=200)
                with col2:
                    if not st.session_state.show_guide:
                        st.markdown('<div class="no-spacing">', unsafe_allow_html=True)
                        if st.button("Start Smart Identification", key="analyze_btn",
                                     use_container_width=True,
                                     help="Click to start analysis"):
                            perform_analysis(image)
                        st.markdown('</div>', unsafe_allow_html=True)
                    if st.session_state.get('analysis_done', False):
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        show_analysis_result()
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif st.session_state.get('show_guide', False):
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.markdown('<h3 class="guide-title">Disease Treatment Guide</h3>', unsafe_allow_html=True)
                        show_disease_guide()
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please upload or capture an image first.")

        elif st.session_state.current_page == 'Disease Treatment':
            if st.session_state.analysis_done:
                show_disease_guide()
            else:
                st.warning("Please complete disease identification first.")

        elif st.session_state.current_page == 'History':
            with st.container():
                st.markdown('<h3>Identification History</h3>', unsafe_allow_html=True)
                history = load_history()
                if history:
                    df = pd.DataFrame(history,
                                      columns=["Username", "Image Name", "Primary Class", "Primary Probability",
                                               "Secondary Class 1", "Secondary Probability 1",
                                               "Secondary Class 2", "Secondary Probability 2", "Identification Time"])
                    st.dataframe(
                        df,
                        column_config={
                            "Username": "Username",
                            "Image Name": "Image Name",
                            "Primary Class": "Disease Type",
                            "Primary Probability": st.column_config.ProgressColumn(
                                "Primary Probability",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100
                            ),
                            "Secondary Class 1": "Secondary Type 1",
                            "Secondary Probability 1": st.column_config.ProgressColumn(
                                "Secondary Probability 1",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100
                            ),
                            "Secondary Class 2": "Secondary Type 2",
                            "Secondary Probability 2": st.column_config.ProgressColumn(
                                "Secondary Probability 2",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100
                            ),
                            "Identification Time": "Identification Time"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("No identification history available")


if __name__ == "__main__":
    main()
