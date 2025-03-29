import json
import time
import csv
import os
import streamlit as st
import pandas as pd

# ====================== Style Definitions ======================
def set_page_style():
    st.markdown("""
    <style>
        /* Main container styles */
        * {
            font-family: "Times New Roman", sans-serif !important;
        }
        h1, h2 {
            font-size: 16pt !important;
            font-family: "Times New Roman", sans-serif !important;
        }
        h3, h4 {
            font-size: 15pt !important;
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

        /* Button styles */
        .stButton>button {
            font-size: 15px !important;
            background: #3498db;
            color: white;
            border-radius: 8px;
            transition: all 0.3s;
            border: none;
        }
        .stButton>button:hover {
            background: #2980b9;
            transform: translateY(-1px);
        }

        /* Table styles */
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 15px !important;
        }

        /* Sidebar styles */
        .sidebar .sidebar-content {
            background: #f8f9fa;
            padding: 1rem;
            font-size: 15px !important;
        }

        /* Navigation menu font size */
        .stRadio label {
            font-size: 15px !important;
        }

        /* Input field font size */
        .stTextInput input {
            font-size: 15px !important;
        }

        /* Dropdown font size */
        .stSelectbox select {
            font-size: 15px !important;
        }

        /* File uploader font size */
        .stFileUploader label {
            font-size: 15px !important;
        }

        /* Table font size */
        .stDataFrame th, .stDataFrame td {
            font-size: 15px !important;
        }

    </style>
    """, unsafe_allow_html=True)


# ====================== Model Management ======================
def model_management():
    st.markdown("# Model Update Management")

    with st.container():
        # Upload section
        with st.expander("File Upload Area", expanded=True):
            if 'show_upload' not in st.session_state:
                st.session_state.show_upload = False

            if not st.session_state.show_upload:
                if st.button("Upload Files", use_container_width=True):
                    st.session_state.show_upload = True

            if st.session_state.show_upload:
                cols = st.columns(2)
                with cols[0]:
                    uploaded_model = st.file_uploader(
                        "Select Model File (.py)",
                        type=["py"],
                        help="Click to select .py file to upload",
                        key="model_uploader"
                    )
                with cols[1]:
                    uploaded_weight = st.file_uploader(
                        "Select Weight File (.pth)",
                        type=["pth"],
                        help="Click to select .pth file to upload",
                        key="weight_uploader"
                    )

                if st.button("Start Update", use_container_width=True):
                    try:
                        if uploaded_model:
                            model_path = os.path.join("./model", uploaded_model.name)
                            with open(model_path, "wb") as f:
                                f.write(uploaded_model.getbuffer())
                            st.success(f"Model file {uploaded_model.name} uploaded successfully!")

                        if uploaded_weight:
                            weight_path = os.path.join("./weights", uploaded_weight.name)
                            with open(weight_path, "wb") as f:
                                f.write(uploaded_weight.getbuffer())
                            st.success(f"Weight file {uploaded_weight.name} uploaded successfully!")

                        if not uploaded_model and not uploaded_weight:
                            st.warning("Please select files to upload")

                        if uploaded_model and uploaded_weight:
                            st.session_state.show_upload = False
                            st.rerun()

                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")

        # File listing section
        with st.expander("Current Files", expanded=True):
            st.markdown("**Model Files**")
            model_files = [f for f in os.listdir("./model") if f.endswith(".py")]
            if model_files:
                for f in model_files:
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"{f}")
                    if cols[1].button("Delete", key=f"del_model_{f}"):
                        try:
                            os.remove(os.path.join("./model", f))
                            st.success(f"{f} deleted successfully")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Deletion failed: {str(e)}")
            else:
                st.info("No model files found")

            st.markdown("**Weight Files**")
            weight_files = [f for f in os.listdir("./weights") if f.endswith(".pth")]
            if weight_files:
                for f in weight_files:
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"{f}")
                    if cols[1].button("Delete", key=f"del_weight_{f}"):
                        try:
                            os.remove(os.path.join("./weights", f))
                            st.success(f"{f} deleted successfully")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Deletion failed: {str(e)}")
            else:
                st.info("No weight files found")


# ====================== User Management ======================
def user_management():
    st.markdown("# User Management")

    def load_users():
        if not os.path.exists("./csv/users.csv"):
            with open("./csv/users.csv", "w", newline="") as f:
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
        with open("users.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([username, password, time.strftime("%Y-%m-%d %H:%M:%S")])

    users = load_users()

    operation = st.radio("Select Operation", ["Add User", "Delete User", "Change Password", "Search User"],
                         horizontal=True,
                         label_visibility="hidden")

    with st.form("user_form"):
        if operation == "Add User":
            st.markdown("### Add New User")
            new_user = st.text_input("Username", placeholder="Enter username")
            new_pass = st.text_input("Password", type="password", placeholder="Enter password")

        elif operation == "Delete User":
            st.markdown("### Delete User")
            del_user = st.selectbox("Select User", list(users.keys()))

        elif operation == "Change Password":
            st.markdown("### Change Password")
            modify_user = st.selectbox("Select User", list(users.keys()))
            new_pass = st.text_input("New Password", type="password", placeholder="Enter new password")

        elif operation == "Search User":
            st.markdown("### User Search")
            search_user = st.text_input("Username", placeholder="Enter username to search")

        if st.form_submit_button(f"Confirm {operation}", use_container_width=True):
            if operation == "Add User":
                if new_user and new_pass:
                    if new_user in users:
                        st.error("Username already exists")
                    else:
                        save_user(new_user, new_pass)
                        st.success(f"User {new_user} added successfully")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("Please enter both username and password")

            elif operation == "Delete User":
                users.pop(del_user)
                with open("./csv/users.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["username", "password", "login_time"])
                    for username, (password, login_time) in users.items():
                        writer.writerow([username, password, login_time])
                st.success(f"User {del_user} deleted successfully")
                time.sleep(1)
                st.rerun()

            elif operation == "Change Password":
                users[modify_user] = (new_pass, users[modify_user][1])
                with open("./csv/users.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["username", "password", "login_time"])
                    for username, (password, login_time) in users.items():
                        writer.writerow([username, password, login_time])
                st.success(f"Password for {modify_user} changed successfully")
                time.sleep(1)
                st.rerun()

            elif operation == "Search User":
                if search_user in users:
                    st.success(f"User {search_user} found")
                    st.json({
                        "Username": search_user,
                        "Password": users[search_user][0],
                        "Last Login": users[search_user][1]
                    })
                else:
                    st.error("User not found")

    st.markdown("### User List")
    if users:
        df = pd.DataFrame(
            [(k, v[0], v[1]) for k, v in users.items()],
            columns=["Username", "Password", "Last Login"]
        )
        st.dataframe(
            df,
            column_config={
                "Password": st.column_config.TextColumn(
                    "Password",
                    help="User password",
                    width="medium"
                )
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No users found")


# ====================== Result Management ======================
def result_management():
    st.markdown("# Identification Results Management")

    def load_history():
        output_folder = "./csv/"
        output_file = os.path.join(output_folder, "history.csv")
        history = []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    history.append(row)
        return history

    def save_history(history):
        output_folder = "./csv/"
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "history.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Username", "Image Source", "Primary Class", "Primary Probability",
                            "Secondary Class 1", "Secondary Probability 1",
                            "Secondary Class 2", "Secondary Probability 2", "Timestamp"])
            for row in history:
                writer.writerow(row)

    history = load_history()

    operation = st.radio("Select Operation", ["View Records", "Delete Records"],
                         horizontal=True,
                         label_visibility="hidden")

    if operation == "View Records":
        search_key = st.text_input("Search records by keyword (username, disease type, etc.)")
        if search_key:
            filtered_history = [row for row in history if any(search_key.lower() in str(cell).lower() for cell in row)]
        else:
            filtered_history = history

        if filtered_history:
            df = pd.DataFrame(filtered_history,
                            columns=["Username", "Image Source", "Primary Class", "Primary Probability",
                                    "Secondary Class 1", "Secondary Probability 1",
                                    "Secondary Class 2", "Secondary Probability 2", "Timestamp"])
            st.dataframe(
                df,
                column_config={
                    "Username": "Username",
                    "Image Source": "Image Source",
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
                    "Timestamp": "Timestamp"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No matching records found")

    elif operation == "Delete Records":
        if history:
            df = pd.DataFrame(history,
                            columns=["Username", "Image Source", "Primary Class", "Primary Probability",
                                    "Secondary Class 1", "Secondary Probability 1",
                                    "Secondary Class 2", "Secondary Probability 2", "Timestamp"])
            selected_index = st.selectbox("Select record to delete", df.index)

            if st.button("Delete Record"):
                history.pop(selected_index)
                save_history(history)
                st.success("Record deleted successfully")
                time.sleep(1)
                st.rerun()
        else:
            st.info("No records available for deletion")

# ====================== Main Application ======================
def main():
    set_page_style()

    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        with st.container():
            st.markdown("<h1 style='text-align: center;'>Dragon Fruit Stem Disease Identification Management System</h1>", unsafe_allow_html=True)
            cols = st.columns([1, 2, 1])
            with cols[1]:
                with st.form("login_form"):
                    username = st.text_input("Admin Username")
                    password = st.text_input("Password", type="password")
                    if st.form_submit_button("Login", use_container_width=True):
                        if username == "admin" and password == "admin123":
                            st.session_state.admin_logged_in = True
                            st.success("Login successful")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid username or password")

    if st.session_state.admin_logged_in:
        with st.sidebar:
            st.markdown("<h1 style='text-align: center;'>Dragon Fruit Stem Disease IM System</h1>", unsafe_allow_html=True)
            page = st.radio(
                "Navigation",
                ["User Management", "Result Management", "Model Management"],
                index=0,
                format_func=lambda x: {
                    "User Management": "User Management",
                    "Result Management": "Result Management",
                    "Model Management": "Model Management"
                }[x]
            )

            if st.button("Logout", use_container_width=True):
                st.session_state.admin_logged_in = False
                st.success("Logged out successfully")
                time.sleep(1)
                st.rerun()

        if page == "User Management":
            user_management()
        elif page == "Result Management":
            result_management()
        elif page == "Model Management":
            model_management()


if __name__ == "__main__":
    main()
