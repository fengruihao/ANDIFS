import json
import time
import csv
import os
import streamlit as st
import pandas as pd

# ====================== 样式定义 ======================
def set_page_style():
    st.markdown("""
    <style>
        /* 主容器样式 */
        * {
            font-family: "Times New Roman", "SimSun", sans-serif !important;
        }
        h1, h2 {
            font-size: 16pt !important;
            font-family: "Times New Roman", "SimSun", sans-serif !important;
        }
        h3, h4 {
            font-size: 15pt !important;
            font-family: "Times New Roman", "SimSun", sans-serif !important;
        }
        h5, h6 {
            font-size: 14pt !important;
            font-family: "Times New Roman", "SimSun", sans-serif !important;
        }
        body {
            font-size: 15pt !important;
        }
        .main {padding: 2rem;}

        /* 按钮样式 */
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

        /* 表格样式 */
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 15px !important;
        }

        /* 侧边栏样式 */
        .sidebar .sidebar-content {
            background: #f8f9fa;
            padding: 1rem;
            font-size: 15px !important;
        }

        /* 导航菜单栏字体放大 */
        .stRadio label {
            font-size: 15px !important;
        }

        /* 侧边栏标题字体放大 */
        .sidebar h1 {
            font-size: 15px !important;
        }

        /* 输入框字体放大 */
        .stTextInput input {
            font-size: 15px !important;
        }

        /* 选择框字体放大 */
        .stSelectbox select {
            font-size: 15px !important;
        }

        /* 文件上传字体放大 */
        .stFileUploader label {
            font-size: 15px !important;
        }

        /* 表格字体放大 */
        .stDataFrame th, .stDataFrame td {
            font-size: 15px !important;
        }

    </style>
    """, unsafe_allow_html=True)


# ====================== 模型管理 ======================
def model_management():
    st.markdown("# 模型升级管理")

    with st.container():
        # 上传模块
        with st.expander("文件上传区", expanded=True):
            # 添加“上传文件”按钮
            if 'show_upload' not in st.session_state:
                st.session_state.show_upload = False

            if not st.session_state.show_upload:
                if st.button("上传文件", use_container_width=True):
                    st.session_state.show_upload = True

            # 显示文件上传组件
            if st.session_state.show_upload:
                cols = st.columns(2)
                with cols[0]:
                    uploaded_model = st.file_uploader(
                        "选择模型文件 (.py)",
                        type=["py"],
                        help="请点击灰色方框选择要上传的.py文件",
                        key="model_uploader"
                    )
                with cols[1]:
                    uploaded_weight = st.file_uploader(
                        "选择权重文件 (.pth)",
                        type=["pth"],
                        help="请点击灰色方框，选择要上传的.pth文件",
                        key="weight_uploader"
                    )

                if st.button("开始上传更新", use_container_width=True):
                    try:
                        if uploaded_model:
                            model_path = os.path.join("./model", uploaded_model.name)
                            with open(model_path, "wb") as f:
                                f.write(uploaded_model.getbuffer())
                            st.success(f"模型文件 {uploaded_model.name} 上传成功！")

                        if uploaded_weight:
                            weight_path = os.path.join("./weights", uploaded_weight.name)
                            with open(weight_path, "wb") as f:
                                f.write(uploaded_weight.getbuffer())
                            st.success(f"权重文件 {uploaded_weight.name} 上传成功！")

                        if not uploaded_model and not uploaded_weight:
                            st.warning("请先选择要上传的文件")

                        # 如果两个文件都上传完成，隐藏上传组件
                        if uploaded_model and uploaded_weight:
                            st.session_state.show_upload = False
                            st.rerun()

                    except Exception as e:
                        st.error(f"文件上传失败: {str(e)}")

        # 文件展示模块
        with st.expander("当前文件列表", expanded=True):
            st.markdown("**模型文件**")
            model_files = [f for f in os.listdir("./model") if f.endswith(".py")]
            if model_files:
                for f in model_files:
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"{f}")
                    if cols[1].button("删除", key=f"del_model_{f}"):
                        try:
                            os.remove(os.path.join("./model", f))
                            st.success(f"{f} 删除成功")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {str(e)}")
            else:
                st.info("暂无模型文件")

            st.markdown("**权重文件**")
            weight_files = [f for f in os.listdir("./weights") if f.endswith(".pth")]
            if weight_files:
                for f in weight_files:
                    cols = st.columns([3, 1])
                    cols[0].markdown(f"{f}")
                    if cols[1].button("删除", key=f"del_weight_{f}"):
                        try:
                            os.remove(os.path.join("./weights", f))
                            st.success(f"{f} 删除成功")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {str(e)}")
            else:
                st.info("暂无权重文件")


# ====================== 用户管理 ======================
def user_management():
    st.markdown("# <span style='font-size: 18px;'>用户管理</span>", unsafe_allow_html=True)

    # 加载用户数据
    def load_users():
        if not os.path.exists("users.csv"):
            with open("users.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["username", "password", "login_time"])
        users = {}
        with open("users.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                users[row[0]] = (row[1], row[2])
        return users

    # 保存用户
    def save_user(username, password):
        with open("users.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([username, password, time.strftime("%Y-%m-%d %H:%M:%S")])

    users = load_users()

    # 操作选择
    operation = st.radio("选择操作", ["新增用户", "删除用户", "修改密码", "查询用户"],
                         horizontal=True,
                         label_visibility="hidden")

    # 操作表单
    with st.form("user_form"):
        if operation == "新增用户":
            st.markdown("### <span style='font-size: 15px;'>新增用户</span>", unsafe_allow_html=True)
            new_user = st.text_input("用户名", placeholder="输入新用户名")
            new_pass = st.text_input("密码", type="password", placeholder="输入密码")

        elif operation == "删除用户":
            st.markdown("### <span style='font-size: 15px;'>删除用户</span>", unsafe_allow_html=True)
            del_user = st.selectbox("选择用户", list(users.keys()))

        elif operation == "修改密码":
            st.markdown("### <span style='font-size: 15px;'>修改密码</span>", unsafe_allow_html=True)
            modify_user = st.selectbox("选择用户", list(users.keys()))
            new_pass = st.text_input("新密码", type="password", placeholder="输入新密码")

        elif operation == "查询用户":
            st.markdown("### <span style='font-size: 15px;'>用户查询</span>", unsafe_allow_html=True)
            search_user = st.text_input("用户名", placeholder="输入要查询的用户名")

        if st.form_submit_button(f"确认{operation}", use_container_width=True):
            if operation == "新增用户":
                if new_user and new_pass:
                    if new_user in users:
                        st.error("用户名已存在")
                    else:
                        save_user(new_user, new_pass)
                        st.success(f"用户 {new_user} 添加成功")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("请输入用户名和密码")

            elif operation == "删除用户":
                users.pop(del_user)
                with open("users.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["username", "password", "login_time"])
                    for username, (password, login_time) in users.items():
                        writer.writerow([username, password, login_time])
                st.success(f"用户 {del_user} 删除成功")
                time.sleep(1)
                st.rerun()

            elif operation == "修改密码":
                users[modify_user] = (new_pass, users[modify_user][1])
                with open("users.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["username", "password", "login_time"])
                    for username, (password, login_time) in users.items():
                        writer.writerow([username, password, login_time])
                st.success(f"用户 {modify_user} 密码修改成功")
                time.sleep(1)
                st.rerun()

            elif operation == "查询用户":
                if search_user in users:
                    st.success(f"找到用户 {search_user}")
                    st.json({
                        "用户名": search_user,
                        "密码": users[search_user][0],
                        "最后登录": users[search_user][1]
                    })
                else:
                    st.error("用户不存在")

    # 用户表格
    st.markdown("### <span style='font-size: 15px;'>用户列表</span>", unsafe_allow_html=True)
    if users:
        df = pd.DataFrame(
            [(k, v[0], v[1]) for k, v in users.items()],
            columns=["用户名", "密码", "最后登录"]
        )
        st.dataframe(
            df,
            column_config={
                "密码": st.column_config.TextColumn(
                    "密码",
                    help="用户登录密码",
                    width="medium"
                )
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("暂无用户信息")


# ====================== 识别结果信息管理 ======================
def result_management():
    st.markdown("# 识别结果信息管理")

    # 加载历史记录
    def load_history():
        output_folder = "./predictions/"
        output_file = os.path.join(output_folder, "history.csv")
        history = []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    history.append(row)
        return history

    # 保存历史记录
    def save_history(history):
        output_folder = "./predictions/"
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "history.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["用户名", "图像来源", "主要类别", "主要类别概率", "次要类别1", "次要类别1概率", "次要类别2", "次要类别2概率", "识别时间"])
            for row in history:
                writer.writerow(row)

    history = load_history()

    # 操作选择
    operation = st.radio("选择操作", ["查询记录", "删除记录"],
                         horizontal=True,
                         label_visibility="hidden")

    # 查询功能
    if operation == "查询记录":
        search_key = st.text_input("输入关键词查询记录（如用户名、病害类型等）")
        if search_key:
            filtered_history = [row for row in history if any(search_key.lower() in str(cell).lower() for cell in row)]
        else:
            filtered_history = history

        # 显示历史记录
        if filtered_history:
            df = pd.DataFrame(filtered_history, columns=["用户名", "图像来源", "主要类别", "主要类别概率", "次要类别1", "次要类别1概率", "次要类别2", "次要类别2概率", "识别时间"])
            st.dataframe(
                df,
                column_config={
                    "用户名": "用户名",
                    "图像来源": "图像来源",
                    "主要类别": "病害类型",
                    "主要类别概率": st.column_config.ProgressColumn(
                        "主要类别概率",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "次要类别1": "次要病害类型1",
                    "次要类别1概率": st.column_config.ProgressColumn(
                        "次要类别1概率",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "次要类别2": "次要病害类型2",
                    "次要类别2概率": st.column_config.ProgressColumn(
                        "次要类别2概率",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "识别时间": "识别时间"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("暂无符合条件的识别记录")

    # 删除功能
    elif operation == "删除记录":
        if history:
            df = pd.DataFrame(history, columns=["用户名", "图像来源", "主要类别", "主要类别概率", "次要类别1", "次要类别1概率", "次要类别2", "次要类别2概率", "识别时间"])
            selected_index = st.selectbox("选择要删除的记录", df.index)

            if st.button("删除记录"):
                history.pop(selected_index)
                save_history(history)
                st.success("记录删除成功")
                time.sleep(1)
                st.rerun()
        else:
            st.info("暂无记录可供删除")

# ====================== 主程序 ======================
def main():
    set_page_style()

    # 初始化session状态
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    # 登录界面
    if not st.session_state.admin_logged_in:
        with st.container():
            st.markdown("<h1 style='text-align: center;'>&nbsp;&nbsp;&nbsp;&nbsp;火龙果茎叶病害识别信息管理系统</h1>", unsafe_allow_html=True)
            cols = st.columns([1, 2, 1])
            with cols[1]:
                with st.form("login_form"):
                    username = st.text_input("管理员账号")
                    password = st.text_input("密码", type="password")
                    if st.form_submit_button("登录", use_container_width=True):
                        if username == "admin" and password == "admin123":
                            st.session_state.admin_logged_in = True
                            st.success("登录成功")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("账号或密码错误")

    # 主界面
    if st.session_state.admin_logged_in:
        with st.sidebar:
            st.markdown("<h1 style='text-align: center;'>火龙果茎叶病害识别信息管理系统</h1>", unsafe_allow_html=True)
            page = st.radio(
                "导航菜单",
                ["用户管理", "识别结果管理", "模型管理"],
                index=0,
                format_func=lambda x: {
                    "用户管理": "用户管理",
                    "识别结果管理": "识别结果管理",
                    "模型管理": "模型管理"
                }[x]
            )

            if st.button("退出登录", use_container_width=True):
                st.session_state.admin_logged_in = False
                st.success("已退出登录")
                time.sleep(1)
                st.rerun()

        # 页面路由
        if page == "用户管理":
            user_management()
        elif page == "识别结果管理":
            result_management()
        elif page == "模型管理":
            model_management()


if __name__ == "__main__":
    main()