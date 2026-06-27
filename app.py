import streamlit as st
from tennis_analysis import process_video
import tempfile
import mediapipe as mp

st.write("MP FILE:", mp.__file__)
st.write("HAS SOLUTIONS:", hasattr(mp, "solutions"))
st.write("DIR:", dir(mp))

st.set_page_config(layout="wide")

st.title("Tennis Form Analysis")

uploaded_file = st.file_uploader(
    "動画をアップロード",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("解析開始"):

        # 一時保存
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        temp_file.write(uploaded_file.read())

        st.write("解析中...")

        frame_placeholder = st.empty()

        process_video(
            temp_file.name,
            frame_placeholder
        )

        st.success("解析完了")