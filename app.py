import streamlit as st
import mediapipe as mp

st.title("TEST")

st.write("MP FILE:", mp.__file__)
st.write("HAS SOLUTIONS:", hasattr(mp, "solutions"))
st.write("DIR:", dir(mp))