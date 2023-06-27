import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Object Detection Dashboard! 👋")

st.sidebar.success("Select a use case.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    ### TEAM MEMBERS
        Chase Dannen, Patrick Harris & Steven Luong
"""
)