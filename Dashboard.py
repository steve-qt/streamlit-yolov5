import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Object Detection Dashboard! 👋")
# st.image("hcl-logo.png")
# st.set_logo("hcl-logo.png")
# st.sidebar.success("Select a use case.")

st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://th.bing.com/th/id/R.1117c9dcb73e4226297f7967b5adadcc?rik=W1PFQJjMCQMG6Q&riu=http%3a%2f%2f4.bp.blogspot.com%2f_Q8UtAKpUjn8%2fS6Y4fgcd26I%2fAAAAAAAACLc%2fSMDUxiAziUc%2fs320%2fhcl_logo.png&ehk=zxggoALZcXYRYKpUhmYxX0kty9iJnuGvb8cwZuDytk8%3d&risl=&pid=ImgRaw&r=0);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
                width: 300px;
                height: auto;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    ### TEAM MEMBERS
        Chase Dannen, Patrick Harris & Steven Luong
"""
)