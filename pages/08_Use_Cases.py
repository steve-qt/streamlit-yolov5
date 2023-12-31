import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import numpy as np
import base64
st.set_page_config(layout="wide")


cfg_safety_model_path = 'models/best_v8safety_openvino_model/'
cfg_weapon_model_path = 'models/weaponv155spt100ep_openvino_model/'
cfg_vehicle_person_model_path = 'models/dronev75spt100ep_openvino_model/'
cfg_switch_model_path = 'models/coxial_openvino_model/' 

model = None
confidence = .25
video_type = None
video_src = None
user_input = None

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

def video_input(data_src, data_path, key):
    vid_file = None
    # if data_path == cfg_vehicle_person_model_path:
    if data_src == 'Live data':
        vid_file = "livewebcam"
    # elif data_src == 'Sample data':
    #     vid_file = "data/sample_videos/5secallguns.mp4"
    elif data_src == 'Upload data':
        vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'], key=key)
        if vid_bytes:
            vid_file = vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())
    elif data_src == 'Rtsp data':
        vid_file = user_input
        st.write("You entered: ", user_input)
    # if data_path == cfg_safety_model_path:
    #     if data_src == 'Live data':
    #         vid_file = "livewebcam"
    #     # elif data_src == 'Sample data':
    #     #     vid_file = "data/sample_videos/5secallguns.mp4"
    #     elif data_src == 'Upload data':
    #         vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'], key=key)
    #         if vid_bytes:
    #             vid_file = vid_bytes.name.split('.')[-1]
    #             with open(vid_file, 'wb') as out:
    #                 out.write(vid_bytes.read())
    #     elif data_src == 'Rtsp data':
    #         vid_file = user_input
    #         st.write("You entered: ", user_input)
    
    # video_src = vid_file

    if vid_file:
        if vid_file == "livewebcam":
            vid_file = 0 #default webcam for windows machine, need to enable webcam for Linux Ubuntu VM [install virtual box extension pack]
        cap = cv2.VideoCapture(vid_file)
        video_src = cap
        custom_size = st.checkbox("Custom frame size", key=key)
        # confidence slider
        # confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)
        else:
            width = 1000
            height = 500

        fps = 0
        # st1, st2, st3 = st.columns(3)

        #COMMENT THIS OUT //--------------------
        # with st1:
            # st.markdown("## Height")
            # st1_text = st.markdown(f"{height}")
        # with st2:
            # st.markdown("## Width")
            # st2_text = st.markdown(f"{width}")
        # with st3:
            # st.markdown("## FPS")
            # st3_text = st.markdown(f"{fps}")
        #COMMENT THIS OUT //--------------------

        # st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            #COMMENT THIS OUT //--------------------
            # st1_text.markdown(f"**{height}**")
            # st2_text.markdown(f"**{width}**")
            # st3_text.markdown(f"**{fps:.2f}**")
            #COMMENT THIS OUT //--------------------

        cap.release()


def infer_image(im, size=None):
    model.conf = confidence
    model.source = video_src
    model.iou = 0.65
    model.agnostic = True  # NMS class-agnostic
    model.multi_label = False
    model.size = 640
    result = model(im, size=size) if size else model(im)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.cache_resource
def load_model(path, device):
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file

def main():
    # global variables
    global model, confidence, cfg_vehicle_person_model_path, cfg_safety_model_path, cfg_switch_model_path, cfg_weapon_model_path, video_src, user_input

    st.title("Use Cases")

    # st.write("This page is still being implemented / Not fully functional")

    st.markdown("---")


    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        #with st.expander("Coaxial Cable Detection"):
        #"""### Coaxial Cable Detection"""
        #st.write("[Coaxial Cable Detection](http://localhost:8501/Coaxial_Cable)")
        style = f"<a href='http://localhost:8501/Coaxial_Cable' style='font-size:24px;'> Coaxial Cable Detection</a>"
        st.write(style,unsafe_allow_html=True)
        file_ = open("data/sample_gifs/coaxialcable2.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64, {data_url}" alt="coxial cable gif">',
            unsafe_allow_html=True,
        )

    with row1_col2:
        #with st.expander("Safety Detection"):
        #"""### Safety Detection"""
        #st.write("[Safety Detection](http://localhost:8501/Safety)")
        style = f"<a href='http://localhost:8501/Safety' style='font-size:24px;'> Safety Detection</a>"
        st.write(style,unsafe_allow_html=True)
        file_ = open("data/sample_gifs/7sec_safety_off_to_on.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64, {data_url}" alt="safety gif">',
            unsafe_allow_html=True,
        )

    st.text("")
    st.text("")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        #with st.expander("Traffic Detection"):
        #"""### Traffic Detection"""
        #st.write("[Traffic Detection](http://localhost:8501/Traffic)")
        style = f"<a href='http://localhost:8501/Traffic' style='font-size:24px;'> Traffic Detection</a>"
        st.write(style,unsafe_allow_html=True)
        file_ = open("data/sample_gifs/3sectrafficcam.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64, {data_url}" alt="traffic gif">',
            unsafe_allow_html=True,
        )

    with row2_col2:
        #with st.expander("Weapon Detection"):
        #"""### Weapon Detection"""
        #st.write("[Weapon Detection](http://localhost:8501/Weapon)")
        style = f"<a href='http://localhost:8501/Weapon' style='font-size:24px;'> Weapon Detection</a>"
        st.write(style,unsafe_allow_html=True)
        file_ = open("data/sample_gifs/5secallguns.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64, {data_url}" alt="weapon gif">',
            unsafe_allow_html=True,
        )



    # st.sidebar.title("Settings")

    # device options
    # if torch.cuda.is_available():
    #     device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
    # else:
    #     device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    # with st.expander("Safety Detection"):
    #     # load model
    #     model = load_model(cfg_safety_model_path, None)

    #     # vid src option slider
    #     #with st.sidebar:
    #     video_type = st.radio("Choose your video type", ["Upload a video", "Rtsp", "Live webcam"], key="key_7") #"Sample data", 

    #     if video_type == "Live webcam":
    #         video_input('Live data', cfg_vehicle_person_model_path, key="key_8")
    #     # elif video_type == "Sample data":
    #     #     video_input('Sample data')
    #     elif video_type == "Upload a video":
    #         video_input('Upload data', cfg_vehicle_person_model_path, key="key_9")
    #     elif video_type == "Rtsp":
    #         user_input = st.text_input("Enter the rtsp address ( rtsp://address )", key="key_29")
    #         # video_src = user_input
    #         if user_input:
    #             video_input('Rtsp data', cfg_vehicle_person_model_path, key="key_10")
                
    #     # confidence slider
    #     confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=.65, key="key_24")
    
    # with st.expander("Coaxial Cable Detection"):
    #     # load model
    #     model = load_model(cfg_safety_model_path, None)

    #     # vid src option slider
    #     #with st.sidebar:
    #     video_type = st.radio("Choose your video type", ["Upload a video", "Rtsp", "Live webcam"], key="key_11") #"Sample data", 

    #     if video_type == "Live webcam":
    #         video_input('Live data', cfg_switch_model_path, key="key_12")
    #     # elif video_type == "Sample data":
    #     #     video_input('Sample data')
    #     elif video_type == "Upload a video":
    #         video_input('Upload data', cfg_switch_model_path, key="key_13")
    #     elif video_type == "Rtsp":
    #         user_input = st.text_input("Enter the rtsp address ( rtsp://address )", key="key_28")
    #         # video_src = user_input
    #         if user_input:
    #             video_input('Rtsp data', cfg_switch_model_path, key="key_14")
    
    #     # confidence slider
    #     confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=.1, key="key_25")
    
    # with st.expander("Traffic Detection"): #with st.expander("Person Detection"):
    #     # load model
    #     model = load_model(cfg_vehicle_person_model_path, None)

    #     # vid src option slider
    #     #with st.sidebar:
    #     video_type = st.radio("Choose your video type", ["Upload a video", "Rtsp", "Live webcam"], key="key_3") #"Sample data", 

    #     if video_type == "Live webcam":
    #         video_input('Live data', cfg_vehicle_person_model_path, key="key_4")
    #     # elif video_type == "Sample data":
    #     #     video_input('Sample data')
    #     elif video_type == "Upload a video":
    #         video_input('Upload data', cfg_vehicle_person_model_path, key="key_5")
    #     elif video_type == "Rtsp":
    #         user_input = st.text_input("Enter the rtsp address ( rtsp://address )", key="key_30")
    #         # video_src = user_input
    #         if user_input:
    #             video_input('Rtsp data', cfg_vehicle_person_model_path, key="key_6")
        
    #     # confidence slider
    #     confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=.5, key="key_23")

    # with st.expander("Weapon Detection"):
    #     # load model
    #     model = load_model(cfg_safety_model_path, None)

    #     # vid src option slider
    #     #with st.sidebar:
    #     video_type = st.radio("Choose your video type", ["Upload a video", "Rtsp", "Live webcam"], key="key_15") #"Sample data", 

    #     if video_type == "Live webcam":
    #         video_input('Live data', cfg_weapon_model_path, key="key_16")
    #     # elif video_type == "Sample data":
    #     #     video_input('Sample data')
    #     elif video_type == "Upload a video":
    #         video_input('Upload data', cfg_weapon_model_path, key="key_17")
    #     elif video_type == "Rtsp":
    #         user_input = st.text_input("Enter the rtsp address ( rtsp://address )", key="key_27")
    #         # video_src = user_input
    #         if user_input:
    #             video_input('Rtsp data', cfg_weapon_model_path, key="key_18")
        
    #     # confidence slider
    #     confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=.5, key="key_26")

    #st.sidebar.markdown("---")

    

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
