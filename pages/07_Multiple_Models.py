import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import numpy as np
import configparser
import ast
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms

st.set_page_config(layout="wide")

confidence = .25
video_type = None
video_src = None
user_input = None
models = []
indexes = []
model_cfg_names = ['COAXIAL', 'SAFETY', 'TRAFFIC', 'WEAPON']
model_names = ['coaxial', 'safety', 'traffic', 'weapon']
model_paths = ['models/coaxial_openvino_model/',
               'models/best_v8safety_openvino_model/',
               'models/dronev75spt100ep_openvino_model/',
               'models/weaponv155spt100ep_openvino_model/']

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


def video_input(data_src, key):
    vid_file = None
    # if data_path == cfg_vehicle_person_model_path:
    if data_src == 'Live data':
        vid_file = "livewebcam"
    elif data_src == 'Sample data':
        vid_file = "data/sample_videos/multimodel480.mp4"
    elif data_src == 'Upload data':
        vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'], key=key)
        if vid_bytes:
            vid_file = vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())
    elif data_src == 'Rtsp data':
        vid_file = user_input
        st.write("You entered: ", user_input)

    if vid_file:
        if vid_file == "livewebcam":
            vid_file = 0  # default webcam for windows machine, need to enable webcam for Linux Ubuntu VM [install virtual box extension pack]
        cap = cv2.VideoCapture(vid_file)
        # video_src = cap

        custom_size = st.checkbox("Custom frame size", key=key)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)
        else:
            width = 1000
            height = 500

        fps = 0
        st1, st2, st3 = st.columns(3)

        # COMMENT THIS OUT //--------------------
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")
        # COMMENT THIS OUT //--------------------

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

            # COMMENT THIS OUT //--------------------
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            # COMMENT THIS OUT //--------------------

        cap.release()


def getClassInfo(class_cfg_name, class_num):
    config = configparser.ConfigParser()
    config.read(".editorconfig")
    label = ast.literal_eval(config[class_cfg_name]['labels'])[class_num]
    color = ast.literal_eval(config[class_cfg_name]['colors'])[class_num]
    return label, color


def infer_image(im, size=None):
    image = Image.fromarray(im)
    transform = transforms.Compose([transforms.PILToTensor()])
    transformed_img = transform(image)

    for model_idx in indexes:
        model = models[model_idx]
        model.conf = confidence
        model.source = video_src
        model.iou = 0.65
        model.agnostic = True  # NMS class-agnostic
        model.multi_label = False
        model.size = 640
        result = model(im, size=size) if size else model(im)
        box_arr = []
        colors = []
        labels = []
        for index, row in result.pandas().xyxy[0].iterrows():
            # print(row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence'])
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            box_arr.append([x1, y1, x2, y2])
            class_cfg_name = model_cfg_names[model_idx]
            class_num = int(row['class'])
            label, color = getClassInfo(class_cfg_name, class_num)
            labels.append(label)
            colors.append(color)

        if not box_arr:
            continue

        boxes = torch.tensor(box_arr, dtype=torch.float)
        transformed_img = draw_bounding_boxes(transformed_img,
                                              boxes,
                                              colors=colors,
                                              labels=labels,
                                              font='arial',
                                              font_size=20,
                                              width=5)

    return torchvision.transforms.ToPILImage()(transformed_img)


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
    global models, model_paths, model_names, indexes, confidence, video_src, user_input, video_type

    st.title("Multiple Models")

    st.write("This page is still being implemented / Not fully functional")

    # load models
    for path in model_paths:
        model = load_model(path, None)
        models.append(model)

    # device options
    # if torch.cuda.is_available():
    #     device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
    # else:
    #     device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    # vid src option slider
    # with st.sidebar:

    options = st.multiselect(
        'What are your choice of models',
        model_names)

    for option in options:
        # st.write('Index:', model_names.index(option))
        indexes.append(model_names.index(option))

    video_type = st.radio("Choose your video type", ["Sample data", "Upload a video", "Rtsp", "Live webcam"],
                          key="key_11")  # "Sample data",

    # confidence slider
    confidence = st.slider('Confidence', min_value=0.1, max_value=1.0, value=.5, key="key_25")

    if st.button('Load the models'):
        if video_type == "Live webcam":
            video_input('Live data', key="key_12")
        elif video_type == "Sample data":
            video_input('Sample data', key="key_22")
        elif video_type == "Upload a video":
            video_input('Upload data', key="key_13")
        elif video_type == "Rtsp":
            user_input = st.text_input("Enter the rtsp address ( rtsp://address )", key="key_28")
            # video_src = user_input
            if user_input:
                video_input('Rtsp data', key="key_14")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
