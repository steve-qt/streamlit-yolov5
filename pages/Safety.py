import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import numpy as np
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
import configparser
import ast

st.set_page_config(layout="wide")

cfg_model_path = 'models/safety_openvino_model/'
model = None
confidence = .5
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

def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model prediction")

def video_input(data_src):
    vid_file = None
    if data_src == 'Live data':
        vid_file = "livewebcam"
    elif data_src == 'Sample data':
        vid_file = "data/sample_videos/helmet3_480.mp4"
    elif data_src == 'Upload data':
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())
    elif data_src == 'Rtsp data':
        vid_file = user_input
        st.write("You entered: ", user_input)
    
    # video_src = vid_file

    if vid_file:
        if vid_file == "livewebcam":
            vid_file = 0 #default webcam for windows machine, need to enable webcam for Linux Ubuntu VM [install virtual box extension pack]
        cap = cv2.VideoCapture(vid_file)
        video_src = cap
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
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
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


def getClassInfo(class_num):
    config = configparser.ConfigParser()
    config.read(".editorconfig")

    label = ast.literal_eval(config['SAFETY']['labels'])[class_num]
    color = ast.literal_eval(config['SAFETY']['colors'])[class_num]
    return label, color

def infer_image(im, size=None):
    model.conf = confidence
    model.iou = 0.65
    model.agnostic = True  # NMS class-agnostic
    model.multi_label = False
    model.size = 300
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
        class_num = int(row['class'])

        label, color = getClassInfo(class_num)
        labels.append(label)
        colors.append(color)

    image = Image.fromarray(im)
    transform = transforms.Compose([transforms.PILToTensor()])
    transformed_img = transform(image)

    boxes = torch.tensor(box_arr, dtype=torch.float)
    img_w_box = draw_bounding_boxes(transformed_img, boxes, colors=colors, labels=labels, width=5)
    img_w_box = torchvision.transforms.ToPILImage()(img_w_box)
    return img_w_box


@st.experimental_singleton
def load_model(path, device):
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


@st.experimental_singleton
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
    global model, confidence, cfg_model_path, video_src, user_input

    st.title("Safety Detector")

    st.sidebar.title("Settings")

    # upload model
    model_src = st.sidebar.radio("Select yolov5 weight file", ["Use pretrained model", "Use your own model"])
    # URL, upload file (max 200 mb)
    if model_src == "Use your own model":
        user_model_path = get_user_model()
        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")

    # check if model file is available
    # if not os.path.isfile(cfg_model_path):
        # st.warning(cfg_model_path)
        # st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")

    # device options
    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
    else:
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    # load model
    model = load_model(cfg_model_path, device_option)

    # confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.5)

    # custom classes
    if st.sidebar.checkbox("Custom Classes"):
        model_names = list(model.names.values())
        assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
        classes = [model_names.index(name) for name in assigned_class]
        model.classes = classes
    else:
        model.classes = list(model.names.keys())

    # vid src option slider
    video_type = st.sidebar.radio("Choose your video type", ["Sample data", "Upload a video", "Rtsp", "Live webcam"])

    if video_type == "Live webcam":
        video_input('Live data')
    elif video_type == "Sample data":
        video_input('Sample data')
    elif video_type == "Upload a video":
        video_input('Upload data')
    elif video_type == "Rtsp":
        user_input = st.sidebar.text_input("Enter the rtsp address ( rtsp://address )")
        # video_src = user_input
        if user_input:
            video_input('Rtsp data')

    st.sidebar.markdown("---")

    # # input options
    # input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])
    #
    # # input src option
    # data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
    #
    # if input_option == 'image':
    #     image_input(data_src)
    # else:
    #     video_input(data_src)

    # video_input('Sample data')

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
