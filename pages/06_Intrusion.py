import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import numpy as np
from streamlit_drawable_canvas import st_canvas
#import streamlit_canvas as st_canvas
st.set_page_config(layout="wide")

cfg_model_path = 'models/dronev75spt100ep_openvino_model/'
model = None
# confidence = .5
video_type = None
video_src = None
user_input = None
# auto_replay = False
# increment = 5
# user_defined_roi = False
# user_defined_ROI = np.zeros((2,2),int) #holds coordinates of mouse click ROI
# within_ROI = None
# count_roi=0
# cv2.namedWindow('Video Stream')


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

# draw_roi = st.sidebar.selectbox(
#     "Draw ROI rect:", ("rect") #point, freedraw, line, circle, transform
# )

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


def video_input(data_src): #, user_roi): #, auto_replay):
    vid_file = None

    if data_src == 'Live data':
        vid_file = "livewebcam"
    elif data_src == 'Sample data':
        vid_file = "data/sample_videos/10secweaponrace.mp4" 
    elif data_src == 'Upload data':
        vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())
    elif data_src == 'Rtsp data':
        vid_file = user_input
        st.write("You entered: ", user_input)
    
    # video_src = vid_file
    # increment = 0

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
        else:
            width = 1000
            height = 500

        fps = 0
        st1, st2, st3 = st.columns(3)
        
        # COMMENT THIS OUT //--------------------
        # with st1: 
        #     st.markdown("## Height")
        #     st1_text = st.markdown(f"{height}")
        # with st2:
        #     st.markdown("## Width")
        #     st2_text = st.markdown(f"{width}")
        # with st3:
        #     st.markdown("## FPS")
        #     st3_text = st.markdown(f"{fps}")
        # COMMENT THIS OUT //---------------------

        st.markdown("---")
        output = st.empty()

        # frames = []

        prev_time = 0
        curr_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                # if auto_replay:
                #     st.write("Can't read frame, video ended? Restarting playback ....")
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                #     break
                # else:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #output_img = cv2.resize(frame,(width,height))
            output_img = infer_image(frame) #frame without the roi
            output.image(output_img) #, use_column_width=True)

            #if user_roi:
            # if count_roi == 2:
            #     a1 = user_defined_ROI[0][0]
            #     b1 = user_defined_ROI[0][1]

            #     a2 = user_defined_ROI[1][0]
            #     b2 = user_defined_ROI[1][1]

            #     cv2.rectangle(frame, (a1, b1), (a2, b2), (0, 250, 250), 3) #create the ROI area on the 2nd click detection (yellow)
            #     within_ROI = frame[b1:b2, a1:a2]

            #     frame = within_ROI
                #output_img = infer_image(within_ROI) #frame without the roi
                #frame = output_img
                #output.image(output_img)   
            #else:
                #output_img = infer_image(frame) #frame without the roi
                #output.image(output_img) #, use_column_width=True)
                #frame=output_img
                
            # cv2.imshow('Video Stream', frame)    
            # cv2.setMouseCallback('Video Stream', userClickPoints)

            # if cv2.waitKey(1) == ord("q"):
            #     output_img.release()
            #     quit()

            #COMMENT THIS OUT //--------------------
            # st1_text.markdown(f"**{height}**")
            # st2_text.markdown(f"**{width}**")
            # st3_text.markdown(f"**{fps:.2f}**")
            #COMMENT THIS OUT //--------------------

            # draw = False
            # point1 = None
            # point2 = None
            # stframe = st.empty()
            # stframe.image(np.zeros((1, 1, 3), dtype=np.uint8))

            # if user_roi == True:
            #     def on_click_event(x, y):

            #         if not draw:
            #             draw = True
            #             point1 = (x, y)

            #         else:
            #             draw = False
            #             point2 = (x, y)

            #         if event == "mousedown":
            #             draw = True
            #             point1 = data_roi["x0"], data_roi["y0"]
            #         elif event == "mouseup":
            #             draw = False
            #             point2 = data_roi["x1"], data_roi["y1"]

            #             cv2.rectangle(frame, point1, point2, (0, 250, 250), 2)
            #             output_img = infer_with_roi(user_roi, frame) #frame with the roi
            #             stframe.image(frame, channels="RGB") #, use_column_width=True)
            # else:
            #     output_img = infer_image(frame) #frame without the roi
            #     output.image(output_img) #, use_column_width=True)


            # if user_roi:
            #     event, x, y, flags = st.captur
                # canvas_result = st_canvas(
                #     fill_color="rgba(250, 250, 0, 0.3)",
                #     stroke_width=2,
                #     stroke_color="rgb(0, 250, 0)",
                #     background_color="#000000",
                #     update_streamlit=True,
                #     height=500, #frame.shape[0],
                #     width=1000,
                #     drawing_mode="RECTANGLE",
                #     key="canvas",
                #     on_event=on_click_event,
                # )

                # if canvas_result is not None:
                #     if canvas_result["type"] == "rectangle":
                #         cx1 = int(canvas_result["geometry"]["x0"])
                #         cy1 = int(canvas_result["geometry"]["x0"])
                #         cx2 = int(canvas_result["geometry"]["x0"])
                #         cy2 = int(canvas_result["geometry"]["x0"])
                #         user_point_1 = (cx1, cy1)
                #         user_point_2 = (cx2, cy2)

                #         cv2.rectangle(frame, user_point_1, user_point_2, (0, 250, 250), 2)

            # stframe.register_on_mousewheel_callback(on_click_event)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

        #cv2.setMouseCallback('Live Camera Feed', userClickPoints)

        cap.release()

        # if auto_replay:
        #     while True:
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #         for frame in frames:
        #             output.image(frame)
        #             time.sleep(1/fps)
    
# def userClickPoints(event, x, y, flags, param):
    # global count_roi, user_defined_ROI, user_defined_roi

    # if event == cv2.EVENT_LBUTTONDOWN: #detects when the 1st ROI point / user click has been made
    #     user_defined_ROI[count_roi] = x,y
    #     count_ROI += 1
    #     if(count_ROI == 2):
    #         user_defined_roi = True

    # if event == cv2.EVENT_RBUTTONDOWN: #detects when the board user wants to reset ROI
    #     count_ROI = 0
    #     user_defined_ROI[count_roi] = [2,2] 
    #     user_defined_ROI[count_roi+1] = [2,2]  
    #     user_defined_roi = False      

def userClickPoints(event, x, y, flags, params):
    global count_roi

    if event == cv2.EVENT_LBUTTONDOWN: #detects when the 1st ROI point / user click has been made
        user_defined_ROI[count_roi] = x,y
        count_roi += 1

    if event == cv2.EVENT_RBUTTONDOWN: #detects when the user wants to reset ROI
        count_roi = 0
        user_defined_ROI[count_roi] = [2,2] 
        user_defined_ROI[count_roi+1] = [2,2]

def infer_image(im, size=None):
    model.conf = confidence
    model.source = video_src
    model.classes = 16
    model.iou = 0.65
    model.agnostic = True  # NMS class-agnostic
    model.multi_label = False
    model.size = 640
    result = model(im, size=size) if size else model(im)
    #result.render()
    # image = Image.fromarray(result.ims[0])
    return result

# def infer_with_roi(user_roi, im, size=None):

#     model.conf = confidence
#     model.source = video_src
#     # model.classes = 16
#     model.iou = 0.65
#     model.agnostic = True  # NMS class-agnostic
#     model.multi_label = False
#     model.size = 640
#     result = model(im, size=size) if size else model(im)
#     result.render()
#     image = Image.fromarray(result.ims[0])
#     return image

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

def regionOfInterest(video_file):
    cap = cv2.VideoCapture(vid_file)
    video_src = cap
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if custom_size:
        width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)
    else:
        width = 1000
        height = 500

    st.markdown("---")

    prev_time = 0
    curr_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # if auto_replay:
            #     st.write("Can't read frame, video ended? Restarting playback ....")
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            #     break
            # else:
            st.write("Can't read frame, stream ended? Exiting ....")
            break


def main():
    # global variables
    global model, confidence, cfg_model_path, video_type, video_src, user_input #, user_defined_roi

    st.title("Intrusion Detector")

    st.write("This page will open the video in a frame in order to track the user defined region of interest [intruders: currently person only]")

    st.write("This page is still being implemented")

    # st.sidebar.title("Settings")

    # # upload model
    # model_src = st.sidebar.radio("Select yolov5 weight file", ["Use pretrained model", "Use your own model"])
    # # URL, upload file (max 200 mb)
    # if model_src == "Use your own model":
    #     user_model_path = get_user_model()
    #     if user_model_path:
    #         cfg_model_path = user_model_path

    #     st.sidebar.text(cfg_model_path.split("/")[-1])
    #     st.sidebar.markdown("---")

    # # check if model file is available
    # # if not os.path.isfile(cfg_model_path):
    #     # st.warning(cfg_model_path)
    #     # st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")

    # # device options
    # if torch.cuda.is_available():
    #     device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
    # else:
    #     device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    # # load model
    # model = load_model(cfg_model_path, device_option)

    # # confidence slider
    # confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.5)

    # # custom classes
    # if st.sidebar.checkbox("Custom Classes"):
    #     model_names = list(model.names.values())
    #     assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
    #     classes = [model_names.index(name) for name in assigned_class]
    #     model.classes = classes
    # else:
    #     model.classes = list(model.names.keys())

    # # vid src option slider
    # video_type = st.radio("Choose your video type", ["Sample data", "Upload a video", "Rtsp", "Live webcam"])

    # # roi_st1, roi_st2 = st.columns(2)

    # # with roi_st1:
    # #     if st.button("Define ROI"):
    # #         user_defined_roi = True
    # # with roi_st2:
    # #     if st.button("Clear ROI"):
    # #         user_defined_roi = False

    # # user_define_roi = st.selectbox(
    # # "Draw ROI rect:", ("rect", "circle") #point, freedraw, line, circle, transform
    # # )

    # vid_file = None
    
    # #customs
    # model.conf = confidence
    # model.source = video_src
    # model.classes = 16
    # model.iou = 0.65
    # model.agnostic = True  # NMS class-agnostic
    # model.multi_label = False
    # model.size = 640


    # if video_type == "Live webcam":
    #     #video_input('Live data') #, user_defined_roi) #, False) 
    #     vid_file = "livewebcam"
    #     if vid_file == "livewebcam":
    #         vid_file = 0 #default webcam for windows machine, need to enable webcam for Linux Ubuntu VM [install virtual box extension pack]
    #     # regionOfInterest(vid_file)
    # elif video_type == "Sample data":
    #     #video_input('Sample data') #, user_defined_roi) #, True) 
    #     vid_file = "data/sample_videos/10secweaponrace.mp4" 
    #     # regionOfInterest(vid_file)
    # elif video_type == "Upload a video":
    #     #video_input('Upload data') #, user_defined_roi) #, True) 
    #     vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
    #     if vid_bytes:
    #         vid_file = vid_bytes.name.split('.')[-1]
    #         with open(vid_file, 'wb') as out:
    #             out.write(vid_bytes.read())
    #         # regionOfInterest(vid_file)
    # elif video_type == "Rtsp":
    #     user_input = st.text_input("Enter the rtsp address ( rtsp://address )")
    #     # video_src = user_input
    #     if user_input:
    #         vid_file = user_input
    #         st.write("You entered: ", vid_file)
    #         #video_input('Rtsp data') #, user_defined_roi) #, False) 
    #         # regionOfInterest(vid_file)


    # st.sidebar.markdown("---")

    # # # input options
    # # input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])
    # #
    # # # input src option
    # # data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
    # #
    # # if input_option == 'image':
    # #     image_input(data_src)
    # # else:
    # #     video_input(data_src)

    # # video_input('Sample data')
    

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
