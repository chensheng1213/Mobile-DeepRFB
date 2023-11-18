import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    deeplab = DeeplabV3()
    mode = "predict"
    '''
    mode Specifies the mode of the test:
    'predict' represents a single image prediction. If you want to modify the prediction process, such as saving images, capturing objects, etc., you can first read the detailed comments below
    'video' indicates video detection. Cameras or videos can be used for detection. For details, see the comments below.
    'fps' stands for testing fps, and the image used is street.jpg in img, see the comments below for details.
    'dir_predict' means to traverse the folder for detection and saving. By default, traverse the img folder and save the img_out folder. See the comments below for details.
    'export_onnx' means that exporting the model as onnx requires pytorch1.7.1 or more.
    '''
    count           = True
    # count Specifies whether the pixel count (i.e. area) and scale calculation of the target are performed
    # count is only valid if mode='predict'
    name_classes = ["background", "soil", "bedrock", "sand", "big rock"]
    '''
    'video_path' is used to specify the path of the video. video_path=0 indicates that the camera is detected
    'video_save_path' indicates the path where the video is saved. If video_save_path="", the video is not saved
    'video_fps' is the fps used to save the video
     video_path, video_save_path, and video_fps are valid only when mode='video'
    '''
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    '''
    'test_interval' is used to specify the number of image detections when measuring fps. Theoretically, the larger the test_interval, the more accurate the fps.
    'fps_image_path' is used to specify the fps image for testing
     test_interval and fps_image_path are valid only in mode='fps'
    '''
    test_interval = 100
    fps_image_path  = "Test_img_mer/images/1n128623896eff0205p1548l0m1.jpg"
    dir_origin_path = "predict_out/mer/img/"
    dir_save_path   = "predict_out/mer/resnet101/dir_predict_2/"
    simplify        = False
    onnx_save_path  = "test_pth/mobilev3_large_rfb.onnx"
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频）")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(deeplab.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

