import numpy as np
import os
import tempfile
import cv2
from cartoonize import CartoonEffect

caart = CartoonEffect()
# Function to resize the frame to match the cartoon effect model dimensions
#def resize_frame(frame, target_width, target_height):
 #   return cv2.resize(frame, (target_width, target_height))

videoCaptureObject = cv2.VideoCapture(0)

out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'m', 'p', '4', 'v'), 30, (640, 480))
result = True
while(result):
    ret,img = videoCaptureObject.read()
    # Resize the frame to match the cartoon effect model dimensions
    #resized_img = resize_frame(img, 640, 480)
     # Save the image to a temporary file
    temp_file_path = os.path.join(tempfile.gettempdir(), "temp_img.png")
    cv2.imwrite(temp_file_path, img)
    cartoonized_img=caart.apply_cartoon_effect(temp_file_path)
    cartoonized_img1=caart.apply_cartoon_effect("test.png")
    cv2.imshow("cartoonimage1",np.array(cartoonized_img1))
    cv2.imshow("original",np.array(img))
    cv2.imshow("cartoonized", np.array(cartoonized_img))
    out.write(cartoonized_img)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
videoCaptureObject.release()
out.release()
cv2.destroyAllWindows()