import sys
import numpy as np
import time

# display images in notebook
import matplotlib.pyplot as plt
from PIL import Image

from official_onnx_test import run as run_onnx
from test import run as run_our

def display_image(image, network_name, prediction_micr, prediction_our, time_micr, time_our):
    text_x_left = 0   # Adjust this value to control the horizontal position (left)
    text_y_left = 1    # Adjust this value to control the vertical position (top)

    text_x_right = 1  # Adjust this value to control the horizontal position (right)
    text_y_right = 1   # Adjust this value to control the vertical position (top)

    plt.axis('off') 
    plt.imshow(image)

    plt.text(text_x_left, text_y_left, f"Microsoft runtime prediction:\n\n{prediction_micr}\nInference time: {time_micr:.3f} seconds",
     fontsize=10, color='white', backgroundcolor='black', ha='left', va='top', transform=plt.gca().transAxes)

    plt.text(text_x_right, text_y_right, f"Our prediction:\n\n{prediction_our}\nInference time: {time_our:.3f} seconds",
     fontsize=10, color='white', backgroundcolor='black', ha='right', va='top', transform=plt.gca().transAxes)
    plt.title(network_name)
    plt.show()

def main():
    arguments = sys.argv

    if len(arguments) == 1:
        image_name = 'siamese-cat'
        network_name = 'resnet'
    elif len(arguments) == 2:
        image_name = arguments[1]
        network_name = 'resnet'
    elif len(arguments) == 3:
        image_name = arguments[1]
        network_name = arguments[2]
    else:
       print('Usage: python3 main.py [image_name] [resnet/mobilenet]')
       exit()

    network_path = 'models/{}.onnx'.format(network_name)

    image_path = 'images/{}.jpeg'.format(image_name)
    image = Image.open(image_path)
    image = np.array(image.convert('RGB'))

    print('Running onnx...')

    start_micr = time.time()
    prediction_micr = run_onnx(image_path, network_path)
    end_micr = time.time()

    start_our = time.time()
    prediction_our = run_our(image_path, network_path)
    end_our = time.time()
    print("Our prediction: ", prediction_our)
    print("Microsoft prediction: ", prediction_micr)
    print("Our inference time: ", end_our - start_our)
    print("Microsoft inference time: ", end_micr - start_micr)
    print('Done')

    display_image(image, network_name, prediction_micr, prediction_our, end_micr - start_micr, end_our - start_our)

if __name__ == '__main__':
    main()



