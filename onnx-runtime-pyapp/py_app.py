import json
import numpy as np
import onnxruntime

# display images in notebook
import matplotlib.pyplot as plt
from PIL import Image

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def preprocess(image):
    # resize so that the shorter side is 256, maintaining aspect ratio
    def image_resize(image, min_len):
        image = Image.fromarray(image)
        ratio = float(min_len) / min(image.size[0], image.size[1])
        if image.size[0] > image.size[1]:
            new_size = (int(round(ratio * image.size[0])), min_len)
        else:
            new_size = (min_len, int(round(ratio * image.size[1])))
        image = image.resize(new_size, Image.BILINEAR)
        return np.array(image)
    image = image_resize(image, 256)

    # Crop centered window 224x224
    def crop_center(image, crop_w, crop_h):
        h, w, c = image.shape
        start_x = w//2 - crop_w//2
        start_y = h//2 - crop_h//2
        return image[start_y:start_y+crop_h, start_x:start_x+crop_w, :]
    image = crop_center(image, 224, 224)

    # transpose
    image = image.transpose(2, 0, 1)

    # convert the input data into the float32 input
    img_data = image.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

labels = load_labels('imagenet-simple-labels.json')

# Run the model on the backend
session = onnxruntime.InferenceSession('../onnx-runtime/tests/models/resnet18-v2-7.onnx', None)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name  

image = Image.open('../onnx-runtime/tests/images/siamese-cat.jpg')
image = np.array(image.convert('RGB'))
input_data = preprocess(image)

# start = time.time()
raw_result = session.run([], {input_name: input_data})
# end = time.time()
res = postprocess(raw_result)

# inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(res)

print('========================================')
print('Final top prediction is: ' + labels[idx])
print('========================================')

# print('========================================')
# print('Inference time: ' + str(inference_time) + " ms")
# print('========================================')

sort_idx = np.flip(np.squeeze(np.argsort(res)))
print('============ Top 5 labels are: ============================')
print(labels[sort_idx[:5]])
print('============ Top 5 values are: ============================')
for idx in sort_idx[:5]:
    value = res[idx] * 100
    formatted_value = "{:.6f}%".format(value)
    print(formatted_value)
print('===========================================================')

plt.axis('off')


