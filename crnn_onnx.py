import onnx
import onnxruntime as ort
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--im_height', type=int, default = 32, help='Reset the height')
parser.add_argument('--im_width', type=int, default = 100, help='Reset the width')

parser.add_argument('-i', '--in_pic', type=str, default='demo.png', help='prepared picture to open')
args = parser.parse_args()

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

class strLabelConverter(object):

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

    def decode(self, t, length, raw=False):
        assert len(t) == length, "text with length: {} does not match declared length: {}".format(len(t), length)
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)


im_height = args.im_height
im_width = args.im_width

img_raw = cv2.imread(args.in_pic, cv2.IMREAD_GRAYSCALE)
img_raw = cv2.resize(img_raw, (im_width, im_height))

img = np.float32(img_raw)
img /= 255.0
img -= 0.5
img /= 0.5

image = img.reshape(1,im_height,im_width)

#-----------------onnx_detect------------------
onnx_file = 'crnn_pytorch.onnx'
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print('The model is checked!')

data = np.expand_dims(image, axis=0)

sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CUDAExecutionProvider'])

ort_inputs = {sess.get_inputs()[0].name: data}
ort_outs = sess.run(None, ort_inputs)

data_out = ort_outs[0]
data_out = data_out.reshape(26,37)
preds  = np.argmax(data_out, axis=1)
preds = preds.tolist()

preds_size = len(preds)

converter = strLabelConverter(alphabet)
raw_pred = converter.decode(preds, preds_size, raw=True)
sim_pred = converter.decode(preds, preds_size, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))

print("Exported model has been predicted by ONNXRuntime!")
