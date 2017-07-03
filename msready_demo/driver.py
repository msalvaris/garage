import numpy as np
import logging, sys, json
import urllib.request
import base64
from cntk.ops.functions import load_model
from cntk.ops import combine
from PIL import Image, ImageOps
from io import BytesIO
import argparse
from timeit import default_timer

logger = logging.getLogger("cntk_svc_logger")
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

trainedModel = None


class Timer(object):
    def __init__(self, timer=default_timer, factor=1):
        self.timer = timer
        self.factor = factor
        self.end = None

    def __call__(self):
        """ Return the current time """
        return self.timer()

    def __enter__(self):
        """ Set the start time """
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Set the end time """
        self.end = self()

    def __str__(self):
        return '%.3f' % (self.elapsed)

    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor


def aml_cli_get_sample_request():
    return 'Sample request here'


def init():
    global trainedModel, label_lookup

    # Load the model from disk and perform evals
    # Load labels txt
    with open('synset.txt', 'r') as f:
        label_lookup = [l.rstrip() for l in f]
    
    # The pre-trained model was trained using brainscript
    # Loading is not we need the right index 
    # See https://github.com/Microsoft/CNTK/wiki/How-do-I-Evaluate-models-in-Python
    # Load model and load the model from brainscript (3rd index)

    with Timer(factor=1000) as t:
        # Load the model from disk and perform evals
        trainedModel = load_model('ResNet_152.model')
        trainedModel = combine([trainedModel.outputs[3].owner])
    loadTimeMsg = "Model loading time: {0} ms".format(round(t.elapsed, 2))
    logger.info(loadTimeMsg)


def _decode_string(base64ImgString):
    if base64ImgString.startswith('b\''):
        base64ImgString = base64ImgString[2:-1]
    return base64ImgString.encode('utf-8')


def _img_str_to_bytes(base64Img):
    decoded_img = base64.b64decode(base64Img)
    return BytesIO(decoded_img)


def _crop_img(img_buffer):
    pil_img = Image.open(img_buffer).convert('RGB')
    return ImageOps.fit(pil_img, (224, 224), Image.ANTIALIAS)


def _img_to_array(pil_img):
    rgb_image = np.array(pil_img, dtype=np.float32)
    # Resnet trained with BGR
    bgr_image = rgb_image[..., [2, 1, 0]]
    return np.ascontiguousarray(np.rollaxis(bgr_image, 2))


def _predict(image_data):
    return np.squeeze(trainedModel.eval({trainedModel.arguments[0]: [image_data]}))


def _select_top_labels(predictions, num_results=3):
    top_indices = (-np.array(predictions)).argsort()[:num_results]
    return [(label_lookup[i], predictions[i] * 100) for i in top_indices]


def run(inputString):
    images=json.loads(inputString)
    result = []

    with Timer(factor=1000) as t:
        for base64ImgString in images:
            base64Img = _decode_string(base64ImgString)
            img_buffer = _img_str_to_bytes(base64Img)
            pil_img = _crop_img(img_buffer)
            image_data = _img_to_array(pil_img)
            predictions = _predict(image_data)
            result.append(_select_top_labels(predictions, num_results=3))

    return (result, 'Computed in {0} ms'.format(t.elapsed))


def _url_img_to_byte_string(url):
    bytfile = BytesIO(urllib.request.urlopen(url).read())
    img = Image.open(bytfile).convert('RGB')  # 3 Channels
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)  # Fixed size 
    imgio = BytesIO()
    img.save(imgio, 'PNG')
    imgio.seek(0)
    dataimg = base64.b64encode(imgio.read())
    return dataimg.decode('utf-8')


def test_driver(img_url):
    img_string = _url_img_to_byte_string(img_url)
    init()
    return run('["{0}"]'.format(img_string))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_url", type=str, help="URL of image to classify", required=True)
    args = parser.parse_args()
    print(test_driver(args.img_url))
    
