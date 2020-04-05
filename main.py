import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, send_file

from io import BytesIO
import tempfile
from six.moves import urllib

import matplotlib
matplotlib.use('agg')
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.core.framework import *

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, graph_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = graph_pb2.GraphDef()
    with open(graph_path, "rb") as pbfile:
        graph_def.ParseFromString(pbfile.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#download_path = '/home/adi/Workspace/models/research/deeplab/datasets/goat_molt_seg/exp/train_on_trainval_set/export/frozen_inference_graph.pb'
download_path = '/home/adi/Workspace/models/research/deeplab/datasets/goat_molt_seg/exp/train_on_trainval_set_mobilenetv2/export/frozen_inference_graph.pb'

MODEL = DeepLabModel(download_path)

SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ''  #@param {type:"string"}

_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
               'deeplab/g3doc/img/%s.jpg?raw=true')

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(original_im)

  vis_segmentation(resized_im, seg_map)


#image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
#run_visualization(image_url)

UPLOAD_FOLDER = '/static/Uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/uploadImage")
def uploadImage():
    return render_template("uploadImage.html")

#@app.route('/upload', methods=['GET', 'POST'])
#def upload():
#    if request.method == 'POST':
#        file = request.files['file']
#        extension = os.path.splitext(file.filename)[1]
#        f_name = str(uuid.uuid4()) + extension
#        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
#    return json.dumps({'filename':f_name})
#def upload():
#    if request.method == 'POST':
#        if 'file' not in request.files:
#            return redirect(request.url)
#        file = request.files['file']
#        if file.filename == '':
#            return redirect(request.url)
#        if file and allowed_file(file.filename):
#            filename = secure_filename(file.filename)
#            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#@app.route('/uploadImage', methods=['GET', 'POST'])
#def upload():
#
#    if request.method == "POST":
#
#        if request.files:
#
#            image = request.files["image"]
#
#        print (image)
#
#        return redirect(request.url)
#

#image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
#run_visualization(image_url)

#@app.route('/test', methods=['GET', 'POST'])
@app.route("/uploadAjax", methods=['GET', 'POST'])
def uploadAjax():
     imgFile=request.files['file']
     print("isthisFile")
     print(imgFile)
     print(imgFile.filename)
     imgFile.save("./static/Uploads/"+imgFile.filename)
#     return render_template("uploadImage.html")
#def upload_file():
     if request.method == 'POST':
        if imgFile:
            #jpeg_str=imgFile.read()
            jFile="./static/Uploads/"+imgFile.filename
            #jpeg_str=imgFile.read()
            #print(len(jpeg_str))
            #print(type(jpeg_str))
            original_im = Image.open(jFile)
            #original_im = Image.open(BytesIO(jpeg_str))
            resized_im, seg_map = MODEL.run(original_im)
            seg_image = label_to_color_image(seg_map).astype(np.uint8)
            print(resized_im.__class__.__name__)
            print(seg_map.__class__.__name__)
            print(seg_image.__class__.__name__)
            #vis_segmentation(resized_im, seg_map)
            seg_img = Image.fromarray(seg_image, 'RGB')
            seg_img.save("./static/Uploads/proc"+imgFile.filename)
            print("XXXXA:")
#            return redirect(url_for('processedImage', msg="./static/Uploads/proc"+imgFile.filename))
#            data = {
#                "imageName": "./static/Uploads/"+imgFile.filename,
#                "imageProcessed": "./static/Uploads/proc"+imgFile.filename
#                }
            return imgFile.filename
#    return render_template('upload_file.html')

@app.route("/processedImage", methods=['GET'])
def processedImage():
    imageName="./static/Uploads/"+request.args.get('img')
    print("XXXXXB:" + imageName)
    imageProcessed="./static/Uploads/proc"+request.args.get('img')
    ret = render_template('processedImage.html', img_filename=imageName, processedimg_filename=imageProcessed)
    print("XXXXXC:" + imageName)
    return ret


if __name__ == '__main__':
    app.run(host='0.0.0.0')
