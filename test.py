# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

# For measuring the inference time.
import time

from render_utils import download_and_resize_image
from render_utils import display_image
from render_utils import draw_boxes

# Print Tensorflow version
print(tf.__version__)

images = dict()
images['boston'] = "https://upload.wikimedia.org/wikipedia/commons/3/30/Boston_Skyline%2C_SONY_NEX-5_Panorama_Mode_%284765830049%29.jpg"
images['beach'] = "https://upload.wikimedia.org/wikipedia/commons/0/0b/170428-A-OI229_33_%2834250568621%29.jpg"
images['festival'] = "https://upload.wikimedia.org/wikipedia/commons/4/47/Sandwich_Folk_%26_Ale_Festival_2018_FUNK4823_%2842643734594%29.jpg"
images['painting'] = "https://upload.wikimedia.org/wikipedia/commons/b/b3/Museo_Teatrale_alla_Scala_04.JPG"
images['road'] = "https://upload.wikimedia.org/wikipedia/commons/d/d5/FL100_and_US129_Signs_%2829109712941%29.jpg"
images['orleans'] = "https://upload.wikimedia.org/wikipedia/commons/d/dc/Broad_%26_Tulane_New_Orleans_26th_Dec_2019_02.jpg"
images['window'] = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Stained_glass_window%2C_St_Clements_church%2C_Hastings_%2850009211986%29.jpg/383px-Stained_glass_window%2C_St_Clements_church%2C_Hastings_%2850009211986%29.jpg"
images['train'] = "https://upload.wikimedia.org/wikipedia/commons/3/3d/Triangle_de_V%C3%A9mars_-_LGV_Interconnexion_Est.jpg"
images['house'] = "https://upload.wikimedia.org/wikipedia/commons/c/cc/Wendenstra%C3%9Fe_10_G%C3%B6ttingen_20180112_001.jpg"

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 

detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, filename, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

  display_image("/mnt/c/Users/safre/testing/{}.png".format(filename), image_with_boxes)

for filename in images.keys():
  downloaded_image_path = download_and_resize_image(images[filename], 1280, 856, False)
  run_detector(detector, filename, downloaded_image_path)
