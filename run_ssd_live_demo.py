from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import os, sys
import argparse

script_dir = os.path.dirname("__file__")
module_path = os.path.abspath(os.path.join(script_dir, "..", "pytorch-distiller-new"))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller

from distiller.data_loggers import *
import distiller.quantization as quantization

# 動作例
# $ python run_ssd_live_demo.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt --data ../pytorch_handbook/chapter7/demo/sample.jpg

parser = argparse.ArgumentParser(description="SSD live demo.")
parser.add_argument("net_type", default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd")
parser.add_argument("model_path", type=str)
parser.add_argument("label_path", type=str)
parser.add_argument("--data", type=str)
parser.add_argument("--debug-dk", type=str, help="for debug.")
distiller.quantization.add_post_train_quant_args(parser)
args = parser.parse_args()

#if len(sys.argv) < 4:
#    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
#    sys.exit(0)
#net_type = sys.argv[1]
#model_path = sys.argv[2]
#label_path = sys.argv[3]

#if len(sys.argv) >= 5:
#    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
#else:
#    cap = cv2.VideoCapture(0)   # capture from camera
#    cap.set(3, 1920)
#    cap.set(4, 1080)

if args.data is not None:
    #cap = cv2.VideoCapture(sys.argv[4])  # capture from file
    img = cv2.imread(args.data)
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(args.label_path).readlines()]
num_classes = len(class_names)

if args.net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif args.net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True, debug_dk=args.debug_dk)
elif args.net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif args.net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif args.net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
print( net )
net.load(args.model_path)

if args.qe_stats_file is not None:
    net.cpu()
    quantizer = quantization.PostTrainLinearQuantizer.from_args(net, args)
    quantizer.prepare_model()

if args.net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif args.net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, debug_dk=args.debug_dk)
elif args.net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif args.net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif args.net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

timer = Timer()
while True:
    #ret, orig_image = cap.read()
    #if orig_image is None:
    #    continue
    #image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(img,        cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(img, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imwrite('./annotated_image.jpg', img)
    cv2.imshow('annotated', img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    break
cap.release()
cv2.destroyAllWindows()
