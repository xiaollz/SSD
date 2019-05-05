from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
_C.MODEL.THRESHOLD = 0.5
_C.MODEL.NUM_CLASSES = 81
# Hard negative mining
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2

_C.MODEL.FCOS_ON = True
# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.FCOS.STRIDES = [8, 16, 32, 64, 128, 256]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Image size
_C.INPUT.IMAGE_SIZE = 512
# Values to be used for image normalization, RGB layout
_C.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# train configs
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# _C.TEST.NMS_THRESHOLD = 0.45
# _C.TEST.CONFIDENCE_THRESHOLD = 0.01
# change MAX_PER_CLASS to 400 as official caffe code will slightly increase mAP(0.8025=>0.8063, 0.7783=>0.7798)
_C.TEST.MAX_PER_CLASS = 200
_C.TEST.MAX_PER_IMAGE = -1

_C.OUTPUT_DIR = 'output'
