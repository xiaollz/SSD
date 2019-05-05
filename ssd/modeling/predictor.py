import torch

from .data_preprocessing import PredictionTransform


class Predictor:
    def __init__(self, cfg, model, iou_threshold, score_threshold, device):
        self.cfg = cfg
        self.model = model
        self.transform = PredictionTransform(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.PIXEL_MEAN)
        self.device = device
        self.model.eval()

    def predict(self, image):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            boxes = self.model(images)
        results = boxes
        boxes, labels, scores = results[0]
        return boxes, labels, scores
