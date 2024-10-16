from ts.torch_handler.base_handler import BaseHandler
import io
import numpy as np
import cv2
from os import path

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class ModelHandler(BaseHandler):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self._context = None
        self._batch_size = 0
        self.initialized = False
        self.predictor = None
        self.model_file = "model_final.pth"
        self.config_file = "config.yaml"
        self.class_name_file = "class_names.txt"
        self.SCORE_THRESH_TEST = 0.5
        self.NUM_CLASSES = 1

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        """

        if not path.exists(self.model_file):
            raise Exception("Model file not found")

        if not path.exists(self.config_file):
            raise Exception("Config file not found")

        if not path.exists(self.class_name_file):
            raise Exception("Class names file not found")

        if path.exists(self.class_name_file):
            with open(self.class_name_file, 'r') as f:
                self.class_names = f.readlines()
            self.class_names = [name.strip() for name in self.class_names]
        else:
            raise Exception("Class names file not found")

        try:
            cfg = get_cfg()
            cfg.set_new_allowed(True)
            cfg.merge_from_file(self.config_file)
            cfg.MODEL.WEIGHTS = self.model_file
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.NUM_CLASSES
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.SCORE_THRESH_TEST
            self.predictor = DefaultPredictor(cfg)

        except Exception as e:
            print("Failed to load model: {}".format(str(e)))

        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.

        :param batch: list of raw requests, should match batch size

        :return: list of preprocessed model input data
        """

        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))

        images = []

        for request in batch:
            request_body = request.get("body")

            input = io.BytesIO(request_body)

            img = cv2.imdecode(np.fromstring(input.read(), np.uint8), 1)

            images.append(img)

        return images

    def inference(self, model_input):
        """
        Internal inference methods

        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """

        outputs = []

        for image in model_input:
            output = self.predictor(image)  # run our predictions

            outputs.append(output)

        return outputs

    def postprocess(self, inference_output):
        """
        Return predict result in batch.

        :param inference_output: list of inference output
        :return: list of predict results
        """

        responses = []

        for output in inference_output:
            predictions = output["instances"].to("cpu")

            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
            masks = (predictions.pred_masks > 0.5).squeeze().numpy() if predictions.has("pred_masks") else None
            classes = [self.class_names[class_id] for class_id in classes]

            # Convert binary masks to contours for saving space
            contours_list = []

            for mask in masks:
                mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_list.append([contour.tolist() for contour in contours])

            responses_json = {'classes': classes, 'scores': scores, "boxes": boxes, "mask_contours": contours_list}
            responses.append(responses_json)

        return responses

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions

        :param data: input data
        :param context: mms context
        """

        model_input = self.preprocess(data)

        model_out = self.inference(model_input)

        output = self.postprocess(model_out)

        whole_result = []

        for i in range(len(data)):
            single_output = {'classes': output[i]['classes'],
                             'scores': output[i]['scores'].tolist(),
                             'boxes': output[i]['boxes'].tensor.tolist(),
                             'mask_contours': output[i]['mask_contours'],
                             'shape': model_input[i].shape,
                             }

            whole_result.append(single_output)

        return whole_result