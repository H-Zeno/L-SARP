from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
from PIL import Image

def train(model, config, data, vis_block=False):

    #training script for the yolo

    save_dir = config["save_dir"]
    epochs = config["epochs"]
    imgsz = config["imgsz"]
    batch_size = config["batch_size"]
    augment = config["augment"]
    shear = config["shear"]
    perspective = config["perspective"]

    results_train = model.train(data=data,
                                epochs=epochs,
                                imgsz=imgsz,
                                batch=batch_size,
                                augment=augment,
                                perspective=perspective,
                                shear=shear,
                                save_dir=save_dir)

    if vis_block:
        plot_results(results_train)

    return results_train

def infer(model, image, vis_block=False):

    results_predict = model.predict(source=image, imgsz=1280, conf=0.3, iou=0.7, max_det=8, agnostic_nms=True, save=False)  # save plotted images
    # results_predict = get_prediction(image, model)
    # results_predict = get_sliced_prediction(image, model)

    if vis_block:
        # results_predict.export_visuals("/home/cvg-robotics/tim_ws/")
        for result in results_predict:
            #
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen

    return results_predict



if __name__ == '__main__':

    mode = "predict"
    name = "yolov8l.pt"
    data = "/home/cvg-robotics/tim_ws/data_switch_detection/20240829/data.yaml"

    if mode == "train":
        model = YOLO(name)
        config = {"epochs": 60,
                  "imgsz": 1280,
                  "batch_size": 8,
                  "augment": False,
                  "perspective": 0.00,
                  "shear": 0,
                  "save_dir": "../../weights/"}
        train(model, config, data)

    if mode == "predict":
        image = Image.open("/home/cvg-robotics/tim_ws/IMG_1011.jpeg")
        model = YOLO("../weights/train30/weights/best.pt")

        results = infer(model, image, vis_block=True)
        a = 2

