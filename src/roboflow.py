import roboflow 

roboflow.login()

rf = roboflow.Roboflow()

project = rf.workspace(WORKSPACE_ID).project("droneandvehicledetection")
dataset = project.version(2).download("yolov8")

project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")