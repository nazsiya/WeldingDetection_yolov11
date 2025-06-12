

from roboflow import Roboflow
rf = Roboflow(api_key="b27p4MNEi5OqwUKepU06")
project = rf.workspace("defspace").project("weld-classifier")
version = project.version(1)
dataset = version.download("yolov11")
