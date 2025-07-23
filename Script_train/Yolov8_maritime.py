from ultralytics import YOLO

# Recommended to load pre-trained model 
model = YOLO('yolov8l.pt')

# Custom YAML file with modified architecture
model.model.cfg = 'path/lib/python3.11/site-packages/ultralytics/cfg/models/v8/maritime.yaml'

# Entrenamiento del modelo con arquitectura modificada e inicialización preentrenada
results = model.train(
    data='/data.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    device='0',
    optimizer='SGD',
    lr0=0.001,
    amp=True,
    plots=True,
    save=True,
    weight_decay=0.0005,
    warmup_epochs=5,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.3,
    erasing=0.4,
    crop_fraction=1.0,
)