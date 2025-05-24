from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    # 학습 실행
    results = model.train(
        data='D:/capstone_project/roboflow_dataset_2/data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/detect',
        name='yolo_final_3class'
    )


