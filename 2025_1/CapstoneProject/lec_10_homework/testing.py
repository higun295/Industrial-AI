from ultralytics import YOLO

if __name__ == '__main__':
    trained_model = YOLO('runs/detect/yolo_final_3class/weights/best.pt')
    test_results = trained_model.predict(
        source='D:/capstone_project/roboflow_dataset_2/test/images',
        save=True,
        project='runs/detect',
        name='final_test_low_conf',
        conf=0.3,
        device=0
    )


