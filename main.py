from ultralytics import YOLO

if __name__ == '__main__':
    # 训练
    model = YOLO('test_model.yaml', task='detect')

    results = model.train(data='test.yaml', epochs=10, imgsz=(640, 640))

    # # 预测
    # model = YOLO("best.pt")
    # source = "1.png"
    # results = model.predict(task='detect', source=source, show=True, imgsz=(640, 640), save=True)
