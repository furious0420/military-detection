  print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ERROR: No GPU detected!")
        print("Please check your CUDA installation")
        device = 'cpu'
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train with forced GPU
    results = model.train(
        data='unified_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='unified_detection',
        name='multi_class_v1',
        workers=4,  # Reduced workers
        amp=True,
        verbose=True
    )
    