from ultralytics import YOLO
import os

def train_plate_detector():
    """Train YOLOv8 model for license plate detection"""
    
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='plate_detector',
        patience=20,
        save=True,
        device=0,  # Use GPU 0, set to 'cpu' for CPU training
        workers=8,
        project='runs/detect',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split='val',
        save_json=False,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        source=None,
        show=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        vid_stride=1,
        line_width=None,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        boxes=True,
        format='torchscript',
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=False,
        opset=None,
        workspace=4,
        nms=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    # Validate the model
    metrics = model.val()
    
    print(f"Training completed!")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    # Export the best model
    best_model_path = 'runs/detect/plate_detector/weights/best.pt'
    if os.path.exists(best_model_path):
        # Copy to models directory
        os.makedirs('../models', exist_ok=True)
        import shutil
        shutil.copy(best_model_path, '../models/best.pt')
        print(f"Best model saved to ../models/best.pt")

if __name__ == "__main__":
    train_plate_detector()
