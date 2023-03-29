python tools/infer.py ^
    -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml ^
    -o Arch.name=ResNet101_vd ^
    -o Global.pretrained_model="output/ResNet101_vd/best_model" ^
    -o Infer.infer_imgs="./test_data/"