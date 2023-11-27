# Lab2 for Video Object Detection

In order to install the modified code from the ["Memory Enhanced Global-Local Aggregation for Video Object Detection"](https://arxiv.org/abs/2003.12063) repository, known as MEGA, you will need to follow the following steps:

1. Download the [BASE](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view?usp=sharing) and [MEGA](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view?usp=sharing) models. This will download two .pth files that are needed for running the demo.
2. Place the R_101.pth and MEGA_R_101.pth files under the mega.pytorch folder.
3. Run the `install.sh` script
4. Test it with the following commands:
    * Test it with BASE model from images directory:
    
        python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG"\
            --visualize-path <image_folder> \
            --output-folder <output_folder>

    * Test it with MEGA model from images directory:

        python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth --suffix ".JPEG"\
            --visualize-path <image_folder> \
            --output-folder <output_folder>

    * Test it with BASE model from video directory:
    
        python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth  --video\
            --visualize-path <path_to_video> \
            --output-folder <path_to_output_folder> --output-video

    * Test it with BASE model from video directory:
    
        python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth  --video\
            --visualize-path <path_to_video> \
            --output-folder <path_to_output_folder> --output-video

We provide a series of test videos and evidences of how the model should work. This folder are under the mega.pytorch directory. In the case of the test videos, we have image_folder, with the frames that compose a video and vid_folder with 4 different test videos. In the case of the evidences folder, we have the vis_im* and vis_vid* folders with several evidences from the test folders for you to compare your own results with the one obtained by us.

One additional note on the commands is that, if the input is a video, the output can be both a video if the `--output-video` flag is activated or a sequences of frames as images is the option is not added.

If any problems are encountered, feel free to contact us:

* Manuel Otero: manuel.otero@estudiante.uam.es
* Javier Muñoz: javier.munnozharo@estudiante.uam.es

    
