# F1TenthTeam8FinalProject

In order to run our code, you must first download the file `nerf_imitation_model_lam_1.pth` from the Google Drive link we shared with the teaching team. Place this file in the directory `final_project/share`. You can then build the package as normal and run it. The `.pth` file could not be provided here due to its large size.

---

We have also shared the file `f1tenth_nerf.zip` with the teaching team on Google Drive. This file includes all of the content we used to generate the NeRFs for our project using NerfStudio. After unzipping this file, you can follow the following procedure we used to train our NeRFs:

- We first activated our conda environment using the command `conda activate nerfstudio`.

- All of the raw training data collected from the car is contained in the directories `f1tenth_nerf/may_2_frames` and `f1tenth_nerf/slow_frames` as individual images, and in the various .mp4 files within the `f1tenth_nerf` directory as videos. To preprocess the video data with COLMAP, we ran the command `ns-process-data video --data {path to video.mp4} --output-dir {path to output directory}`. Thus, for each video `output_VIDEO_NAME.mp4`, we created a directory `f1tenth_nerf/outputs_VIDEO_NAME` containing all of the COLMAP data for the frames in that video.

- Then we trained the NeRF for a selected directory of COLMAP data by running the command `ns-train nerfacto --data {path to output directory, same as above}`. This would create a new directory in the `f1tenth_nerf/outputs` directory, containing config files from which the NeRFs could be viewed by running `ns-viewer --load-config {path to config file listed in terminal output during training}`.

- We rendered many frames from the NeRFs to generate our training images, and these are held in the `f1tenth_nerf/renders` directory. This directory contains both subdirectories containing the individual rendered frames, as well as videos of the NeRF environments.

---

Given our NeRF renderings, we trained our model using the file `train_and_validation.py`, provided in this repository. Also provided is our dataset file, `dataset.csv` which is a list of poses alongside the NeRF rendering and the action taken by pure pursuit corresponding to each pose. `train_and_validation.py` references `dataset.csv`, which in turn references the NeRF-rendered image files in `f1tenth_nerf.zip`. Also note that `train_and_validation.py` contains an absolute file path in Line 24, so you will have to change the path here to your home directory in order to run `train_and_validation.py` yourself.

