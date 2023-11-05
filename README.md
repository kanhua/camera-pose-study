# Playground of studying 3D reconstruction, camera pose and camera calibration 

## Install colmap and pycolmap

I installed the following versions of colmap and pycolmap:
- pycolmap: `0f74d3dcddbb134d058df6e8e484dc714998d7b0`
- colmap: `0478d88a4eb695d7c89a8313a30f0061803522fa`


## Calculate the relative pose between two images:

- script: `get_rlative_pose.py`

- This script uses the matched results by colmap and calculate the relative pose between two images.

The dataset is downloaded from [colmap website](https://colmap.github.io/datasets.html). "South-Building" is used in this script.


## Tips of using pycolmap

```python


for image_id, image in reconstruction.images.items():
   print(image_id, image) 
   print(image.points2D[0].point3D_id)  # get the point3D_id of the first point2D of the "image"
   print(image.get_valid_point2D_ids()) # get the list of all the point2D ids whose point_3D_id is not -1
   for valid_point_2D_id in image.get_valid_point2D_ids():
       print(image.points2D[valid_point_2D_id].point3D_id)  # get the point3D_id of the valid point2D
   print(image.ListPoint2D)

 for point3D_id, point3D in reconstruction.points3D.items():
    print(point3D_id, point3D)

 for camera_id, camera in reconstruction.cameras.items():
    print(camera_id, camera)

```