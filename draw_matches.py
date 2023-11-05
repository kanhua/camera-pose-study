import pycolmap
from pycolmap import Reconstruction, Image
import tqdm
from hloc.utils import viz
import matplotlib.pyplot as plt
import numpy as np


def get_matches_from_query(reconstruction_model: Reconstruction, query_image_1_index: int, query_image_2_index: int):
    query_image_1 = reconstruction_model.images[query_image_1_index]
    query_image_2 = reconstruction_model.images[query_image_2_index]
    image_1_coords = []
    image_2_coords = []
    for valid_point_2D_id_1 in tqdm.tqdm(query_image_1.get_valid_point2D_ids()):
        for valid_point_2D_id_2 in query_image_2.get_valid_point2D_ids():
            if query_image_1.points2D[valid_point_2D_id_1].point3D_id == query_image_2.points2D[
                valid_point_2D_id_2].point3D_id:
                print("found: {}".format(query_image_1.points2D[valid_point_2D_id_1].xy))
                image_1_coords.append(query_image_1.points2D[valid_point_2D_id_1].xy)
                image_2_coords.append(query_image_2.points2D[valid_point_2D_id_2].xy)

    return image_1_coords, image_2_coords


def get_matches_from_3d_points(reconstruction_model: Reconstruction, query_image_1_index: int,
                               query_image_2_index: int):
    query_image_1 = reconstruction_model.images[query_image_1_index]
    query_image_2 = reconstruction_model.images[query_image_2_index]

    image_1_coords = []
    image_2_coords = []

    for point3D_id, point3D in reconstruction_model.points3D.items():
        found_image_1_index = None
        found_image_2_index = None
        for track_element in point3D.track.elements:
            if found_image_1_index is None and track_element.image_id == query_image_1_index:
                found_image_1_index = track_element.point2D_idx

            if found_image_2_index is None and track_element.image_id == query_image_2_index:
                found_image_2_index = track_element.point2D_idx
        if found_image_1_index is not None and found_image_2_index is not None:
            image_1_coords.append(query_image_1.points2D[found_image_1_index].xy)
            image_2_coords.append(query_image_2.points2D[found_image_2_index].xy)

    return np.array(image_1_coords), np.array(image_2_coords)


import pathlib
import cv2


def get_image_from_index(reconstruction_model: Reconstruction, image_index: int, image_dir):
    image_path = image_dir.joinpath(reconstruction_model.images[image_index].name)
    image = cv2.imread(str(image_path))
    return image


def main():
    colmap_output = "datasets/south-building/sparse/"
    reconstruction = pycolmap.Reconstruction(colmap_output)
    print(reconstruction.summary())

    query_image_1_index = 3
    query_image_2_index = 4
    image_1_coords, image_2_coords = get_matches_from_3d_points(reconstruction, 3, 4)
    print(image_1_coords)
    print(image_2_coords)

    image_dir = pathlib.Path("datasets/south-building/images")

    image_1 = get_image_from_index(reconstruction, query_image_1_index, image_dir)
    image_2 = get_image_from_index(reconstruction, query_image_2_index, image_dir)

    viz.plot_images([image_1, image_2])
    viz.plot_matches(image_1_coords, image_2_coords)
    plt.show()

    # for image_id, image in reconstruction.images.items():
    #    print(image_id, image)
    #    print(image.points2D[0].point3D_id)
    #    print(image.get_valid_point2D_ids())
    #    for valid_point_2D_id in image.get_valid_point2D_ids():
    #        print(image.points2D[valid_point_2D_id].point3D_id)
    # print(image.ListPoint2D)

    # for point3D_id, point3D in reconstruction.points3D.items():
    #    print(point3D_id, point3D)

    # for camera_id, camera in reconstruction.cameras.items():
    #    print(camera_id, camera)

    reconstruction.write(colmap_output)


if __name__ == "__main__":
    main()
