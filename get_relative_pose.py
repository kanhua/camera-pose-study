import pycolmap
from pycolmap import Reconstruction, Image
import tqdm
from hloc.utils import viz
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import cv2
import logging

logging.basicConfig(level=logging.INFO)
def get_matches_from_query(reconstruction_model: Reconstruction, query_image_1_index: int, query_image_2_index: int):
    """
    Get matches from query images by going through all the feature points
    :param reconstruction_model:
    :param query_image_1_index:
    :param query_image_2_index:
    :return:
    """
    query_image_1 = reconstruction_model.images[query_image_1_index]
    query_image_2 = reconstruction_model.images[query_image_2_index]
    image_1_coords = []
    image_2_coords = []
    for valid_point_2D_id_1 in tqdm.tqdm(query_image_1.get_valid_point2D_ids()):
        for valid_point_2D_id_2 in query_image_2.get_valid_point2D_ids():
            if query_image_1.points2D[valid_point_2D_id_1].point3D_id == query_image_2.points2D[
                valid_point_2D_id_2].point3D_id:
                logging.info("found: {}".format(query_image_1.points2D[valid_point_2D_id_1].xy))
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


def get_image_from_index(reconstruction_model: Reconstruction, image_index: int, image_dir):
    image_path = image_dir.joinpath(reconstruction_model.images[image_index].name)
    image = cv2.imread(str(image_path))
    return image


def array_converter(arr):
    list_of_arrays = [np.array([row[0], row[1]], dtype=np.float64) for row in arr]
    return list_of_arrays


def main():
    colmap_output = "datasets/south-building/sparse/"
    reconstruction = pycolmap.Reconstruction(colmap_output)
    logging.info(reconstruction.summary())

    # image 3 and image 4 have reasonable number of matches
    query_image_1_index = 3
    query_image_2_index = 4
    image_1_coords, image_2_coords = get_matches_from_3d_points(reconstruction, 3, 4)
    logging.debug(image_1_coords)
    logging.debug(image_2_coords)

    image_dir = pathlib.Path("datasets/south-building/images")

    image_1 = get_image_from_index(reconstruction, query_image_1_index, image_dir)
    image_2 = get_image_from_index(reconstruction, query_image_2_index, image_dir)

    viz.plot_images([image_1, image_2])
    viz.plot_matches(image_1_coords, image_2_coords)
    plt.savefig("./outouts/matches_{:03d}_{:03d}.png".format(query_image_1_index, query_image_2_index))

    answer = pycolmap.fundamental_matrix_estimation(
        image_1_coords,
        image_2_coords
        # [options],  # optional dict or pycolmap.RANSACOptions
    )
    logging.info("Fundamental matrix between these two images: {}".format(answer))

    camera_model = reconstruction.cameras[1]

    answer = pycolmap.essential_matrix_estimation(image_1_coords, image_2_coords, camera_model, camera_model)

    tv_options = pycolmap.TwoViewGeometryOptions()
    tv_options.compute_relative_pose = True

    logging.info("calculate relateive pose:")
    answer = pycolmap.estimate_calibrated_two_view_geometry_from_matches(camera_model, image_1_coords, camera_model,
                                                                         image_2_coords, tv_options)

    logging.info("relative pose: {}".format(answer.cam2_from_cam1))
    logging.info("essential matrix: {}".format(answer.E))

    reconstruction.write(colmap_output)


if __name__ == "__main__":
    main()
