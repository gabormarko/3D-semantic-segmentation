# Utility functions to read COLMAP binary camera and image files
# Based on https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
# (Minimal version for your use case)
import struct
import collections
import numpy as np

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = {
                0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12
            }[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, CAMERA_MODEL_IDS[model_id], width, height, np.array(params))
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = image_properties[0]
            qvec = np.array(image_properties[1:5])
            tvec = np.array(image_properties[5:8])
            camera_id = image_properties[8]
            name = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points2D):
                xy = read_next_bytes(fid, 16, "dd")
                point3D_id = read_next_bytes(fid, 8, "q")[0]
                xys.append(xy)
                point3D_ids.append(point3D_id)
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, np.array(xys), np.array(point3D_ids))
    return images
