import open3d as o3d
import numpy as np
import os
import collections

# defines
sphere_radius = 1
distance_to_camera = 0.01
baseline = 0.8


BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
    
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")


if __name__=='__main__':
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    height_to_base_ratio = baseline / (sphere_radius + distance_to_camera)
    print(f"ratio: {height_to_base_ratio}")
    angular_step = np.arctan(height_to_base_ratio)
    print(f"angular_step: {angular_step}")
    print(f"angle: {np.rad2deg(angular_step)}")
    
    frames = []
    images = {}
    
    frame_pos = np.array([sphere_radius + distance_to_camera, 0, 0])
    angles = np.arange(0, 2 * np.pi, angular_step)

    sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
    sphere.compute_vertex_normals()
    
    image_id = 0

    for phi in angles[:-1]:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # frame.translate(frame_pos)
        R1 = frame.get_rotation_matrix_from_axis_angle([0, -np.pi/2, 0])
        R2 = frame.get_rotation_matrix_from_axis_angle([0, 0, -phi])
        frame.rotate(R2@R1)
        qvec = rotmat2qvec(R2@R1)
        
        # translate frame relative to objects' frame
        R = frame.get_rotation_matrix_from_axis_angle([0, 0, phi])
        frame.translate(frame_pos @ R)
        tvec = frame_pos @ R
        
        image_id += 1        
        images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=image_id, name=f'{image_id}.jpg')

        frames.append(frame)


    sphere_radius1 = sphere_radius * np.cos(angular_step)
    height_to_base_ratio1 = baseline / (sphere_radius1 + distance_to_camera)
    print(f"ratio: {height_to_base_ratio1}")
    angular_step1 = np.arctan(height_to_base_ratio1)
    print(f"angular_step: {angular_step1}")
    print(f"angle: {np.rad2deg(angular_step1)}")


    frame_pos = np.array([sphere_radius1 + distance_to_camera, 0, 0.5])
    angles = np.arange(0, 2 * np.pi, angular_step1)    

    for phi in angles:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        R1 = frame.get_rotation_matrix_from_axis_angle([0, -np.pi/2, 0])
        R2 = frame.get_rotation_matrix_from_axis_angle([0, 0, -phi])
        R3 = frame.get_rotation_matrix_from_axis_angle([0, -np.pi/6, 0])
        frame.rotate(R2 @ R1 @ R3)
        qvec = rotmat2qvec(R2@R1@R3)
        
        # translate frame relative to objects' frame
        R = frame.get_rotation_matrix_from_axis_angle([0, 0, phi])
        frame.translate(frame_pos @ R)
        tvec = frame_pos @ R

        image_id += 1        
        images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=image_id, name=f'{image_id}.jpg')
        
        frames.append(frame)


    frame_pos = np.array([sphere_radius1 + distance_to_camera, 0, -0.5])
    angles = np.arange(0, 2 * np.pi, angular_step1)    

    for phi in angles:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        R1 = frame.get_rotation_matrix_from_axis_angle([0, -np.pi/2, 0])
        R2 = frame.get_rotation_matrix_from_axis_angle([0, 0, -phi])
        R3 = frame.get_rotation_matrix_from_axis_angle([0, np.pi/6, 0])
        frame.rotate(R2@R1@R3)
        qvec = rotmat2qvec(R2@R1@R3)
        
        # translate frame relative to objects' frame
        R = frame.get_rotation_matrix_from_axis_angle([0, 0, phi])
        frame.translate(frame_pos @ R)
        tvec = frame_pos @ R

        image_id += 1        
        images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=image_id, name=f'{image_id}.jpg')

        frames.append(frame)


    print(f"number of images: {len(frames)}")
    write_images_text(images=images, path="images.txt")
    frames.append(sphere)
    o3d.visualization.draw_geometries(frames)
