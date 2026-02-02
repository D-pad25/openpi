from numpy.typing import NDArray
from typing import Optional
import rerun as rr
import spatialgeometry as sg
import trimesh
import roboticstoolbox as rtb
import numpy as np
from spatialmath import UnitQuaternion

class ReRunRobot:
    def __init__(
        self, robot: rtb.Robot, prefix: str, alpha: Optional[float] = None
    ) -> None:
        self.robot = robot
        self.name = f"{prefix}/{robot.name}"
        self.alpha = alpha
        self._load_meshes()
        print(self.name)

    def log_robot_state(self, q: NDArray) -> None:
        """
        Log robot positions to rerun
        q: joint positions
        """
        transforms = self.robot.fkine_all(q)
        rr.log(self.name + "/q", rr.Scalars(q))
        for ix, link in enumerate(self.robot.links):
            O_T_ix: NDArray[np.float64] = transforms[ix + 1].A
            rr.log(
                self.name + "/" + link.name,
                rr.Transform3D(translation=O_T_ix[:3, 3], mat3x3=O_T_ix[:3, :3]),
            )

    def log_pose(
        self, position: NDArray, quaternion: NDArray, name: str = "pose"
    ) -> None:
        """
        Log a 3D pose to rerun
        position: [x, y, z]
        quaternion: [w, x, y, z]
        """
        quad = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]  # xyzw
        rr.log(
            f"{self.name}/{name}",
            rr.Transform3D(
                translation=position, rotation=rr.Quaternion(xyzw=quad), axis_length=0.1
            ),
        )

    def log_frame(self, image, camera_name):
        """
        Log a camera frame to rerun
        image: HxWx3 array
        """
        img = rr.Image(image)
        # Only compress typical RGB uint8 images
        if isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3:
            img = img.compress()
        rr.log(f"{self.name}/{camera_name}", img)



    def log_target_pose(self, target_joint : NDArray):
        """
        Log a camera frame to rerun
        image: HxWx3 array
        """

        transforms = self.robot.fkine(target_joint[0:6])
        position = transforms.t
        quad = UnitQuaternion(transforms.R).A
        rr.log(
            f"{self.name}/target_joint",
            rr.Transform3D(
                translation=position, rotation=rr.Quaternion(xyzw=quad), axis_length=0.1
            ),
        )




    def log_joint_to_pose(self,name,  joints : NDArray):
        """
        Log a camera frame to rerun
        image: HxWx3 array
        """
        i = 0
        for joint in joints:
            transforms = self.robot.fkine(joint[0:6])
            position = transforms.t
            quad = UnitQuaternion(transforms.R).A

            rr.log(
                f"{self.name}/{name}_{i}",
                rr.Transform3D(
                    translation=position, rotation=rr.Quaternion(xyzw=quad), axis_length=0.1
                ),
            )

            rr.log(
                f"{self.name}/{name}__{i}",
                rr.Points3D(positions=position, radii=0.001,)  # radius in meters
            )         
            i = i+1





    def log_grads(self, grad):
        """
        Log a camera frame to rerun
        image: HxWx3 array
        """
        rr.log(self.name + "/grads", rr.Scalars(grad))


    def _apply_alpha(self, vertex_colour: NDArray) -> NDArray:
        if self.alpha is None:
            return vertex_colour
        vertex_colour = np.atleast_2d(vertex_colour)
        vertex_colour[:, -1] = int(self.alpha * 255)
        return vertex_colour

    def _load_meshes(self) -> None:
        for link in self.robot.links:
            if len(link.geometry) == 0:
                continue
            geom = link.geometry[0]
            if isinstance(geom, sg.Mesh):
                mesh = trimesh.load_mesh(geom.filename, process=False)
                # vertex_colour = self._apply_alpha(mesh.visual.to_color().vertex_colors)
                rr.log(
                    f"{self.name}/{link.name}",
                    rr.Mesh3D(
                        vertex_positions=mesh.vertices,
                        triangle_indices=mesh.faces,
                        vertex_normals=mesh.vertex_normals,
                        # vertex_colors=vertex_colour,
                    ),
                    static=True,
                )

    
if __name__ == "__main__":
    robot = rtb.robot.ERobot.URDF("/home/qcrvgs/diffusion_pytorch/asset/xarm6.urdf")
    rr.init("urdf_test", recording_id="testing", spawn=True)
    rr.log(f"/{robot.name}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rerun_robot = ReRunRobot(robot, "test")
    # rerun_robot_2 = ReRunRobot(robot, "transparent", alpha=0.1)
    for _ in range(20):
        rerun_robot.log_robot_state(np.zeros(6))
        
        # rerun_robot_2.log_robot_state(rerun_robot_2.robot.random_q())