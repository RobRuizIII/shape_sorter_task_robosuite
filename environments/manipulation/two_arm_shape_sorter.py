import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
from robosuite.utils.observables import Observable, sensor

from models.objects.composite_body.shape_sorter import ShapeSorterObject
from models.objects.composite.shape_pegs import CrossPegObject, DiamondPegObject, PentagonPegObject, TrianglePegObject

# Adapted from two_arm_peg_in_hole.py manipulation task


class TwoArmShapeSorter(TwoArmEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        side_of_cube_half_size=(0.165, 0.01),
        target_shape="square",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # Assert that the gripper type is None
        assert gripper_types is None, "Tried to specify gripper other than None in TwoArmPegInHole environment!"

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # save target shape
        possible_shapes = ["square", "cross", "diamond", "triangle", "pentagon"]
        self.target_shape = "".join(target_shape.split()).lower()
        assert self.target_shape in possible_shapes, "Target shape must be in " + possible_shapes


        # Save peg specs
        self.plate_half_size = np.array([side_of_cube_half_size[0], side_of_cube_half_size[0], side_of_cube_half_size[1]])
        self.peg_size = np.array([side_of_cube_half_size[0] / 3 * .9, side_of_cube_half_size[0] / 3 * .9, self.plate_half_size[0] * .65])

        self.d_val_check = self.plate_half_size[0] * 0.1
        self.t_val_check_pos = self.peg_size[2] + self.plate_half_size[2]
        self.t_val_check_neg = - (self.peg_size[2] - self.plate_half_size[2])
        self.cos_val_check = 0.97

 

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):

        reward = 0

        # Right location and angle
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            # Grab relevant values
            t, d, cos = self._compute_orientation()
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, scale sparse reward so that the max reward is identical to its dense version
        else:
            reward *= 5.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 5.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["empty"]
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # Add arena and robot
        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0666432116509934, 1.4903257668114777e-08, 2.0563394967349096],
            quat=[0.6530979871749878, 0.27104058861732483, 0.27104055881500244, 0.6530978679656982],
        )

        # initialize objects of interest

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="peg_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        

        self.shape_sorter_cube = ShapeSorterObject(
            name = "shape_sorter_cube",
            plate_half_size=self.plate_half_size,
            density=10.0,
        )

        self.hole = None
        self.peg = None
        peg_name = "peg"

        if (self.target_shape == "square"):
            
            self.peg = BoxObject(
                name=peg_name,
                size = self.peg_size,
                material=greenwood,
                rgba=[0, 1, 0, 1],
                joints=None,
            )

            self.hole = self.shape_sorter_cube.square_plate
        
        elif (self.target_shape == "cross"):
            
            self.peg = CrossPegObject(
                name = peg_name
            )

            self.hole = self.shape_sorter_cube.cross_plate

        elif (self.target_shape == "diamond"):
            
            self.peg = DiamondPegObject(
                name = peg_name
            )

            self.hole = self.shape_sorter_cube.diamond_plate

        elif (self.target_shape == "triangle"):
            
            self.peg = TrianglePegObject(
                name = peg_name
            )

            self.hole = self.shape_sorter_cube.triangle_plate
        
        elif (self.target_shape == "pentagon"):
            
            self.peg = PentagonPegObject(
                name = peg_name
            )

            self.hole = self.shape_sorter_cube.pentagon_plate


        # Load shape sorter object
        shape_sorter_obj = self.shape_sorter_cube.get_obj()
        shape_sorter_obj.set("quat", "1 0 0 0")
        shape_sorter_obj.set("pos", f"0 0 {self.plate_half_size[0] * .75 + self.plate_half_size[2]}")


        # Load peg object
        peg_obj = self.peg.get_obj()
        peg_obj.set("pos", array_to_string((0, 0, self.plate_half_size[0] * .65)))

        # Append appropriate objects to arms
        if self.env_configuration == "bimanual":
            r_eef, l_eef = [self.robots[0].robot_model.eef_name[arm] for arm in self.robots[0].arms]
            r_model, l_model = [self.robots[0].robot_model, self.robots[0].robot_model]
        else:
            r_eef, l_eef = [robot.robot_model.eef_name for robot in self.robots]
            r_model, l_model = [self.robots[0].robot_model, self.robots[1].robot_model]
        r_body = find_elements(root=r_model.worldbody, tags="body", attribs={"name": r_eef}, return_first=True)
        l_body = find_elements(root=l_model.worldbody, tags="body", attribs={"name": l_eef}, return_first=True)
        r_body.append(peg_obj)
        l_body.append(shape_sorter_obj)

        # task includes arena, robot, and objects of interest
        # We don't add peg and hole directly since they were already appended to the robots
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.shape_sorter_cube)
        self.model.merge_assets(self.peg)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id("shape_sorter_cube_" + self.hole.root_body)
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            if self.env_configuration == "bimanual":
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
            else:
                pf0 = self.robots[0].robot_model.naming_prefix
                pf1 = self.robots[1].robot_model.naming_prefix
            modality = "object"

            # position and rotation of peg and hole
            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hole_body_id])

            @sensor(modality=modality)
            def hole_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.hole_body_id], to="xyzw")

            @sensor(modality=modality)
            def peg_to_hole(obs_cache):
                return (
                    obs_cache["hole_pos"] - np.array(self.sim.data.body_xpos[self.peg_body_id])
                    if "hole_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")

            # Relative orientation parameters
            @sensor(modality=modality)
            def angle(obs_cache):
                t, d, cos = self._compute_orientation()
                obs_cache["t"] = t
                obs_cache["d"] = d
                return cos

            @sensor(modality=modality)
            def t(obs_cache):
                return obs_cache["t"] if "t" in obs_cache else 0.0

            @sensor(modality=modality)
            def d(obs_cache):
                return obs_cache["d"] if "d" in obs_cache else 0.0

            sensors = [hole_pos, hole_quat, peg_to_hole, peg_quat, angle, t, d]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        t, d, cos = self._compute_orientation()

        return d < self.d_val_check and self.t_val_check_neg<= t <= self.t_val_check_pos and cos > self.cos_val_check

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos

        t = (center - peg_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, peg_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)),
        )

    def _peg_pose_in_hole_frame(self):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.

        Returns:
            np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos(self.peg.root_body)
        peg_rot_in_world = self.sim.data.get_body_xmat(self.peg.root_body).reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        # World frame
        hole_pos_in_world = self.sim.data.get_body_xpos(self.hole.root_body)
        hole_rot_in_world = self.sim.data.get_body_xmat(self.hole.root_body).reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(peg_pose_in_world, world_pose_in_hole)
        return peg_pose_in_hole
