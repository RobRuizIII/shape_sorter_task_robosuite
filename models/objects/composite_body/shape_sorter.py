import numpy as np

from robosuite.models.objects import BoxObject, CompositeBodyObject, CylinderObject
from robosuite.utils.mjcf_utils import BLUE, RED, CustomMaterial, array_to_string

from models.objects.composite.shape_sorter_plates import ShapeSorterBaseObject, SquareHolePlateObject, TriangleHolePlateObject, DiamondHolePlateObject, PentagonHolePlateObject, CrossHolePlateObject

class ShapeSorterObject(CompositeBodyObject):

    def __init__(
        self,
        name,
        plate_half_size=(0.165, 0.165, 0.01),
        quat = None,
        density = 100.0,
        amt_box_compose_triangle = 16,
    ):

        # length and height must be equal

        (half_length, half_height, half_width) = plate_half_size
        
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }

        redwood_material = CustomMaterial(
            texture="WoodRed",
            tex_name="red-wood",
            mat_name="plate_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.base_plate = ShapeSorterBaseObject(
            name = "base_plate",
            plate_half_size = plate_half_size,
            quat = quat,
            density = density
        )


        self.square_plate = SquareHolePlateObject(
            name = "square_plate",
            plate_half_size = plate_half_size,
            quat = quat,
            density = density,
        )

        self.cross_plate = CrossHolePlateObject(
            name = "cross_plate",
            plate_half_size = plate_half_size,
            quat = quat,
            density = density,
        )

        self.triangle_plate = TriangleHolePlateObject(
            name = "triangle_plate",
            plate_half_size = plate_half_size,
            quat = quat,
            density = density,
            amt_box_compose_triangle = amt_box_compose_triangle,
        )

        self.diamond_plate = DiamondHolePlateObject(
            name = "diamond_plate",
            plate_half_size = plate_half_size,
            quat = quat,
            density = density,
            amt_box_compose_triangle = amt_box_compose_triangle,
        )

        self.pentagon_plate = PentagonHolePlateObject(
            name = "pentagon_plate",
            plate_half_size = plate_half_size,
            quat = quat,
            density = density,
            amt_box_compose_triangle = amt_box_compose_triangle,
        )



        self.square_plate_center = np.array([-(half_width + half_length), 0, (half_height + half_width)]) # left
        self.cross_plate_center = np.array([(half_width + half_length), 0, (half_height + half_width)]) # right
        self.diamond_plate_center = np.array([0, -(half_width + half_height), (half_height + half_width)]) # front
        self.triangle_plate_center = np.array([0, (half_width + half_height), (half_height + half_width)]) # back
        self.pentagon_plate_center = np.array([0, 0, (half_height + half_width) * 2]) # top

        self.square_plate_quat = np.array([0.707, 0, 0.707, 0]) # left
        self.cross_plate_quat = np.array([0.707, 0, 0.707, 0]) # right
        self.diamond_plate_quat = np.array([0.707, 0.707, 0, 0]) # front
        self.triangle_plate_quat= np.array([0.707, 0.707, 0, 0]) # back
        self.pentagon_plate_quat = quat # top



        
        # need boxes to solidify shape sorter object (4 extended length and 8 small length)
        
        connecting_boxes_long = []
        for i in range(4):
            connecting_boxes_long.append(
                BoxObject(
                    name=f"connecting_box_long_{i + 1}",
                    size=np.array([(half_length + half_width * 2), half_width, half_width]),
                    material=redwood_material,
                )
            )

        connecting_boxes_short = []
        for i in range(8):
            connecting_boxes_short.append(
                BoxObject(
                    name=f"connecting_box_short_{i + 1}",
                    size=np.array([half_length, half_width, half_width]),
                    material=redwood_material,
                )
            )
        

        
        # define positions and quats for connecting_boxes_long
        # for square_plate_side and cross_plate_side place below and above
        connecting_pos_long = [
            np.array([self.square_plate_center[0], 0, 0]), # left_bottom
            np.array([self.cross_plate_center[0], 0, 0]), # right_bottom
            np.array([self.square_plate_center[0], 0, self.pentagon_plate_center[2]]), # left_top
            np.array([self.cross_plate_center[0], 0, self.pentagon_plate_center[2]]), # right_top
        ]
        
        connecting_quat_long = [
            self.square_plate_quat,
            self.cross_plate_quat,
            self.square_plate_quat,
            self.cross_plate_quat,
        ]

        connecting_quat_long = [np.array([0, 0.707, 0.707, 0]) for i in range(4)]

        
        # define positions and quats for connecting_boxes_short
        # for rest of empty space
        connecting_pos_short = [
            np.array([0, self.diamond_plate_center[1], 0]), # front_bottom
            np.array([0, self.triangle_plate_center[1], 0]), # back_bottom

            np.array([0, self.diamond_plate_center[1], self.pentagon_plate_center[2]]), # front_top
            np.array([0, self.triangle_plate_center[1], self.pentagon_plate_center[2]]), # back_top

            np.array([self.square_plate_center[0], self.diamond_plate_center[1], self.square_plate_center[2]]), # front_left
            np.array([self.square_plate_center[0], self.triangle_plate_center[1], self.square_plate_center[2]]), # back_left

            np.array([self.cross_plate_center[0], self.diamond_plate_center[1], self.cross_plate_center[2]]), # front_right
            np.array([self.cross_plate_center[0], self.triangle_plate_center[1], self.cross_plate_center[2]]), # back_right
        ]

        connecting_quat_short = [self.diamond_plate_quat] * 4 + [self.square_plate_quat] * 4

        

        objects = [
            self.base_plate,
            self.square_plate,
            self.cross_plate,
            self.diamond_plate,
            self.triangle_plate,
            self.pentagon_plate
        ]

        positions = [
            np.zeros(3),
            self.square_plate_center,
            self.cross_plate_center,
            self.diamond_plate_center,
            self.triangle_plate_center,
            self.pentagon_plate_center,
        ]

        quats = [
            quat,
            self.square_plate_quat,
            self.cross_plate_quat,
            self.diamond_plate_quat,
            self.triangle_plate_quat,
            self.pentagon_plate_quat,
        ]
        

        objects = objects + connecting_boxes_long + connecting_boxes_short
        positions = positions + connecting_pos_long + connecting_pos_short
        quats = quats + connecting_quat_long + connecting_quat_short

        parents = [None for i in range(len(objects))]

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=positions,
            object_quats=quats,
            object_parents=parents,
            joints=None,
            body_joints={},
        )

