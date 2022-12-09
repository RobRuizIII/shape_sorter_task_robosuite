import numpy as np
import copy

from robosuite.models.objects import BoxObject, CylinderObject, CompositeObject, CompositeBodyObject
from robosuite.utils.mjcf_utils import BLUE, RED, CYAN, GREEN, CustomMaterial, array_to_string

class ShapeSorterBaseObject(CompositeObject):
    def __init__(
        self,
        name,
        plate_half_size=(0.165, 0.165, 0.01),
        quat=None,
        density=100.0,
    ):
    
        self._name = name
        

        # Set materials for each box
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

        geom_types = ["box", "cylinder"]
        quats = [quat for i in range(2)] if quat is not None else None
        geom_materials = ["plate_mat" for i in range(2)]
        
        cylinder_handle_len = plate_half_size[1] * 0.75

        geom_sizes = [np.array(plate_half_size), np.array([0.025, cylinder_handle_len])]
        geom_locations = [np.zeros(3), np.array([0, 0, -(plate_half_size[2] + cylinder_handle_len)])]
        

        super().__init__(
            name=self.name,
            total_size=plate_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(redwood_material)

        

class SquareHolePlateObject(CompositeObject):
    def __init__(
        self,
        name,
        plate_half_size=(0.165, 0.165, 0.01),
        quat=None,
        density=100.0,
    ):
    
        self._name = name
        
        hole_sides_half_length = plate_half_size[0] / 3
        hole_sides_half_height = plate_half_size[1] / 3

        self.left_edge_center = np.array([-hole_sides_half_length, 0, 0])
        self.right_edge_center = np.array([hole_sides_half_length, 0, 0])
        self.top_edge_center = np.array([0, hole_sides_half_height, 0])
        self.bottom_edge_center = np.array([0, hole_sides_half_height, 0])

        # left and right box sizes
        side_box_size = np.array((hole_sides_half_length, plate_half_size[1], plate_half_size[2]))

        # middle - top and bottom box sizes
        mid_box_size = np.array((hole_sides_half_length, hole_sides_half_height, plate_half_size[2]))

        # Set materials for each box
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

        geom_types = ["box" for i in range(4)]
        quats = [quat for i in range(4)] if quat is not None else None
        geom_materials = ["plate_mat" for i in range(4)]
        

        geom_sizes = [side_box_size] * 2 + [mid_box_size] * 2


        left_side_center = np.array([-hole_sides_half_length * 2, 0, 0])
        right_side_center = np.array([hole_sides_half_length * 2, 0, 0])
        top_side_center = np.array([0, hole_sides_half_height * 2, 0])
        bottom_side_center = np.array([0, - hole_sides_half_height * 2, 0])

        geom_locations = [left_side_center, right_side_center, top_side_center, bottom_side_center]
        

        super().__init__(
            name=self.name,
            total_size=plate_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(redwood_material)



class CrossHolePlateObject(CompositeObject):
    def __init__(
        self,
        name,
        plate_half_size=(0.165, 0.165, 0.01),
        quat=None,
        density=100.0,
    ):

        # amt_box_compose_triangle must be even or have to check and make even before tot_pieces calc

        amt_box_compose_triangle = 4

        self._name = name
        self.amt_box_compose_triangle = amt_box_compose_triangle

        hole_sides_half_length = plate_half_size[0] / 3
        hole_sides_half_height = plate_half_size[1] / 3

        self.left_edge_center = np.array([-hole_sides_half_length, 0, 0])
        self.right_edge_center = np.array([hole_sides_half_length, 0, 0])
        self.top_edge_center = np.array([0, hole_sides_half_height, 0])
        self.bottom_edge_center = np.array([0, hole_sides_half_height, 0])

        # left and right box sizes
        side_box_size = np.array((hole_sides_half_length, plate_half_size[1], plate_half_size[2]))

        # middle - top and bottom box sizes
        mid_box_size = np.array((hole_sides_half_length, hole_sides_half_height, plate_half_size[2]))

        # Set materials for each box
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

        tot_pieces = amt_box_compose_triangle * 2 + 2
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["plate_mat" for i in range(tot_pieces)]

        


        left_side_center = np.array([-hole_sides_half_length * 2, 0, 0])
        right_side_center = np.array([hole_sides_half_length * 2, 0, 0])
        top_side_center = np.array([0, hole_sides_half_height * 2, 0])
        bottom_side_center = np.array([0, - hole_sides_half_height * 2, 0])

        top_pos, top_sizes = _triangle_creator(top_side_center, mid_box_size, self.amt_box_compose_triangle, cover_percent=0.5, top=True)
        bot_pos, bot_sizes = _triangle_creator(bottom_side_center, mid_box_size, self.amt_box_compose_triangle, cover_percent=0.5, top=False)

        geom_sizes = [side_box_size] * 2 + top_sizes + bot_sizes

        geom_locations = [left_side_center, right_side_center] + top_pos + bot_pos
        

        super().__init__(
            name=self.name,
            total_size=plate_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(redwood_material)





class TriangleHolePlateObject(CompositeObject):
    def __init__(
        self,
        name,
        plate_half_size=(0.165, 0.165, 0.01),
        quat=None,
        density=100.0,
        amt_box_compose_triangle = 16,
    ):
        self._name = name

        self.amt_box_compose_triangle = amt_box_compose_triangle

        hole_sides_half_length = plate_half_size[0] / 3
        hole_sides_half_height = plate_half_size[1] / 3

        self.left_edge_center = np.array([-hole_sides_half_length, 0, 0])
        self.right_edge_center = np.array([hole_sides_half_length, 0, 0])
        self.top_edge_center = np.array([0, hole_sides_half_height, 0])
        self.bottom_edge_center = np.array([0, hole_sides_half_height, 0])

        # left and right box sizes
        side_box_size = np.array((hole_sides_half_length, plate_half_size[1], plate_half_size[2]))

        # middle - top and bottom box sizes
        mid_box_size = np.array((hole_sides_half_length, hole_sides_half_height, plate_half_size[2]))

        # Set materials for each box
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

        tot_pieces = amt_box_compose_triangle + 3
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["plate_mat" for i in range(tot_pieces)]

        


        left_side_center = np.array([-hole_sides_half_length * 2, 0, 0])
        right_side_center = np.array([hole_sides_half_length * 2, 0, 0])
        top_side_center = np.array([0, hole_sides_half_height * 2, 0])
        bottom_side_center = np.array([0, - hole_sides_half_height * 2, 0])

        top_pos, top_sizes = _triangle_creator(top_side_center, mid_box_size, self.amt_box_compose_triangle, top=True)

        geom_sizes = [side_box_size] * 2 + [mid_box_size] + top_sizes

        geom_locations = [left_side_center, right_side_center, bottom_side_center] + top_pos
        

        super().__init__(
            name=self.name,
            total_size=plate_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(redwood_material)


class PentagonHolePlateObject(CompositeObject):
    def __init__(
        self,
        name,
        plate_half_size=(0.165, 0.165, 0.01),
        quat=None,
        density=100.0,
        amt_box_compose_triangle = 16,
    ):
        self._name = name

        self.amt_box_compose_triangle = amt_box_compose_triangle

        hole_sides_half_length = plate_half_size[0] / 3
        hole_sides_half_height = plate_half_size[1] / 3

        self.left_edge_center = np.array([-hole_sides_half_length, 0, 0])
        self.right_edge_center = np.array([hole_sides_half_length, 0, 0])
        self.top_edge_center = np.array([0, hole_sides_half_height, 0])
        self.bottom_edge_center = np.array([0, hole_sides_half_height, 0])

        # left and right box sizes
        side_box_size = np.array((hole_sides_half_length, plate_half_size[1], plate_half_size[2]))

        # middle - top and bottom box sizes
        mid_box_size = np.array((hole_sides_half_length, hole_sides_half_height, plate_half_size[2]))

        # Set materials for each box
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

        tot_pieces = amt_box_compose_triangle + 3
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["plate_mat" for i in range(tot_pieces)]

        


        left_side_center = np.array([-hole_sides_half_length * 2, 0, 0])
        right_side_center = np.array([hole_sides_half_length * 2, 0, 0])
        top_side_center = np.array([0, hole_sides_half_height * 2, 0])
        bottom_side_center = np.array([0, - hole_sides_half_height * 2, 0])

        top_pos, top_sizes = _triangle_creator(top_side_center, mid_box_size, self.amt_box_compose_triangle, cover_percent=0.5, top=True)

        geom_sizes = [side_box_size] * 2 + [mid_box_size] + top_sizes

        geom_locations = [left_side_center, right_side_center, bottom_side_center] + top_pos
        

        super().__init__(
            name=self.name,
            total_size=plate_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(redwood_material)


class DiamondHolePlateObject(CompositeObject):
    def __init__(
        self,
        name,
        plate_half_size=(0.165, 0.165, 0.01),
        quat=None,
        density=100.0,
        amt_box_compose_triangle = 16,
    ):
        self._name = name

        self.amt_box_compose_triangle = amt_box_compose_triangle

        hole_sides_half_length = plate_half_size[0] / 3
        hole_sides_half_height = plate_half_size[1] / 3

        self.left_edge_center = np.array([-hole_sides_half_length, 0, 0])
        self.right_edge_center = np.array([hole_sides_half_length, 0, 0])
        self.top_edge_center = np.array([0, hole_sides_half_height, 0])
        self.bottom_edge_center = np.array([0, hole_sides_half_height, 0])

        # left and right box sizes
        side_box_size = np.array((hole_sides_half_length, plate_half_size[1], plate_half_size[2]))

        # middle - top and bottom box sizes
        mid_box_size = np.array((hole_sides_half_length, hole_sides_half_height, plate_half_size[2]))

        # Set materials for each box
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

        tot_pieces = amt_box_compose_triangle * 2 + 2
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["plate_mat" for i in range(tot_pieces)]

        


        left_side_center = np.array([-hole_sides_half_length * 2, 0, 0])
        right_side_center = np.array([hole_sides_half_length * 2, 0, 0])
        top_side_center = np.array([0, hole_sides_half_height * 2, 0])
        bottom_side_center = np.array([0, - hole_sides_half_height * 2, 0])

        top_pos, top_sizes = _triangle_creator(top_side_center, mid_box_size, self.amt_box_compose_triangle, cover_percent=0.5, top=True)
        bot_pos, bot_sizes = _triangle_creator(bottom_side_center, mid_box_size, self.amt_box_compose_triangle, cover_percent=0.5, top=False)

        geom_sizes = [side_box_size] * 2 + top_sizes + bot_sizes

        geom_locations = [left_side_center, right_side_center] + top_pos + bot_pos
        

        super().__init__(
            name=self.name,
            total_size=plate_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(redwood_material)
    


def _triangle_creator(center, size, sides, cover_percent=1.0, top=True):
    # center: (x, y, z)
    # size: half-size of box to make triangle
    # sides: must be >= 2
    
    if (sides % 2 != 0):
        sides += 1

    half_sides = sides // 2
    (length, height, width) = size
    (x_center, y_center, z_center) = center
    triangle_height = height * cover_percent
    
    half_length_change = length / sides
    half_height_change = triangle_height / half_sides

    if (top):
        half_height_change *= -1


    left_positions = [np.array([x_center - half_length_change - (i * 2 * half_length_change), y_center + (i * half_height_change), z_center]) for i in range(half_sides)]
    left_sizes = [np.array([half_length_change, height + (i * np.abs(half_height_change)), width]) for i in range(half_sides)]

    right_positions = [np.array([x_center + half_length_change + (i * 2 * half_length_change), y_center + (i * half_height_change), z_center]) for i in range(half_sides)]
    right_sizes = copy.deepcopy(left_sizes)

    return left_positions + right_positions, left_sizes + right_sizes