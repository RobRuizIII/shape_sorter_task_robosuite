import numpy as np
import copy

from robosuite.models.objects import BoxObject, CylinderObject, CompositeObject, CompositeBodyObject
from robosuite.utils.mjcf_utils import BLUE, RED, CYAN, GREEN, CustomMaterial, array_to_string


class CrossPegObject(CompositeObject):
    def __init__(
        self,
        name,
        peg_half_size=(0.0495, 0.0495, 0.1),
        quat=None,
        density=100.0,
    ):

        # amt_box_compose_triangle must be even or have to check and make even before tot_pieces calc
        amt_box_compose_triangle=4

        self._name = name
        self.amt_box_compose_triangle = amt_box_compose_triangle

        
        # Set materials for each box
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }


        custom_mat = CustomMaterial(
            texture="WoodGreen",
            tex_name="green-wood",
            mat_name="peg_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )



        tot_pieces = amt_box_compose_triangle * 2
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["peg_mat" for i in range(tot_pieces)]

        
        self.left_edge_center = np.array([-peg_half_size[0], 0, 0])
        self.right_edge_center = np.array([peg_half_size[0], 0, 0])
        self.top_edge_center = np.array([0, peg_half_size[1], 0])
        self.bottom_edge_center = np.array([0, -peg_half_size[1], 0])

        top_pos, top_sizes = _triangle_creator_peg(np.zeros(3), peg_half_size, self.amt_box_compose_triangle, cover_percent=0.5, top=True)
        bot_pos, bot_sizes = _triangle_creator_peg(np.zeros(3), peg_half_size, self.amt_box_compose_triangle, cover_percent=0.5, top=False)

        geom_sizes = top_sizes + bot_sizes

        geom_locations = top_pos + bot_pos
        

        super().__init__(
            name=self.name,
            total_size=peg_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(custom_mat)


class TrianglePegObject(CompositeObject):
    def __init__(
        self,
        name,
        peg_half_size=(0.0495, 0.0495, 0.1),
        quat=None,
        density=100.0,
        amt_box_compose_triangle=16,
    ):

        # amt_box_compose_triangle must be even or have to check and make even before tot_pieces calc

        self._name = name
        self.amt_box_compose_triangle = amt_box_compose_triangle

        
        # Set materials for each box
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }


        custom_mat = CustomMaterial(
            texture="WoodGreen",
            tex_name="green-wood",
            mat_name="peg_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )


        tot_pieces = amt_box_compose_triangle
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["peg_mat" for i in range(tot_pieces)]

        
        self.left_edge_center = np.array([-peg_half_size[0], 0, 0])
        self.right_edge_center = np.array([peg_half_size[0], 0, 0])
        self.top_edge_center = np.array([0, peg_half_size[1], 0])
        self.bottom_edge_center = np.array([0, -peg_half_size[1], 0])

        top_pos, top_sizes = _triangle_creator_peg(np.zeros(3), peg_half_size, self.amt_box_compose_triangle, cover_percent=1.0, top=True)
        

        geom_sizes = top_sizes

        geom_locations = top_pos
        

        super().__init__(
            name=self.name,
            total_size=peg_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(custom_mat)




class DiamondPegObject(CompositeObject):
    def __init__(
        self,
        name,
        peg_half_size=(0.0495, 0.0495, 0.1),
        quat=None,
        density=100.0,
        amt_box_compose_triangle=16,
    ):

        # amt_box_compose_triangle must be even or have to check and make even before tot_pieces calc

        self._name = name
        self.amt_box_compose_triangle = amt_box_compose_triangle

        
        # Set materials for each box
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }


        custom_mat = CustomMaterial(
            texture="WoodGreen",
            tex_name="green-wood",
            mat_name="peg_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )



        tot_pieces = amt_box_compose_triangle * 2
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["peg_mat" for i in range(tot_pieces)]

        
        self.left_edge_center = np.array([-peg_half_size[0], 0, 0])
        self.right_edge_center = np.array([peg_half_size[0], 0, 0])
        self.top_edge_center = np.array([0, peg_half_size[1], 0])
        self.bottom_edge_center = np.array([0, -peg_half_size[1], 0])

        top_pos, top_sizes = _triangle_creator_peg(np.zeros(3), peg_half_size, self.amt_box_compose_triangle, cover_percent=0.5, top=True)
        bot_pos, bot_sizes = _triangle_creator_peg(np.zeros(3), peg_half_size, self.amt_box_compose_triangle, cover_percent=0.5, top=False)

        geom_sizes = top_sizes + bot_sizes

        geom_locations = top_pos + bot_pos
        

        super().__init__(
            name=self.name,
            total_size=peg_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(custom_mat)



class PentagonPegObject(CompositeObject):
    def __init__(
        self,
        name,
        peg_half_size=(0.0495, 0.0495, 0.1),
        quat=None,
        density=100.0,
        amt_box_compose_triangle=16,
    ):

        # amt_box_compose_triangle must be even or have to check and make even before tot_pieces calc

        self._name = name
        self.amt_box_compose_triangle = amt_box_compose_triangle

        
        # Set materials for each box
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }


        custom_mat = CustomMaterial(
            texture="WoodGreen",
            tex_name="green-wood",
            mat_name="peg_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )


        tot_pieces = amt_box_compose_triangle + 1
        geom_types = ["box" for i in range(tot_pieces)]
        quats = [quat for i in range(tot_pieces)] if quat is not None else None
        geom_materials = ["peg_mat" for i in range(tot_pieces)]

        
        self.left_edge_center = np.array([-peg_half_size[0], 0, 0])
        self.right_edge_center = np.array([peg_half_size[0], 0, 0])
        self.top_edge_center = np.array([0, peg_half_size[1], 0])
        self.bottom_edge_center = np.array([0, -peg_half_size[1], 0])

        top_pos, top_sizes = _triangle_creator_peg(np.zeros(3), peg_half_size, self.amt_box_compose_triangle, cover_percent=0.5, top=True)
        

        geom_sizes = [np.array([peg_half_size[0], peg_half_size[1] / 2, peg_half_size[2]])] + top_sizes

        geom_locations = [np.array([0, -peg_half_size[1] / 2, 0])] + top_pos
        

        super().__init__(
            name=self.name,
            total_size=peg_half_size,
            geom_types=np.array(geom_types),
            geom_sizes=np.array(geom_sizes),
            geom_quats=quats,
            geom_locations=geom_locations,
            geom_materials=geom_materials,
            density=density,
            locations_relative_to_center=True,
            joints=None,
            )

        self.append_material(custom_mat)

    


def _triangle_creator_peg(center, size, sides, cover_percent=1.0, top=True):
    # center: (x, y, z)
    # size: half-size of box to make triangle
    # sides: must be >= 2
    
    if (sides % 2 != 0):
        sides += 1
    

    half_sides = sides // 2
    (length, height, width) = size
    (x_center, y_center, z_center) = center
    triangle_height = height * cover_percent
    y_center_change = -1 * (1 - cover_percent) * height

    
    half_length_change = length / sides
    half_height_change = triangle_height / half_sides

    if (top):
        half_height_change *= -1
        y_center_change *= -1

    y_center += y_center_change

    left_positions = [np.array([x_center - half_length_change - (i * 2 * half_length_change), y_center + (i * half_height_change), z_center]) for i in range(half_sides)]
    left_sizes = [np.array([half_length_change, triangle_height - (i * np.abs(half_height_change)), width]) for i in range(half_sides)]

    right_positions = [np.array([x_center + half_length_change + (i * 2 * half_length_change), y_center + (i * half_height_change), z_center]) for i in range(half_sides)]
    right_sizes = copy.deepcopy(left_sizes)

    return left_positions + right_positions, left_sizes + right_sizes