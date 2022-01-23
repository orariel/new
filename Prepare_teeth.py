import copy
import numpy as np
import open3d as o3d
import pymeshfix as mf
import pyvista as pv
from stl import mesh
import alphashape
from shapely.geometry import Point



def fill_tooth(filename):
    tooth_path = pv.read(filename)
    tooth = mf.MeshFix(tooth_path)
    tooth.repair(verbose=False, joincomp=False, remove_smallest_components=True)
    return tooth
""""
Filling tooth individually 
 input: path to the hollow tooth_num.stl

    Parameters
    ----------
    filename : str
        The string path to the file to read. 
  
    Returns
    -------
    full tooth_num.stl
"""


def clean_pcd(pcd):
    pcd = np.ascontiguousarray(pcd)
    unique_pcd = np.unique(pcd.view([('', pcd.dtype)] * pcd.shape[1]))
    return unique_pcd.view(pcd.dtype).reshape((unique_pcd.shape[0], pcd.shape[1]))

""""
This function remove duplicate records for the Point Cloud
 input: Numpy array of Point Cloud.points . [N_rows,3_Columns]

    Returns
    -------
    PCD matrix that each record is unique
"""


def fill_teeth(teeth_filename):
    teeth_path = pv.read(teeth_filename)
    teeth = mf.MeshFix(teeth_path)
    teeth.repair(verbose=False, joincomp=False, remove_smallest_components=True)
    return teeth


""""
Filling the all teeth 
 input: path to the hollow teeth.stl

    Parameters
    ----------
    filename : str
        The string path to the file to read. 

    Returns
    -------
    full teeth.stl
"""
def gltf_to_polydata(gltf_filename):
    p1 = pv.read(gltf_filename)
    mesh_polydata = p1[0][0][0]
    return mesh_polydata
""""
Converting .GLTF to Polydata
 input: path to the hollow teeth.GLTF

    Parameters
    ----------
    filename : str
        The string path to the file to read. 

    Returns
    -------
    teeth (PolyData)
"""

def Prepre_for_vucuum_forming(teeth_original_filename,Euler_angles_vec):

    filled_teeth_unclean= fill_teeth(teeth_original_filename)
    teeth_filled_filename="STL'S/save_only_for_read_agin.stl"
    filled_teeth_unclean.save(teeth_filled_filename)
    #<------------------------------------------Stl to PCD (original)----------------------------------------------------------------->
    meshes_original = np.array(mesh.Mesh.from_file(teeth_original_filename))
    points_original = meshes_original.reshape(len(meshes_original) * 3, 3)  # point_cloud
    indexes = np.array([x for x in range(len(points_original))]).reshape(len(meshes_original), 3)
    vertices = copy.deepcopy(points_original)
    faces = copy.deepcopy(indexes)
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]
    pcd_original = o3d.geometry.PointCloud()
    points_original = (clean_pcd(points_original))
    pcd_original.points = o3d.utility.Vector3dVector(points_original)
    filled_teeth_unclean="STL'S/filled_teeth_unclean.stl"
    meshes_filled = np.array(mesh.Mesh.from_file(filled_teeth_unclean))
    #<------------------------------------------Stl to PCD (filled teeth and unclean)----------------------------------------------------------------->
    points_filled = meshes_filled.reshape(len(meshes_filled) * 3, 3)  # point_cloud
    indexes_filled = np.array([x for x in range(len(points_filled))]).reshape(len(meshes_filled), 3)
    vertices_filled = copy.deepcopy(points_filled)
    faces_filled = copy.deepcopy(indexes_filled)
    cube_filled = mesh.Mesh(np.zeros(faces_filled.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_filled):
        for j in range(3):
            cube_filled.vectors[i][j] = vertices_filled[f[j], :]
    pcd_filled = o3d.geometry.PointCloud()
    points_filled = (clean_pcd(points_filled))
    pcd_filled.points = o3d.utility.Vector3dVector(points_filled)

    pcd_original_split = np.hsplit(points_original, 3)
    x_cor_original = pcd_original_split[0]
    y_cor_original = pcd_original_split[1]
    xy_original_mat = np.concatenate((x_cor_original, y_cor_original), axis=1)

    pcd_filled_split = np.hsplit(points_filled, 3)
    x_cor_filled = pcd_filled_split[0]
    y_cor_filled = pcd_filled_split[1]
    #<------------------------------------------Check if the point(x,y) is in the alfa shape boundary  ----------------------------------------------------------------->
    alpha_shape = alphashape.alphashape(xy_original_mat, 2.0)
    clean_arr_index = []
    i = 0
    dem = int(x_cor_filled.size)
    for t in range(dem):
        x = x_cor_filled[t]
        y = y_cor_filled[t]
        point = Point(x, y)
        if alpha_shape.contains(point) == False:
            clean_arr_index.append(t)
    arr_clean = np.delete(points_filled, clean_arr_index, axis=0)
    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = o3d.utility.Vector3dVector(arr_clean)
    #<------------------------------------------PCD to STL using possion method  ----------------------------------------------------------------->
    pcd_clean.estimate_normals()
    density = 0.00000008
    tengent_plane = 100
    depth_mesh = 9
    pcd_clean.orient_normals_consistent_tangent_plane(tengent_plane)
    mesh_cleaned, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=depth_mesh)
    vertices_to_remove = densities < np.quantile(densities, density)
    mesh_cleaned.remove_vertices_by_mask(vertices_to_remove)
    mesh_cleaned.compute_triangle_normals()
    o3d.io.write_triangle_mesh("STL'S/gum_not_filled.stl'",mesh_cleaned)
    teeth_gum_filled = pv.read("STL'S/gum_not_filled.stl")
    teeth_gum_filled = mf.MeshFix(teeth_gum_filled)
    teeth_gum_filled.repair(verbose=False, joincomp=False, remove_smallest_components=True)
    points_ = teeth_gum_filled.points()
    faces_= teeth_gum_filled.faces()
    teeth_gum_filled = pv.make_tri_mesh(points_, faces_)

    #<------------------------------------------Create box to align  the teeth for Vuucum forming----------------------------------------------------------------->

    axes = pv.Axes()
    axes.origin = (0.0, 0.0, 0.0)
    teeth_gum_filled.rotate_x(Euler_angles_vec[0], point=axes.origin)
    teeth_gum_filled.rotate_y(Euler_angles_vec[1], point=axes.origin)
    teeth_gum_filled.rotate_z(Euler_angles_vec[2], point=axes.origin)
    box_cor = np.asarray(teeth_gum_filled.bounds)
    b_box = pv.Box(bounds=box_cor, level=0)
    b_box.translate((0, 0,box_cor[3]-2)) #The distance between the box and the teeth
    b_box.save('b_box.stl')
    b_box=pv.read('b_box.stl')
    p1 = pv.Plotter()
    results = b_box.merge(teeth_gum_filled)
    p1.add_mesh(results)
    p1.show()
    results.save('STL/vacuum_forming_N.stl')#Saving command
    return teeth_gum_filled
""""
The function add gum to the filled and unclean stl.
Stages:
1.Converting to PCD_original and PCD_filled
2.Define the original boundary using alfa shape.
3.Check if every point(X,Y) from PCD_filled is in the boundary of PCD_original
4.Convert to mesh using poisson method
5.Filled the remains holes.
6. Create a boundary box for the base of the platte and concatenate the SLT's
 input: path to the hollow teeth.stl

    Parameters
    ----------
    teeth_original_filename : str
        The string path to the file to read.
    Euler_angles_vec: Numpy array of (1x3)
        vector that contain the 3 angles of rotation. Using pyvisa you can rotate each time around one axis.

    Returns
    -------
    full teeth on a box-plate (PolyData file) . for saving file : teeth.save("teeth.stl")
"""
Euler_angles_vec=np.array([1, 2, 5])
Prepre_for_vucuum_forming("STL'S/to_print.stl",Euler_angles_vec)



