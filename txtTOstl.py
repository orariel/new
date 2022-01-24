import numpy as np
import open3d as o3d
import pyvista as pv


source_arr=np.loadtxt("exellent.txt")
p = np.hsplit(source_arr, 4)
# # # -----------------------------slicing arry and making list of teeth-(each element contain np.arry of tooth)------------------------#
dem = int(source_arr.size / 4)
x_cor = p[0]
y_cor = p[1]
z_cor = p[2]
xyz_cor_source = np.concatenate((x_cor, y_cor, z_cor), axis=1)
teeth_seg = p[3]
index_of_tooth = np.array([1])  # arry that includ all teeth index
i=0

xyz_segmented=[]
p=0
for j in range(dem-1):
    if teeth_seg[j] != teeth_seg[j + 1]:
        xyz_segmented.append(xyz_cor_source[p:j+1,:])
        p = j + 1
xyz_segmented.append(xyz_cor_source[p:dem, :])
numOFtooth=len((xyz_segmented))
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind) # pcd with the pcd without the noise
    outlier_cloud = cloud.select_by_index(ind, invert=True)# pcd with only noise
    return inlier_cloud

segmonte_pcd = o3d.geometry.PointCloud()
# # # # -----------------------------looping on teeth ------------------------#
for i in range(numOFtooth):
 segmonte_pcd.points = o3d.utility.Vector3dVector(xyz_segmented[i])
 # # # # -----------------------------remove noise in pcd for each tooth ------------------------#
 cl, ind = segmonte_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
 pcd_i=display_inlier_outlier(segmonte_pcd, ind)
 o3d.visualization.draw_geometries([pcd_i],width=900)

 # # # # ----------------------------Convrting PCD to mesh using PYVISTA ------------------------#
#  xyz_load = np.asarray(pcd_i.points)
#  cloud=pv.PolyData(xyz_load)
#  surf = cloud.delaunay_2d()
#  surf.plot(show_edges=True)
#  surf.save('pcd_to_mesh_pv_'+str(i)+'.stl')
 # # # # ----------------------------Convrting PCD to mesh using OPEN3D ------------------------#
 pcd_i.estimate_normals()
 density = 0.08
 tengent_plane = 100
 depth_mesh = 9
 pcd_i.orient_normals_consistent_tangent_plane(tengent_plane)
 mesh_cleaned, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_i, depth=depth_mesh)
 vertices_to_remove = densities < np.quantile(densities, density)
 mesh_cleaned.remove_vertices_by_mask(vertices_to_remove)
 mesh_cleaned.compute_triangle_normals()
 o3d.io.write_triangle_mesh('tooth_poisson_'+str(i)+'.stl',mesh_cleaned)
 o3d.visualization.draw_geometries([mesh_cleaned],width=900)

