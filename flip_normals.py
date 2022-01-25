import pyvista as pv
import numpy as np


# <---------------------------------read VF modle and subsruc teeth from the bounding box ------------------------------------------------>
p = pv.Plotter()
teeth = pv.read('STLS/teeth_gum_filled.stl')
teeth.rotate_x(190,inplace=True)
box_cor = np.asarray(teeth.bounds)

hight=pv.read('hight.stl')
teeth.decimate(0.6)

box_cor[2] = box_cor[2]
box_cor[5] = box_cor[5]-5

box = pv.Box(bounds=box_cor, level=4)

box_points = np.asarray(box.points)
box.save('b-box.stl')
box = pv.read('b-box.stl')
# # plot.add_mesh(teeth)
# hight.rotate_x(180)
# hight.translate((0, 0,-2),inplace=True)
t = teeth.clip((0,0,-1),value=3.9)
t=teeth.extrude([0,0,25], capping=True)
t=t.clip((0,0,-1),value=7)
t=t.decimate(0.7)
t=t.smooth(100)
t.save('vf.stl')
p.add_mesh(teeth)
p.add_mesh(box)
box.translate((0, 0,-3),inplace=True)
# p.add_mesh(t)
p.show()
teeth_center = np.asarray(teeth.center)
#

result = teeth.boolean_difference(box)
p1=pv.Plotter()
p1.add_mesh(result)
result.save('hight.stl')
p1.show()
# teeth.extrude((0,0,10))
p1.show()