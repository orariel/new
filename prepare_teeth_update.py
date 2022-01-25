import pyvista as pv
import numpy as np


p = pv.Plotter()
teeth = pv.read('STLS/teeth_gum_filled.stl')
teeth=teeth.decimate(0.8)
teeth.rotate_x(190,inplace=True)
b_box_cor=np.asarray(teeth.bounds)
# teeth= teeth.clip((0,0,-1),value=3.9)
teeth=teeth.extrude([0,0,25], capping=True)
# teeth=teeth.clip((0,0,-1),value=7)
teeth=teeth.smooth(200)
teeth.save('vf2.stl')
p.add_mesh(teeth)
p.show()

#

