import napari
from CTA.widget import CalciumControls

v = napari.Viewer()
v.window.add_dock_widget(CalciumControls(v), area='right', name='CTA')
napari.run()
