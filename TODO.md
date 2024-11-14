- [X] Fix bundle regression based on https://www.research-collection.ethz.ch/handle/20.500.11850/248154
- [X] Fix the bundle and tag definitions (see https://wiki.ros.org/apriltag_ros)
- [X] Now, test inside Grasshopper
- [X] Upgrade to Python3.9/Open3d 0.18 to match Rhino
- [X] Then, find the old Grasshopper definition for the panels. Extract the panel meshes and the relative transformations
- [X] Figure out why point clouds aren't aligned with the 
- [X] Set it up to align the sheet of paper bundle with the world frame
- [ ] Align lowest numerical panel to its intended location
- [ ] Get measurements on the other panels (using that thingy method - ADD 6d?) to detect the difference between the intended and placed
- [ ] Add additional panel?
- [ ] PCL data? As an afterthought perhaps
- [ ] Break up the script into separate concerns (utilities, config, pointcloud, apriltags?)
- [ ] New data for 2 panels, 3 

NOTES
---realtek rtl8821ce
So the processing/production of points, pointclouds etc. in Rhino itself is slooooow compared to o3d etc. We'll be better off avoiding non-native visualisations etc. where possible and keep it within Python.