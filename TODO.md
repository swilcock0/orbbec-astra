- [X] Fix bundle regression based on https://www.research-collection.ethz.ch/handle/20.500.11850/248154
- [X] Fix the bundle and tag definitions (see https://wiki.ros.org/apriltag_ros)
- [X] Now, test inside Grasshopper
- [X] Upgrade to Python3.9/Open3d 0.18 to match Rhino
- [ ] Then, find the old Grasshopper definition for the panels. Extract the panel meshes and the relative transformations
- [ ] Set it up to align the sheet of paper bundle with the world frame
- [ ] Align lowest numerical panel to its intended location
- [ ] Get measurements on the other panels (using that thingy method) to detect the difference between the intended and placed
- [ ] Add additional panel?
- [ ] PCL data? As an afterthought perhaps
- [ ] Break up the script into separate concerns (utilities, config, pointcloud, apriltags?)


NOTES
---
So the processing/production of points, pointclouds etc. in Rhino itself is VERY slow compared to o3d etc. We'll be better off avoiding non-native visualisations etc. where possible and keep it within Python.