Install the stuff in requirements.txt and run main.py with Python 3. Program takes graphs in json format specifically made with Networkx's json_graph class, go poke at the preprocessors in the data folder I guess. Good luck.

To enter new data, go to main.py and change the filepath variable.

Current controls (subject to change):  
-> Mouse wheel: zoom  
-> Right click and drag: move camera (flying)
-> Mouse wheel click and drag: move camera (orbital)  

Known issues to be fixed:  
-> You can zoom through the object, in which case zooming out will make it appear to flip (it was the camera that flipped)
-> The orbital camera is flipping the object if looked at from directly above or directly below (phi = 0° or phi = 180°)