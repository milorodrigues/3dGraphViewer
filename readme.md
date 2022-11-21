Install the stuff in requirements.txt and run main.py with Python 3. Program takes graphs in json format specifically made with Networkx's json_graph class, go poke at the preprocessors in the data folder I guess. Good luck.

## Command line usage (supports -h):

`python main.py -f <filepath> [-i <iterations>] [-m <model>] [-t <target>]`

* filepath: relative path to the .json file containing the graph
* iterations: number of iterations the drawing algorithm will perform
* model: drawing algorithm to use (currently: barycentric, eades, random)
* target: whether to render an extra point in the screen showing the specific coordinate the camera is looking at

## Current controls (subject to change):  
* Mouse wheel: move camera forward/backward  
* Right click and drag: move camera (flying)  
* Mouse wheel click and drag: move camera (orbital)