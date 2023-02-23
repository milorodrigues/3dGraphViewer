Proof of concept graph visualization tool, written in Python3+OpenGL in 2022 as my undergrad thesis project. The paper (Analyzing Three-Dimensional Graph-Drawing Heuristics on Protein Interaction Data and General Graphs) is included.

## Instructions

* Install the dependencies in requirements.txt and run main.py with Python 3
* Graph data must have been created by Networkx's json_graph method (there's examples in /data)

## Command line usage (supports -h):

`python3 main.py -f <filepath> [-i <iterations>] [-m <model>] [-t <target>]`

* filepath: relative path to the .json file containing the graph
* iterations: number of iterations the drawing algorithm will perform (default 100)
* model: drawing algorithm to use (currently: barycentric, spring, random) (default spring)
* target: whether to render an extra point in the screen showing the specific coordinate the camera is looking at (Y/N) (default N)

## Current controls (subject to change):

* Mouse wheel: move camera forward/backward  
* Right click and drag: move camera (flying)  
* Mouse wheel click and drag: move camera (orbital)