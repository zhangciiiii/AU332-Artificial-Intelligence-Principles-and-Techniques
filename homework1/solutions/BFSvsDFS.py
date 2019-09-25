# -*- coding: utf-8 -*-
from queue import LifoQueue
from queue import Queue
from queue import PriorityQueue

class Graph:
    """
    Defines a graph with edges, each edge is treated as dictionary
    look up. function neighbors pass in an id and returns a list of 
    neighboring node
    
    """
    def __init__(self):
        self.edges = {}
    
    def neighbors(self, id):
        # check if the edge is in the edge dictionary
        if id in self.edges:
            return self.edges[id]
        else:
            print("The node ", id , " is not in the graph")
            return False


def reconstruct_path(came_from, start, goal):
    """
    Given a dictionary of came_from where its key is the node 
    character and its value is the parent node, the start node
    and the goal node, compute the path from start to the end

    Arguments:
    came_from -- a dictionary indicating for each node as the key and 
                 value is its parent node
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    path. -- A list storing the path from start to goal. Please check 
             the order of the path should from the start node to the 
             goal node
    """
    path = []
    ### START CODE HERE ### (≈ 6 line of code)
    if came_from is None:
        print("Path reconstruction failed!!")
        return 

    current_node = goal
    while(current_node is not start):
        path.append(current_node)
        current_node = came_from[current_node]
    path.append(current_node)
    path.reverse()


    ### END CODE HERE ###
    return path

def breadth_first_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize breadth first search algorithm by finding the path from 
    start node to the goal node
    Use early stoping in your code
    This function returns back a dictionary storing the information of each node
    and its corresponding parent node
    Arguments:
    graph -- A dictionary storing the edge information from one node to a list 
             of other nodes
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    came_from -- a dictionary indicating for each node as the key and 
                value is its parent node
    """
    came_from = {}
    came_from[start] = None
    ### START CODE HERE ### (≈ 10 line of code)

    ## check goal and start ===========================
    if not ((goal in graph.edges) and (start in graph.edges) ):
        print("Error! Goal or Start is not in the graph")
        return None
    ##=================================================
    
    closed_set = [start]
    BFS_queue = Queue(maxsize=0)
    BFS_queue.put([None, start])
    while not BFS_queue.empty():
        Expand = BFS_queue.get()
        came_from[Expand[1]] = Expand[0]
        if goal in graph.neighbors(Expand[1]):
            came_from[goal] = Expand[1]
            return came_from
        for value in graph.neighbors(Expand[1]):
            if value not in closed_set:
                BFS_queue.put([Expand[1], value])
                closed_set.append(value)


    ### END CODE HERE ###
    return came_from




def depth_first_search(graph, start, goal):
    """
    Given a graph, a start node and a goal node
    Utilize depth first search algorithm by finding the path from 
    start node to the goal node
    Use early stoping in your code
    This function returns back a dictionary storing the information of each node
    and its corresponding parent node
    Arguments:
    graph -- A dictionary storing the edge information from one node to a list 
             of other nodes
    start -- A character indicating the start node
    goal --  A character indicating the goal node

    Return:
    came_from -- a dictionary indicating for each node as the key and 
                value is its parent node
    """
    came_from = {}
    came_from[start] = None
    ### START CODE HERE ### (≈ 10 line of code)

    ## check goal and start ===========================
    if not ((goal in graph.edges) and (start in graph.edges) ):
        print("Error! Goal or Start is not in the graph")
        return 
    ##=================================================

    closed_set = [start]
    DFS_queue = LifoQueue(maxsize=0)
    DFS_queue.put([None, start])
    while not DFS_queue.empty():
        Expand = DFS_queue.get()
        came_from[Expand[1]] = Expand[0]
        if goal in graph.neighbors(Expand[1]):
            came_from[goal] = Expand[1]
            return came_from
        for value in graph.neighbors(Expand[1]):
            if value not in closed_set:
                DFS_queue.put([Expand[1], value])
                closed_set.append(value)

    ### END CODE HERE ###
    
    return came_from



# The main function will first create the graph, then use depth first search
# and breadth first search which will return the came_from dictionary 
# then use the reconstruct path function to rebuild the path.
if __name__=="__main__":
    small_graph = Graph()
    small_graph.edges = {
        'A': ['B','D'],
        'B': ['A', 'C', 'D'],
        'C': ['A'],
        'D': ['E', 'A'],
        'E': ['B']
    }
    large_graph = Graph()
    large_graph.edges = {
        'S': ['A','C'],
        'A': ['S','B','D'],
        'B': ['S', 'A', 'D','H'],
        'C': ['S','L'],
        'D': ['A', 'B','F'],
        'E': ['G','K'],
        'F': ['H','D'],
        'G': ['H','E'],
        'H': ['B','F','G'],
        'I': ['L','J','K'],
        'J': ['L','I','K'],
        'K': ['I','J','E'],
        'L': ['C','I','J']
    }
    print("Large graph")
    start = 'S'
    goal = 'E'
    came_fromDFS = depth_first_search(large_graph, start, goal)
    print("came from DFS" , came_fromDFS)
    pathDFS = reconstruct_path(came_fromDFS, start, goal)
    print("path from DFS", pathDFS)
    came_fromBFS = breadth_first_search(large_graph, start, goal)
    print("came from BFS", came_fromBFS)
    pathBFS = reconstruct_path(came_fromBFS, start, goal)
    print("path from BFS", pathBFS)

    print("Small graph")
    start = 'A'
    goal = 'E'
    came_fromDFS = depth_first_search(small_graph, start, goal)
    print("came from DFS" , came_fromDFS)
    pathDFS = reconstruct_path(came_fromDFS, start, goal)
    print("path from DFS", pathDFS)
    came_fromBFS = breadth_first_search(small_graph, start, goal)
    print("came from BFS", came_fromBFS)
    pathBFS = reconstruct_path(came_fromBFS, start, goal)
    print("path from BFS", pathBFS)
