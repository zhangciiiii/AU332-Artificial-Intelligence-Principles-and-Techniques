from queue import PriorityQueue
import math

class Graph:
    """
    Defines a graph with edges, each edge is treated as dictionary
    look up. function neighbors pass in an id and returns a list of 
    neighboring node
    
    """
    def __init__(self):
        self.locations = {}

def heuristic(graph, node, target = [5,2]):
    distance = abs(graph.locations['({},{})'.format(node[0],node[1])][0]-target[0]) + abs(graph.locations['({},{})'.format(node[0],node[1])][1]-target[1])
    return distance

def reconstruct_path(came_from, start, goal):

    path = []
    current_node = goal
    while(came_from[current_node] is not None):
        path.append(current_node)
        # print(current_node)
        if came_from[current_node] is  None :
            break
        current_node = '({},{})'.format( came_from[current_node][0],came_from[current_node][1])
    path.append(current_node)
    path.reverse()
    return path

def AstarSearch(graph, start = [1,2], target = [5,2]):
    came_from = {} # key is child and value is father
    cost_so_far = {}
    
    came_from['({},{})'.format(start[0],start[1])] = None
    cost_so_far['({},{})'.format(start[0],start[1])] = 0

    closed_set = [start]
    Astar_queue = PriorityQueue(maxsize=0)
    Astar_queue.put((0 + heuristic(graph,start), [None, start, 0]))

    step = 0

    while not Astar_queue.empty():
        Expand = Astar_queue.get()
        if Expand[1][1] in closed_set and step is not 0:  
            continue

        step+=1
        print('\item '+"step "+ str(step))
        print(Expand[1][1])
        print('\n')

        closed_set.append(Expand[1][1])
        came_from['({},{})'.format(Expand[1][1][0],Expand[1][1][1])] = Expand[1][0]
        cost_so_far['({},{})'.format(Expand[1][1][0],Expand[1][1][1])] = Expand[1][2]

        if (Expand[1][1][0] is target[0])  and (Expand[1][1][1] is target[1]) :
            return came_from,cost_so_far
        
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if abs(i+j) is not 1 :
                    continue
                next_node = [Expand[1][1][0]+i, Expand[1][1][1]+j]
                if (next_node not in closed_set) and ( '({},{})'.format(next_node[0],next_node[1]) in graph.locations):
                    Astar_queue.put((1+Expand[1][2]+heuristic(graph,next_node), [Expand[1][1], next_node, 1+Expand[1][2]]))
                    
        print('fringe list')
        print(Astar_queue.queue)
        print('\n')
        print('closed set')
        print(closed_set)
        print('\n')

    return came_from,cost_so_far
    

if __name__=="__main__":
    graph = Graph()
    for i in range (0,7):
        for j in range (0,5):
            graph.locations['({},{})'.format(i,j)] = [i, j]
    
    del graph.locations['(3,1)']
    del graph.locations['(3,2)']
    del graph.locations['(3,3)']

    came_from, cost_so_far =  AstarSearch(graph)

    path = reconstruct_path(came_from,"(1,2)","(5,2)")

    print('final path ')
    print(path)

