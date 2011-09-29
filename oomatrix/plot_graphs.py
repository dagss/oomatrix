import subprocess
import shlex

def plot_graph(graph, name='hey'):
    #verticelist = []
    #edges = {}
    #payloads = {}
    graphstr = get_graphstr(graph, name)
    ofile = open('temp.gv', 'w')
    ofile.write(graphstr)
    subprocess.call(shlex.split('dot temp.gv -Teps -o temp.eps'))
    subprocess.call(shlex.split('gv temp.eps'))

def get_graphstr(graph, name='hey'):
    rawgraph = ''
    for vertex in graph.get_vertices():
        edges = []
        payloads = []

        #vertexlist.append(str(vertex))
        #edges[vertex] = []
        #payloads[vertex] = []
        for edge, cost, payload in graph.get_edges(vertex):
            rawgraph += (str(vertex) + ' -> ' + str(edge) + ' [label="' + 
                        str(payload) + '"] \n')
            #edges.append(str(edge))
            #payloads.append(str(payload))
            #edges[vertex].append(str(edge))
            #payloads[vertex].append(str(payload))
    return 'digraph ' + name + ' { \n' + rawgraph + '}'

