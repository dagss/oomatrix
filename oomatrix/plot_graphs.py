import subprocess
import shlex

def plot_graph(graph, name='hey'):
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
        rawgraph += '"' + str(vertex) + '"\n'
        for edge, cost, payload in graph.get_edges(vertex):
            rawgraph += ('"' + str(vertex) + '" -> "' + str(edge) + '" [label="' + 
                        str(payload) + '"] \n')
    return 'digraph ' + name + ' { \n' + rawgraph + '}\n'

