import subprocess
import os
from tempfile import mkstemp
import threading

def spawn_gv_with_temporary(postscript_file, block=False):
    # Spawns gv (execute + wait for it in a background thread)
    # and deletes file after gv has terminated
    def thread():
        retcode = subprocess.call(['gv', postscript_file])
        if retcode != 0:
            raise Exception("Command 'gv' failed with return code: %s" % retcode)
        os.unlink(postscript_file)
    if block:
        thread()
    else:
        threading.Thread(target=thread).start()

def plot_graph(graph, block=False, **options):
    graphstr = get_graphstr(graph, **options)
    fd, temp_filename = mkstemp()
    os.close(fd)
    # Open a dot process to read from standard input and give it our graph
    process = subprocess.Popen(['dot', '-Teps', '-o', temp_filename],
                               stdin=subprocess.PIPE)
    process.stdin.write(graphstr)
    process.stdin.close()
    retcode = process.wait()
    if retcode != 0:
        raise Exception("Command 'dot' failed with return code: %s" % retcode)
    spawn_gv_with_temporary(temp_filename, block=block)

def _format_label(v):
    return dict(label=str(v))

def _format_edge(cost, payload):
    return "%.2f" % cost

def get_graphstr(graph, max_node_size=4, name='', format_vertex=str,
                 format_edge=_format_edge):
    rawgraph = ''
    vertex_names = {}
    def nameof(v):
        name = vertex_names.get(v, None)
        if name is None:
            name = 'vertex%d' % len(vertex_names)
            vertex_names[v] = name
        return name

    def attrstr(attrdict):
        if isinstance(attrdict, str):
            attrdict = dict(label=attrdict)
        return ','.join('%s="%s"' % (key, value)
                        for key, value in attrdict.iteritems())    
    for vertex in graph.get_vertices(max_node_size):
        edges = []
        payloads = []
        attrs = attrstr(format_vertex(vertex))
        rawgraph += '%s [%s];\n' % (nameof(vertex), attrs)
        for target_vertex, cost, payload in graph.get_edges(vertex):
            attrs = attrstr(format_edge(cost, payload))
            rawgraph += '%s -> %s [%s]\n' % (
                nameof(vertex), nameof(target_vertex), attrs)
    return 'digraph %s { \n %s }\n' % (name, rawgraph)

