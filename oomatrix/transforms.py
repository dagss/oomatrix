
from . import metadata, symbolic


class MatrixMetadataLeaf(object):
    # Expression node for matrix metadata in a tree
    kind = universe = ncols = nrows = dtype = None # TODO remove these from symbolic tree
    
    def __init__(self, argument_index, metadata):
        self.argument_index = argument_index
        self.metadata = metadata

    def get_key(self):
        return self.metadata.kind

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_metadata_leaf(*args, **kw)

    def as_tuple(self):
        # Important: should sort by kind first
        return self.metadata.as_tuple() + (self.argument_index,)

    def _repr(self, indent):
        return [indent + '<arg:%d, %r>' % (self.argument_index, self.metadata)]



def recurse(transform, ctor, node):
    sorted_children = node.get_sorted_children()
    child_trees = [child.accept_visitor(transform, child)
                   for child in sorted_children]
    return ctor(child_trees)

class BaseTransform(object):
    def visit_multiply(self, node):
        return recurse(self, symbolic.add, node)

    def visit_add(self, node):
        return recurse(self, symbolic.multiply, node)
    

class MetadataTransform(BaseTransform):
    """
    Metadata transform -- split a tree with MatrixImpl leafs
    to a) a list of MatrixImpl instances ("arguments"), b) a
    tree of metadata and indices into the list of a).
    """

    def execute(self, node):
        # We build up the tree through the return values, while the argument
        # list is simply appended to as we proceed
        assert not hasattr(self, 'child_args') # self is throw-away
        self.child_args = []
        child_tree = node.accept_visitor(self, node)
        return child_tree, self.child_args

    def visit_add(self, node):
        return recurse(self, symbolic.add, node)

    def visit_multiply(self, node):
        return recurse(self, symbolic.multiply, node)

    def visit_conjugate_transpose(self, node):
        result = node.child.accept_visitor(self, node.child)
        return symbolic.conjugate_transpose(result)

    def visit_inverse(self, node):
        result = node.child.accept_visitor(self, node.child)
        return symbolic.inverse(result)

    def visit_leaf(self, node):
        i = len(self.child_args)
        self.child_args.append(node.matrix_impl)
        meta = metadata.MatrixMetadata(node.kind, (node.nrows,), (node.ncols,),
                                       node.dtype)
        return MatrixMetadataLeaf(i, meta)

def metadata_transform(tree):
    return MetadataTransform().execute(tree)


class KindKeyTransform(object):
    """
    Turns a tree with metadata leafs into a key-tuple (the as_tuple
    representation of a tree with Kind as leafs).

    Since a Metadata tree is already sorted by kind first, we can simply
    serialize the Metadata tree.
    """

    def execute(self, node):
        return node.accept_visitor(self, node)

    def recurse(self, node):
        sorted_children = node.get_sorted_children()
        return (node.symbol,) + tuple([child.accept_visitor(self, child)
                                       for child in sorted_children])

    visit_add = visit_multiply = visit_conjugate_transpose = recurse
    visit_inverse = recurse

    def visit_metadata_leaf(self, node):
        return node.metadata.kind


def kind_key_transform(tree):
    return KindKeyTransform().execute(tree)
     
