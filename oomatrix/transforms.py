from . import metadata, symbolic, utils

class MatrixMetadataLeaf(symbolic.ExpressionNode):
    # Expression node for matrix metadata in a tree
    kind = universe = ncols = nrows = dtype = None # TODO remove these from symbolic tree
    
    def __init__(self, leaf_index, metadata):
        self.leaf_index = leaf_index
        self.metadata = metadata

    def accept_visitor(self, visitor, *args, **kw):
        return visitor.visit_metadata_leaf(*args, **kw)

    def as_tuple(self):
        # Important: should sort by kind first
        return self.metadata.as_tuple() + (self.leaf_index,)

    def _repr(self, indent):
        return [indent + '<arg:%s, %r>' % (self.leaf_index, self.metadata)]


    
class ImplToMetadataTransform(object):
    """
    Split a tree with MatrixImpl leafs
    to a) a list of MatrixImpl instances ("arguments"), b) a
    tree of metadata with leaves corresponding sequentially to
    the list of a).
    """

    def execute(self, node):
        child_tree, child_args = node.accept_visitor(self, node)
        return child_tree, child_args

    def process_children(self, node):
        child_trees = []
        child_arg_lists = []
        for child in node.children:
            tree, args = child.accept_visitor(self, child)
            child_trees.append(tree)
            child_arg_lists.append(args)
        return child_trees, child_arg_lists        

    def visit_add(self, node):
        child_trees, child_arg_lists = self.process_children(node)
        # Sort result by the metadata before returning
        permutation = utils.argsort(child_trees)
        child_trees = [child_trees[i] for i in permutation]
        child_arg_lists = [child_arg_lists[i] for i in permutation]
        return symbolic.add(child_trees), sum(child_arg_lists, [])

    def visit_multiply(self, node):
        child_trees, child_arg_lists = self.process_children(node)
        return symbolic.multiply(child_trees), sum(child_arg_lists, [])

    def visit_conjugate_transpose(self, node):
        tree, args = node.child.accept_visitor(self, node.child)
        return symbolic.conjugate_transpose(tree), args

    def visit_inverse(self, node):
        tree, args = node.child.accept_visitor(self, node.child)
        return symbolic.inverse(tree), args

    def visit_leaf(self, node):
        meta = metadata.MatrixMetadata(node.kind, (node.nrows,), (node.ncols,),
                                       node.dtype)
        return MatrixMetadataLeaf(None, meta), [node.matrix_impl]

class IndexMetadataTransform(object):
    """
    Take a MatrixMetadataLeaf tree and annotates each leaf
    with a global leaf index
    """
    def execute(self, node):
        self.leaf_index = 0
        return node.accept_visitor(self, node)

    def recurse_multi_child(self, node):
        return type(node)([child.accept_visitor(self, child)
                           for child in node.children])

    def recurse_single_child(self, node):
        child = node.child
        return type(node)(child.accept_visitor(self, child))

    visit_add = visit_multiply = recurse_multi_child
    visit_conjugate_transpose = visit_inverse = recurse_single_child

    def visit_metadata_leaf(self, node):
        node.leaf_index = self.leaf_index
        self.leaf_index += 1
        return node

def metadata_transform(tree):
    pre_tree, args_list = ImplToMetadataTransform().execute(tree)
    result_tree = IndexMetadataTransform().execute(pre_tree)
    return result_tree, args_list

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
        return (node.symbol,) + tuple([child.accept_visitor(self, child)
                                       for child in node.children])

    visit_add = visit_multiply = visit_conjugate_transpose = recurse
    visit_inverse = recurse

    def visit_metadata_leaf(self, node):
        return node.metadata.kind


def kind_key_transform(tree):
    return KindKeyTransform().execute(tree)
     
