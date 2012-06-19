from . import metadata, symbolic, utils, task


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
        new_node = symbolic.add(child_trees)
        return new_node, sum(child_arg_lists, [])

    def visit_multiply(self, node):
        child_trees, child_arg_lists = self.process_children(node)
        new_node = symbolic.multiply(child_trees)
        return new_node, sum(child_arg_lists, [])

    def visit_conjugate_transpose(self, node):
        tree, args = node.child.accept_visitor(self, node.child)
        new_node = symbolic.conjugate_transpose(tree)
        return new_node, args

    def visit_inverse(self, node):
        tree, args = node.child.accept_visitor(self, node.child)
        new_node = symbolic.inverse(tree)
        return new_node, args

    def visit_decomposition(self, node):
        tree, args = node.child.accept_visitor(self, node.child)
        new_node = symbolic.DecompositionNode(tree, node.decomposition)
        return new_node, args

    def visit_leaf(self, node):
        meta = metadata.MatrixMetadata(node.kind, (node.nrows,), (node.ncols,),
                                       node.dtype)
        new_node = symbolic.MatrixMetadataLeaf(meta)
        new_node.set_leaf_index(None)
        return new_node, [node]

class IndexMetadataTransform(object):
    """
    Take a MatrixMetadataLeaf tree and annotates each leaf
    with a global leaf index in-place
    """
    def execute(self, node):
        self.leaf_index = 0
        node.accept_visitor(self, node)
        return node

    def recurse_multi_child(self, node):
        for child in node.children:
            child.accept_visitor(self, child)

    def recurse_single_child(self, node):
        child, = node.children
        child.accept_visitor(self, child)

    visit_add = visit_multiply = recurse_multi_child
    visit_conjugate_transpose = visit_inverse = recurse_single_child
    visit_decomposition = recurse_single_child

    def visit_metadata_leaf(self, node):
        node.set_leaf_index(self.leaf_index)
        self.leaf_index += 1

def metadata_transform(tree):
    tree, args_list = ImplToMetadataTransform().execute(tree)
    IndexMetadataTransform().execute(tree)
    return tree, args_list

class KindKeyTransform(object):
    """
    Turns a tree with metadata leafs into a key-tuple (the as_tuple
    representation of a tree with Kind as leafs). Also returns a
    MatrixKindUniverse containing at least one of the kinds.

    Since a Metadata tree is already sorted by kind first, we can simply
    serialize the Metadata tree.
    """

    def execute(self, node):
        self.universe = None
        return node.accept_visitor(self, node), self.universe

    def recurse(self, node):
        return (node.symbol,) + tuple([child.accept_visitor(self, child)
                                       for child in node.children])

    visit_add = visit_multiply = visit_conjugate_transpose = recurse
    visit_inverse = visit_decomposition = recurse

    def visit_metadata_leaf(self, node):
        self.universe = node.metadata.kind.universe
        return node.metadata.kind

    def visit_task_leaf(self, node):
        self.universe = node.metadata.kind.universe
        return node.metadata.kind

def kind_key_transform(tree):
    return KindKeyTransform().execute(tree)
     

class FlattenTransform(object):
    """
    Turns a tree into (root_metadata, args); used for computing costs
    and constructing Tasks.
    """
    def execute(self, node):
        self.flattened = []
        return node.accept_visitor(self, node), self.flattened

    def recurse(self, children):
        return [child.accept_visitor(self, child) for child in children]

    def visit_add(self, node):
        return metadata.meta_add(self.recurse(node.children))    

    def visit_multiply(self, node):
        return metadata.meta_multiply(self.recurse(node.children))

    def visit_leaf(self, node):
        self.flattened.append(node)
        return node.metadata

    visit_task_leaf = visit_metadata_leaf = visit_leaf

    def visit_single(self, node):
        return node.child.accept_visitor(self, node.child) 

    visit_conjugate_transpose = visit_inverse = visit_single
    visit_decomposition = visit_single


def flatten(node):
    return FlattenTransform().execute(node)
