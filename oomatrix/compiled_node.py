from __future__ import division

import hashlib
import struct
from . import utils

class CompiledNode(object):
    """
    Objects of this class make up the final "program tree".

    It is immutable and compares by value.
    
    `children` are other CompiledNode instances whose output should
    be fed into `computation`.

    For comparison and hashing, we use a sha512 and trust that it is
    going to be unique.
    """
    def __init__(self, computation, weighted_cost, children, metadata, shuffle=None,
                 flat_shuffle=None):
        if shuffle is not None and flat_shuffle is not None:
            raise ValueError('either shuffle or flat_shuffle must be given')
        self.computation = computation
        self.weighted_cost = float(weighted_cost)
        self.children = tuple(children)
        self.metadata = metadata
        self.is_leaf = computation is None
        if self.is_leaf:
            if shuffle not in (None, ()):
                raise ValueError('invalid shuffle for leaf node')
            self.shuffle = ()
            self.arg_count = 1
        else:
            if shuffle is None:
                if flat_shuffle is None:
                    flat_shuffle = range(sum(child.arg_count for child in children))
                elif not all(isinstance(i, int) for i in flat_shuffle):
                    raise ValueError('invalid flat_shuffle')
                shuffle = []
                i = 0
                for child in self.children:
                    n = child.arg_count
                    shuf_part = tuple(flat_shuffle[i:i + n])
                    if len(shuf_part) != n:
                        raise ValueError('Invalid flat_shuffle %r, trailing part %r does not match child arg_count %d' %
                                         (flat_shuffle, shuf_part, n))
                    shuffle.append(shuf_part)
                    i += n
                if i != len(flat_shuffle):
                    raise ValueError('Invalid flat_shuffle %r, not all indices consumed' % flat_shuffle)
                self.shuffle = tuple(shuffle)
            else:
                self.shuffle = tuple(shuffle)
                if len(self.shuffle) != len(self.children):
                    raise ValueError('len(shuffle) != len(children)')
                for child, indices in zip(self.children, self.shuffle):
                    if child.arg_count != len(indices):
                        raise ValueError('shuffle part %r does not match child %r' % (indices,
                                                                                      child))
            flat_shuffle = sum(self.shuffle, ())
            self.arg_count = max(flat_shuffle) + 1
        # total_cost is computed
        self.total_cost = self.weighted_cost + sum(child.total_cost for child in self.children)
        self._make_hash()

    def secure_hash(self):
        return self._shash

    def _make_hash(self):
        h = hashlib.sha512()
        h.update(struct.pack('Qd', id(self.computation), self.weighted_cost))
        h.update(str(self.shuffle))
        h.update(self.metadata.secure_hash())
        # Finally add hashes of children
        h.update(struct.pack('Q', len(self.children)))
        for child in self.children:
           h.update(child._shash)
        self._shash = h.digest()

    def __eq__(self, other):
        # Note that this definition is recursive, as the comparison of children will
        # end up doing an element-wise comparison
        if type(other) is not CompiledNode:
            return False
        else:
            return self._shash == other._shash

    def __hash__(self):
        return hash(self._shash)

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def create_leaf(metadata):
        return CompiledNode(None, 0, (), metadata, ())

    def __repr__(self):
        lines = []
        self._repr('', lines)
        return '\n'.join(lines)

    def _repr(self, indent, lines):
        if self.is_leaf:
            lines.append(indent + '<leaf:%s cost=%.1f>' % (
                self.metadata.kind.name, self.weighted_cost))
        else:
            lines.append(indent + '<node:%s:%s cost=%.1f arg_count=%d shuffle=%s;' % (
                self.metadata.kind.name,
                self.computation.name,
                self.total_cost,
                self.arg_count,
                self.shuffle))
            for child in self.children:
                child._repr(indent + '  ', lines)
            lines.append(indent + '>')

    def leaves(self):
        if self.is_leaf:
            return [self]
        else:
            return sum([child.leaves() for child in self.children], [])

    def substitute(self, substitutions, shuffle=None, flat_shuffle=None, node_factory=None):
        """Substitute each leaf node of the tree rooted at `self` with the
        CompiledNodes given in substitutions, and return the resulting tree.

        substitutions can either be a list with `self.arg_count` elements (None meaning
        "do not replace"), or a dict mapping argument indices to replacement leaves.

        Optionally also converts the interior nodes in the tree by using a supplied
        node factory for the interior nodes.
        """
        if isinstance(substitutions, dict):
            substitutions, substitutions_dict = [None] * self.arg_count, substitutions
            for i, subst in substitutions_dict.iteritems():
                substitutions[i] = subst

        # For every inserted substitution we may need to insert its arguments
        # into our argument list
        arg_remapping = [None] * self.arg_count
        shift = 0
        for i, subst in enumerate(substitutions):
            if subst is not None:
                arg_remapping[i] = tuple(range(i + shift, i + shift + subst.arg_count))
                shift += subst.arg_count - 1
            else:
                arg_remapping[i] = (i + shift,)
        
        if node_factory is None:
            def node_factory(node, new_shuffle, converted_children):
                use_shuffle_arg = (node is self) and not (shuffle is flat_shuffle is None)
                shuffle_ = shuffle if use_shuffle_arg else new_shuffle
                flat_shuffle_ = flat_shuffle if use_shuffle_arg else None
                return CompiledNode(node.computation, node.weighted_cost,
                                    converted_children, node.metadata,
                                    shuffle_, flat_shuffle_)

        return self._substitute(substitutions, arg_remapping, node_factory)

    def _substitute(self, substitutions, arg_remapping, node_factory):
        if self.is_leaf:
            r = substitutions[0] or self
            assert r.metadata == self.metadata
            return r
        else:
            # Shuffle arguments and recurse
            new_children = []
            for child, child_shuffle in zip(self.children, self.shuffle):
                new_substitutions = [substitutions[i] for i in child_shuffle]
                new_child = child._substitute(new_substitutions, arg_remapping, node_factory)
                new_children.append(new_child)
            # Make new shuffle based on arg_remapping
            new_shuffle = tuple(sum((arg_remapping[i] for i in shuf_part), ())
                                for shuf_part in self.shuffle)
            new_node = node_factory(self, new_shuffle, new_children)
            return new_node
            
    def substitute_linked(self, substitute_at, substitution_cnode,
                          substitution_indices, nonsubstitution_indices=None):
        substitution_indices = tuple(substitution_indices)
        assert len(substitution_indices) == substitution_cnode.arg_count
        new_arg_count = self.arg_count - len(substitute_at) + substitution_cnode.arg_count
        if nonsubstitution_indices is None:
            nonsubstitution_indices = utils.complement_range(substitution_indices, new_arg_count)
        index_remapping = [None] * self.arg_count
        j = 0
        for i in range(self.arg_count):
            if i in substitute_at:
                index_remapping[i] = substitution_indices
            else:
                index_remapping[i] = (nonsubstitution_indices[j],)
                j += 1
                
        return self._substitute_linked(substitute_at, substitution_cnode, index_remapping)

    def _substitute_linked(self, indices, substitute_at, substitute_cnode, index_remapping):
        if self.is_leaf:
            index, = indices
            if index in substitute_at:
                return substitute_cnode
            else:
                return self
        else:
            new_children = []
            new_shuffle = []
            for child, shuf_part in zip(self.children, self.shuffle):
                new_indices = [indices[i] for i in shuf_part]
                new_child = child._substitute_linked(new_indices, substitute_at, substitute_cnode,
                                                     substitution_indices, index_remapping)
                new_children.append(new_child)

                new_shuf_part = tuple(sum((index_remapping[i] for i in new_indices), ()))
                new_shuffle.append(new_shuf_part)
                
            new_node = CompiledNode(self.computation, self.weighted_cost, new_children,
                                    self.metadata, shuffle=new_shuffle)
            return new_node

