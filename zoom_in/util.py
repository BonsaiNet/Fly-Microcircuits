"Utility to analysis the microcircutes of fruit fly"
import numpy as np
import pandas as pd
from sklearn import preprocessing
from copy import deepcopy

def get_nodes_in_box(skeleton_coordinates,
                     centroid = [28000, 26000, 53000],
                     radius = 3000):
    """Returning the part of skeleton in the a given box.
    Parameters:
    -----------
    skeleton_coordinates: pandas
        Dataframe of all nodes. It should have these keys: ['skeleton_id',
        ' treenode_id', ' parent_treenode_id', ' x', ' y', ' z', ' r']

    centroid: list
        the centroid of the box. 3 values in the centroid are the coordinates
        of the centroid.
        Range of x is: max_x = 105207, min_x = 3197
        Range of y is: max_y = 117569, min_y = 2636
        Range of z is: max_z = 239650, min_z = 8550

    radius: flaot
        the radius around the centroid.

    Returns:
    --------
    box_skeleton: panadas
        A Dataframe with the structure of skeleton_coordinates for all the node in
        a cubic box with the center of 'centorid' and L1 distance of 'radius.

    """
    x = np.abs(skeleton_coordinates[' x'] - centroid[0])<radius
    y = np.abs(skeleton_coordinates[' y'] - centroid[1])<radius
    z = np.abs(skeleton_coordinates[' z'] - centroid[2])<radius
    index = np.where(np.logical_and(np.logical_and(x,y),z))[0]
    box_skeleton = skeleton_coordinates.iloc[index,:]
    box_skeleton = box_skeleton.reset_index()
    return box_skeleton

def connected_components_in_skeletons(skeleton_coordinates):
    """Returning the part of skeleton in the a given box.
    Parameters:
    -----------
    skeleton_coordinates: pandas
        Dataframe of all nodes. It should have these keys: ['skeleton_id',
        ' treenode_id', ' parent_treenode_id', ' x', ' y', ' z', ' r']


    Returns:
    --------
    skeleton_coordinates_with_cc: panadas
        a Dataframe similar to the skeleton_coordinates, but with the connected components.
        Index of connected components is under the title 'cc_skeleton_id'.

    """
    skeleton_coordinates_with_cc = deepcopy(skeleton_coordinates)
    index_id = np.array(skeleton_coordinates[' treenode_id']).astype(int)
    parent_id = np.array(skeleton_coordinates[' parent_treenode_id']).astype(int)

    # all the somas in the box
    soma_id = np.isnan(parent_id)

    # defining a big number to not messing with other indcies when applying labelencoder.
    # Notice that Labelencoder sort the indcies before preforming.
    big_number = parent_id[~soma_id].max() + index_id.max()

    # changing parent_id of any somas in the skeleton to big numbers
    parent_id[soma_id] = np.arange(big_number, big_number+sum(soma_id))

    # refreshing
    index_id, parent_id = refresh_id(index_id, parent_id)

    # pushing the indecis of all the borders to high values
    border_id = np.array(list(set(parent_id) - set(index_id)))
    border_index = np.array([])
    for i in border_id:
        border_index = np.append(border_index, np.where(parent_id == i)[0])
    parent_id[border_index.astype(int)] = 2*big_number + np.arange(len(border_index))

    index_id, parent_id = refresh_id(index_id, parent_id)

    # extending to the all id
    ext_index_id = np.append(index_id, np.arange(index_id.max()+1, parent_id.max()+1))
    ext_parent_id = np.append(parent_id, np.arange(index_id.max()+1, parent_id.max()+1))
    inverse_ext_index_id = inverse_array(ext_index_id)
    inverse_ext_index_id = inverse_ext_index_id.astype(int)

    # finding the skeleton id
    cc_skeleton_id = ext_parent_id
    while cc_skeleton_id.min()< index_id.max():
        cc_skeleton_id = ext_parent_id[inverse_ext_index_id[cc_skeleton_id]]
    cc_skeleton_id = cc_skeleton_id[:index_id.max()+1]

    le = preprocessing.LabelEncoder()
    le.fit(cc_skeleton_id)
    cc_skeleton_id = le.transform(cc_skeleton_id)
    skeleton_coordinates_with_cc['cc_skeleton_id'] = cc_skeleton_id
    return skeleton_coordinates_with_cc

def get_swc(skeleton_coordinates):
    """Returing the swc format of the neuron (neurite or any cutie to call) from skeleton
    Parameters:
    -----------
    skeleton_coordinates: pandas
        Dataframe of all nodes. It should have these keys: ['skeleton_id',
        ' treenode_id', ' parent_treenode_id', ' x', ' y', ' z', ' r']

    Returns:
    --------
    swc: numpy
        A numpy array of shape [n, 7]

    """
    n_node = skeleton_coordinates.shape[0]
    swc = np.zeros([n_node,7])
    swc[1:,0] = np.arange(1,n_node)
    swc[:, 2] = np.array(skeleton_coordinates[' x'])
    swc[:, 3] = np.array(skeleton_coordinates[' y'])
    swc[:, 4] = np.array(skeleton_coordinates[' z'])
    swc[:, 5] = np.array(skeleton_coordinates[' r'])
    swc[:, 6] = skeleton_coordinates[' new_parent_treenode_id']+1
    swc[0, 1] = 1
    swc[0, 6] = -1
    return swc

def get_one_neurite_in_box(box_skeleton, index):
    """Returning the swc of one neurite (with cc==index) from the skeleton.
    Parameters:
    -----------
    box_skeleton: pandas
        Dataframe of all nodes. It should have these keys: ['skeleton_id',
        'cc_skeleton_id', ' treenode_id', ' parent_treenode_id', ' x', ' y', ' z', ' r']

    index: int
        Index of connected component

    Returns:
    --------
    swc: numpy

    part_neuron: panadas
        A Dataframe with the structure of box_skeleton for the extracted neurites.
        It has two extra keys: ' new_treenode_id' and ' new_parent_treenode_id'

    """
    part_neuron_index = np.where(box_skeleton['cc_skeleton_id']==index)[0]
    part_neuron = box_skeleton.iloc[part_neuron_index,:]

    index_id = np.array(part_neuron[' treenode_id']).astype(int)
    parent_id = np.array(part_neuron[' parent_treenode_id']).astype(int)
    index_id, parent_id = refresh_id(index_id, parent_id)

    m = parent_id.max() +  index_id.max()
    border_id = np.array(list(set(parent_id) - set(index_id)))
    for i in border_id:
        parent_id[parent_id == i] = 2*m+i

    index_id, parent_id = refresh_id(index_id, parent_id)
    part_neuron[' new_treenode_id'] = index_id
    part_neuron[' new_parent_treenode_id'] = parent_id
    n_node = index_id.shape[0]

    # getting the depth
    depth = get_depth(index_id, parent_id)
    part_neuron[' depth'] = depth

    # Getting the inverse transform based on the depth
    permutation = np.zeros(n_node)
    start = 0
    for i in range(depth.max()+1):
        level_set = depth==i
        node_in = sum(level_set)
        permutation[level_set] = range(start, start+ node_in)
        start += node_in
    permutation = permutation.astype(int)
    part_neuron = part_neuron.iloc[inverse_array(permutation)]

    # transform the node of neuron_info
    invers_node = inverse_array(np.array(part_neuron[' new_treenode_id']))
    invers_node = np.append(invers_node, n_node)
    parent_id = np.array(part_neuron[' new_parent_treenode_id']).astype(int)
    part_neuron[' new_parent_treenode_id'] = invers_node[parent_id]
    part_neuron[' new_treenode_id'] = np.arange(n_node)
    swc = get_swc(part_neuron)
    part_neuron = part_neuron.reset_index()
    return swc, part_neuron

def get_all_neurites_in_box(box_skeleton):
    """Returning the swc of all connected neurites in the skeleton.
    Parameters:
    -----------
    box_skeleton: pandas
        Dataframe of all nodes. It should have these keys: ['skeleton_id',
        'cc_skeleton_id', ' treenode_id', ' parent_treenode_id', ' x', ' y', ' z', ' r']

    Returns:
    --------
    list_swc: list
        the list of numpy array of shape [7, n]. The ordering is random.
    """
    list_swc = []
    for i in range(box_skeleton['cc_skeleton_id'].max()):
        swc, _ = get_one_neurite_in_box(box_skeleton, index=i)
        list_swc.append(swc)
    return list_swc

def are_same(node1, node2, tree):
    """Checking that the subtree after one node is graphically isometric to the subtree after other node.
    it
    Parameters:
    -----------
    node1& node2: int
        Index of two nodes on tree.

    tree: numpy
        the parent index of the tree.

    Returns:
    --------
    boolean
        return True if the subtree after node1 is isomorphic with the subtree after node2.
    """
    (child1,) = np.where(tree==node1)
    (child2,) = np.where(tree==node2)
    if len(child1) == 0 and len(child2) == 0:
        return 1
    elif len(child1) != len(child2):
        return 0
    else:
        return (are_same(child1[0], child2[0], tree) and are_same(child1[1], child2[1], tree))\
                or \
               (are_same(child1[0], child2[1], tree) and are_same(child1[1], child2[0], tree))

def are_two_trees_same(tree1, tree2):
    """Checking if two tree are isomorphic.
    it
    Parameters:
    -----------
    tree1& tree2: numpy array
        the parentindex of two tree. The root should be the first index (0).

    Returns:
    --------
    boolean
        return True if two trees are isomorphic.
    """
    big_tree = np.append(0,np.append(np.append(0, tree1[1:]+1),np.append(0, tree2[1:]+len(tree1)+1)))
    return are_same(1, len(tree1)+1, big_tree)

def find_all_good_centroids(skeleton_coordinates,
                            max_val=[110000, 120000, 250000],
                            min_val=[0,0,0],
                            radius=5000,
                            min_points_in_box = 40000,
                            consecuative_dis=5000):
    """Returning all the centroid in the space that a box around them (with given radius)
    has at least a givennumber of point inside.

    Parameters:
    -----------

    Returns:
    --------

    """
    x_loc = np.arange(min_val[0], max_val[0], consecuative_dis)
    y_loc = np.arange(min_val[1], max_val[1], consecuative_dis)
    z_loc = np.arange(min_val[2], max_val[2], consecuative_dis)
    centroids = np.array(list(set(itertools.product(x_loc, y_loc, z_loc))))
    index_centroids = []
    x = np.array(skeleton_coordinates[' x'])
    y = np.array(skeleton_coordinates[' y'])
    z = np.array(skeleton_coordinates[' z'])
    xyz = np.zeros([len(x), 3])
    xyz[:, 0] = x
    xyz[:, 1] = y
    xyz[:, 2] = z
    for i in range(centroids.shape[0]):
        print i
        n_point = np.where(np.linalg.norm(xyz - centroids[i, 0],
                                          ord=1, axis=1)< radius)[0].shape[0]
        if n_point > min_points_in_box:
            index_centroids.append(i)
    return centroids[index_centroids, :]

def branch_decompose_one_parent_index(parent_index, down_depth):
    """decomposing a parent index from all the nodes which means that for each node in the
    tree, make a subtree that that node is its root. if down_depth=-1 it will have all the
    possible nodes, otherwise, it only considers the subtree up to the given depth.

    Parameters:
    -----------
    parent_index: numpy

    down_depth: int
        if -1, then it doesn't consider depth, otherwise it will cut the subtree up to tha
        depth.

    Returns:
    --------
    list_subtree: list
        the parent index of all subtrees.
    """
    length = len(parent_index)
    adjacency = np.zeros([length, length])
    adjacency[parent_index[1:], range(1, length)] = 1
    full_adjacency = np.linalg.inv(np.eye(length) - adjacency)
    list_subtree = []
    depth = full_adjacency.sum(axis=0) - 1
    for i in range(parent_index.shape[0]):
        ids_in_subtree = np.where(full_adjacency[i,:])[0]
        up_to_depth = np.where(depth < depth[i] + down_depth)[0]
        if down_depth != -1:
            ids_in_subtree = ids_in_subtree[np.in1d(ids_in_subtree, up_to_depth)]
        sub_full = full_adjacency[np.ix_(ids_in_subtree, ids_in_subtree)]
        sub_adj = np.eye(sub_full.shape[0]) - np.linalg.inv(sub_full)
        list_subtree.append(np.argmax(sub_adj, axis=0))
    return list_subtree

def get_regularized(Subsample, list_swc):
    """Returning the regularized sunsample of neurons. The regularized subsample of a neuron
    only consider the nodes that are root, branch or terminal.

    Parameters:
    -----------
    Subsample: Subsample object from McNeuron
        a subsample class from McNeuron package

    list_swc: list
        A list of swc of many neurons.

    Returns:
    --------
    reg_swc: list
        A list of swc of regularized subsample of the neurons.
    """
    reg_swc = []
    for i in range(len(list_swc)):
        Subsample.set_swc(list_swc[i])
        Subsample.fit()
        reg_swc.append(Subsample.subsample(subsample_type='regular'))
    return reg_swc

def get_parent_index(list_swc):
    """Returning the parent index of a list of swc of many neurons.

    Parameters:
    -----------
    list_swc: list
        A list of swc of many neurons.

    Returns:
    --------
    parent_list: list
        parent index of the neurons. Parent index start with 0 (root) and the parent id of each node
        is based on the python numbering (start from 0)
    """
    parent_list = []
    for i in range(len(list_swc)):
        parent_index = list_swc[i][:,6].astype(int) -1
        parent_index[0] = 0
        parent_list.append(parent_index)
    return parent_list

def branch_decompose(parent_list, down_depth=-1):
    """Retrun all possible subtrees of a list of trees up to the given depth. Trees are represented
    as parent index and subtrees are the part of trees that start with onde node on the
    tree. if down_depth=-1 it will have all the possible nodes, otherwise, it only considers
    the subtree up to the given depth.

    Parameters:
    -----------
    parent_list: list
        parent index of the neurons. Parent index start with 0 (root) and the parent id of each node
        is based on the python numbering (start from 0)

    down_depth: int
        if -1, then it doesn't consider depth, otherwise it will cut the subtree up to tha
        depth.

    Returns:
    --------
    branch_list: list
        list of all possible subtrees in parent_list up to the depth down_depth.
    """
    branch_list = []
    for parent_index in parent_list:
        list_subtree = branch_decompose_one_parent_index(parent_index,
                                                         down_depth=down_depth)
        branch_list.extend(list_subtree)
    return branch_list

def count_motifs(motifs, trees):
    """Counting the number of motifs in trees. motifs are trees also.
    Parameters:
    -----------
    motifs: list
        a list of trees (parent index)

    trees: list
        a list of trees (parent index)

    Returns:
    --------
    count: numpy
        an array with the same size of motifs. For each motifs it counts how many of them
        are in the trees.
    """
    n_motifs = len(motifs)
    count = np.zeros(n_motifs)
    for t in trees:
        for i in range(n_motifs):
            count[i] += are_two_trees_same(t, motifs[i])
    return count
############### Utility functions #################

def get_depth(index_id, parent_id):
    """Returning the (centrifugal) depth of each node. Notice that index should be between 0 and number of
    nodes and the parent of soma should be n_node (number of nodes).

    Parameters:
    -----------
    index_id: numpy
        id of nodes.
    parent_id: numpy
        parent_id of nodes

    Returns:
    --------
    depth: numpy
        (centrifugal) depth of each node.
    """
    n_node = len(index_id)
    ext_parent_id = np.append(parent_id, n_node)
    ext_inv_index_id = inverse_array(np.append(index_id, n_node)).astype(int)
    depth = np.zeros(n_node+1)
    grand_parent_id = ext_parent_id
    while set(grand_parent_id)!={n_node}:
        far_index = grand_parent_id!=n_node
        depth[far_index] = depth[far_index] + 1
        grand_parent_id = ext_parent_id[ext_inv_index_id[grand_parent_id]]
    depth = depth[:-1]
    depth = depth.astype(int)
    return depth

def refresh_id(index_id, parent_id):
    """Relabeling all the id from 0 to the number of different ids. It sorts all the
    indcies and then relabel them. It applys to both index_id and parent_id simultaneously.
    Parameters:
    -----------
    index_id: numpy
        id on nodes.

    parent_id: numpy
        index of parent of nodes.

    Returns:
    --------
    new_index_id, new_parent_id: both numpy
    """
    le = preprocessing.LabelEncoder()
    le.fit(np.append(index_id, parent_id))
    new_index_id = le.transform(index_id)
    new_parent_id = le.transform(parent_id)
    return new_index_id, new_parent_id

def inverse_array(array):
    """giving the inverse permutation of an array"""
    inverse = np.zeros(array.shape[0])
    inverse[array] = np.arange(array.shape[0])
    return inverse
