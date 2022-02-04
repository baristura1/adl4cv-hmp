import torch
import numpy as np
import networkx as nx




def spatio_temporal_graph(joints_to_consider, temporal_kernel_size,
                          spatial_adjacency_matrix):  # given a normalized spatial adj.matrix,creates a spatio-temporal adj.matrix

    number_of_joints = joints_to_consider

    spatio_temporal_adj = np.zeros((temporal_kernel_size, temporal_kernel_size, joints_to_consider, joints_to_consider))

    for t in range(temporal_kernel_size - 1):
        for i in range(number_of_joints):
            spatio_temporal_adj[t, t + 1, i, i] = 1  # create edge between same body joint,for t consecutive frames
            spatio_temporal_adj[t+1, t, i, i] = 1  # create edge between same body joint,for t consecutive frames

    for t in range(temporal_kernel_size):
        for j in range(number_of_joints):
            if spatial_adjacency_matrix[i, j] != 0:  # if the body joints are connected
                spatio_temporal_adj[t, t, i, j] = spatial_adjacency_matrix[i, j]
    spatio_temporal_adj=torch.from_numpy(spatio_temporal_adj)
    spatio_temporal_adj=spatio_temporal_adj.permute(0,2,1,3)
    spatio_temporal_adj=spatio_temporal_adj.reshape(joints_to_consider*temporal_kernel_size,joints_to_consider*temporal_kernel_size)
    spatio_temporal_adj=spatio_temporal_adj.numpy()
    return spatio_temporal_adj


# In[20]:

def normalize_A(A):  # given an adj.matrix, normalize it by multiplying left and right with the degree matrix, in the -1/2 power

    A = A + np.eye(A.shape[0])

    D = np.sum(A, axis=0)

    D = np.diag(np.ravel(D))

    D_inv = D ** -0.5
    D_inv[D_inv == np.infty] = 0

    return D_inv * A * D_inv

def get_adj_human(joints_to_consider, temporal_kernel_size):
    '''
    parent_nofilter = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1
    parent = parent_nofilter.tolist()
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31]).tolist()
    for idx in sorted(joint_to_ignore, reverse=True):
        del parent[idx]

    translation_list=[2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]
    '''
    edgelist = [
        (0, 4), (1, 0), (2, 1),
        (3, 2), (4, 8), (5, 4),
        (6, 5), (7, 6), (8, 0),
        (9, 8), (10, 9),
        (11, 10),
        (12, 8), (13, 12), (14, 13), (15, 14),
        (16, 14), (17, 8),
        (18, 17), (19, 18), (20, 19), (21, 19)]


    # create a graph
    G = nx.Graph()
    G.add_edges_from(edgelist)
    # create adjacency matrix
    A = nx.adjacency_matrix(G, nodelist=list(range(0, joints_to_consider))).todense()

    st_graph=spatio_temporal_graph(joints_to_consider, temporal_kernel_size, A)
    # normalize adjacency matrix

    return torch.from_numpy(normalize_A(st_graph)).float()
