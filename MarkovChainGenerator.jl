using PyCall

py"""
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import ujson
import networkx as nx
import networkx.algorithms.cycles as nxc

class MarkovChainGenerator:
    
    def __init__(self):
        pass
    
    def create_graph_from_adj_matrix(self, adj_mat):
        size, _ = adj_mat.shape
        
        G = nx.Graph()
        for i in range(size):
            G.add_node(i)
        
        rows, cols = np.where(adj_mat == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)
        return G
    
    def random_adj_matrix(self, m, prob):
        M = np.zeros((m, m), dtype=int)
        for i in range(m):
            for j in range(m):
                if j > i:
                    if npr.random() < prob:
                        M[i, j] = 1
                        M[j, i] = 1
        return M
    
    def transition_rate_matrix(self, connected_loopy_graph, rand_max = 1.0):
        # W[j, i] = transition rate of W(i -> j)
        adj_mat = nx.adjacency_matrix(connected_loopy_graph)
        (N, _) = adj_mat.shape
        M = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if adj_mat[i, j] == 1:
                    M[i, j] = npr.random() * rand_max
                    
        for i in range(N):
            M[i, i] = -np.sum(M[:, i])
            
        return M
    
    def steady_state_distribution(self, transition_rate_matrix): # Mapleと一致
        TRM = np.matrix(transition_rate_matrix)
        evs, evecs = npl.eig(TRM)
        zero_index = np.argmin(np.abs(evs))
        
        tmp = (evecs[:, zero_index]).A1 / np.sum((evecs[:, zero_index]).A1)
        return tmp.real

    def random_connected_graph(self, m, prob = 0.5):
        if m < 2:
            raise ValueError('The node number N should be N >= 2')

        M = self.random_adj_matrix(m, prob)
        G = self.create_graph_from_adj_matrix(M)
        
        if nx.is_connected(G) == False:
            return self.random_connected_graph(m, prob = prob)
        else:
            return G
    
    def random_connected_loopy_graph(self, m, prob = 0.5):
        M = self.random_adj_matrix(m, prob)
        G = self.create_graph_from_adj_matrix(M)
        
        cycle_basis = nxc.cycle_basis(G)
        if len(cycle_basis) == 0 or nx.is_connected(G) == False:
            return self.random_connected_loopy_graph(m, prob = prob)
        else:
            return G
        
    def cycle_basis_current_matrix(self, G): # Check OK
        cb_list = nx.cycle_basis(G)
        current_matrix_list = []
        s = len(cb_list)
        dim = G.number_of_nodes()
        for i in range(s):
            cbasis = cb_list[i]
            loop_len = len(cbasis)
            M = np.zeros((dim, dim), dtype=int)
            for l in range(loop_len - 1):
                M[cbasis[l],cbasis[l + 1]] = 1
                M[cbasis[l + 1],cbasis[l]] = -1
            M[cbasis[loop_len-1],cbasis[0]] = 1
            M[cbasis[0],cbasis[loop_len-1]] = -1
            current_matrix_list.append(M)
        return current_matrix_list
    
    def entropy_production(self, transition_rate_matrix):
        W = np.matrix(transition_rate_matrix)
        Pss = self.steady_state_distribution(W)
        (N, _) = W.shape
        
        tmp = 0
        for i in range(N):
            for j in range(N):
                if i != j and np.abs(W[i, j]) > 1.0E-10:
                    tmp += Pss[i]*W[j, i] * np.log((Pss[i] * W[j, i]) / (Pss[j] * W[i, j]))
        
        return tmp

"""