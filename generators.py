import networkx as nx
import numpy as np
import torch

class NetworkStructureGenerator:
    def __init__(self, num_nodes, preset="small_world", **kwargs):
        """
        Generates a network graph based on the specified preset.

        Parameters:
        - num_nodes (int): Number of nodes in the network.
        - preset (str): Type of network structure.
          Options: "small_world", "scale_free", "echo_chambers", "polarized_groups"
        - kwargs: Additional parameters for specific graph models.
        """
        self.num_nodes = num_nodes
        self.preset = preset
        self.kwargs = kwargs

    def generate(self):
        """Generates the network graph based on the selected preset."""
        if self.preset == "small_world":
            # Watts-Strogatz Small World Network
            k = self.kwargs.get("k", 4)  # Each node is connected to k nearest neighbors
            p = self.kwargs.get("p", 0.1)  # Rewiring probability
            G = nx.watts_strogatz_graph(self.num_nodes, k, p)

        elif self.preset == "scale_free":
            # Barabási-Albert Scale-Free Network
            m = self.kwargs.get("m", 2)  # Each new node connects to m existing nodes
            G = nx.barabasi_albert_graph(self.num_nodes, m)

        elif self.preset == "echo_chambers":
            # Clustered communities with strong intra-group connections
            num_communities = self.kwargs.get("num_communities", 4)
            intra_prob = self.kwargs.get("intra_prob", 0.8)
            inter_prob = self.kwargs.get("inter_prob", 0.05)
            G = nx.stochastic_block_model(
                sizes=[self.num_nodes // num_communities] * num_communities,
                p=[[intra_prob if i == j else inter_prob for j in range(num_communities)]
                   for i in range(num_communities)]
            )

        elif self.preset == "polarized_groups":
            # Two opposing groups with weak inter-group connections
            group_size = self.num_nodes // 2
            intra_prob = self.kwargs.get("intra_prob", 0.9)
            inter_prob = self.kwargs.get("inter_prob", 0.02)
            G = nx.stochastic_block_model(
                sizes=[group_size, self.num_nodes - group_size],
                p=[[intra_prob, inter_prob], [inter_prob, intra_prob]]
            )

        else:
            raise ValueError("Invalid preset! Choose from 'small_world', 'scale_free', 'echo_chambers', or 'polarized_groups'.")

        return G


class PopulationGenerator:
    def __init__(self, G: nx.Graph, 
                 belief_dist="normal", openness_dist="uniform", confidence_bound_dist="uniform",
                 belief_params=None, openness_params=None, confidence_bound_params=None, preset=None):
        """
        Initialize the population generator.

        Parameters:
        - G (nx.Graph): The network graph.
        - belief_dist (str): Type of belief distribution ("normal", "bimodal", "beta").
        - openness_dist (str): Type of openness distribution ("uniform", "normal", "beta").
        - confidence_bound_dist (str): Type of confidence bound distribution.
        - preset (str): Preset mode for specific population structures ("echo_chambers", "polarized_groups").
        - belief_params (dict): Parameters for belief distribution.
        - openness_params (dict): Parameters for openness distribution.
        - confidence_bound_params (dict): Parameters for confidence bound distribution.
        """
        self.G = G
        self.belief_dist = belief_dist
        self.openness_dist = openness_dist
        self.confidence_bound_dist = confidence_bound_dist
        self.preset = preset

        self.belief_params = belief_params if belief_params else {}
        self.openness_params = openness_params if openness_params else {}
        self.confidence_bound_params = confidence_bound_params if confidence_bound_params else {}

    def generate_beliefs(self, num_nodes):
        """Generates belief values based on the selected distribution."""
        if self.preset == "echo_chambers":
            # Create strongly clustered beliefs
            communities = list(nx.community.louvain_communities(self.G, seed=42))
            beliefs = np.zeros(num_nodes)
            for i, community in enumerate(communities):
                mean = (-1) ** i * 0.8  # Alternate between strong positive and negative beliefs
                print(mean)
                std = 0.2
                community_nodes = list(community)
                beliefs[community_nodes] = np.random.normal(loc=mean, scale=std, size=len(community_nodes))
        
        elif self.preset == "polarized_groups":
            # Two opposing groups with minimal overlap
            half_size = num_nodes // 2
            beliefs = np.concatenate([
                np.random.normal(loc=-0.9, scale=0.1, size=half_size),
                np.random.normal(loc=0.9, scale=0.1, size=num_nodes - half_size)
            ])
            np.random.shuffle(beliefs)  # Shuffle to avoid order bias
        
        else:
            # Default behavior based on specified belief distribution
            if self.belief_dist == "normal":
                mean = self.belief_params.get("mean", 0)
                std = self.belief_params.get("std", 0.5)
                beliefs = np.random.normal(loc=mean, scale=std, size=num_nodes)
            
            elif self.belief_dist == "bimodal":
                size1 = num_nodes // 2
                size2 = num_nodes - size1
                beliefs = np.concatenate([
                    np.random.normal(loc=-0.8, scale=0.2, size=size1),
                    np.random.normal(loc=0.8, scale=0.2, size=size2)
                ])
            
            elif self.belief_dist == "beta":
                a = self.belief_params.get("a", 5)
                b = self.belief_params.get("b", 2)
                beliefs = 2 * np.random.beta(a, b, size=num_nodes) - 1  # Scale to [-1,1]
            
            else:
                raise ValueError("Unsupported belief distribution type!")
        
        return np.clip(beliefs, -1, 1)  # Clip between -1 and 1

    def generate_openness(self, num_nodes):
        """Generates openness values based on the selected distribution."""
        if self.preset == "polarized_groups":
            # Lower openness for highly polarized groups
            openness = np.random.uniform(0.1, 0.3, size=num_nodes)
        else:
            if self.openness_dist == "uniform":
                a = self.openness_params.get("a", 0.1)
                b = self.openness_params.get("b", 0.9)
                openness = np.random.uniform(a, b, size=num_nodes)

            elif self.openness_dist == "normal":
                mean = self.openness_params.get("mean", 0.5)
                std = self.openness_params.get("std", 0.1)
                openness = np.random.normal(loc=mean, scale=std, size=num_nodes)

            elif self.openness_dist == "beta":
                a = self.openness_params.get("a", 2)
                b = self.openness_params.get("b", 5)
                openness = np.random.beta(a, b, size=num_nodes)
            
            else:
                raise ValueError("Unsupported openness distribution type!")
        
        return np.clip(openness, 0, 1)  # Openness should always be in [0,1]
    
    def generate_confidence_bound(self, num_nodes):
        """Generates confidence bound values based on the selected distribution."""
        if self.confidence_bound_dist == "uniform":
            a = self.confidence_bound_params.get("a", 0.1)
            b = self.confidence_bound_params.get("b", 0.5)
            confidence_bound = np.random.uniform(a, b, size=num_nodes)

        elif self.confidence_bound_dist == "normal":
            mean = self.confidence_bound_params.get("mean", 0.3)
            std = self.confidence_bound_params.get("std", 0.1)
            confidence_bound = np.random.normal(loc=mean, scale=std, size=num_nodes)

        elif self.confidence_bound_dist == "beta":
            a = self.confidence_bound_params.get("a", 2)
            b = self.confidence_bound_params.get("b", 5)
            confidence_bound = np.random.beta(a, b, size=num_nodes) * 0.5  # Scale to [0, 0.5]
        
        else:
            raise ValueError("Unsupported confidence bound distribution type!")
        
        return np.clip(confidence_bound, 0, 0.5)  # Confidence bound should be between 0 and 0.5

    def generate(self):
        """Generates the population and returns belief and openness tensors."""
        num_nodes = len(self.G.nodes)
        beliefs = self.generate_beliefs(num_nodes)
        openness = self.generate_openness(num_nodes)
        confidence_bound = self.generate_confidence_bound(num_nodes)

        x = torch.from_numpy(beliefs).float().view(-1, 1)  # Belief tensor
        alpha_matrix = torch.from_numpy(openness).float().view(-1, 1)  # Openness tensor
        confidence_bound_tensor = torch.from_numpy(confidence_bound).float().view(-1, 1)  # Confidence tensor

        return x, alpha_matrix, confidence_bound_tensor
