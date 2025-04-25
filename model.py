import torch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, degree
from tqdm import tqdm


class BeliefMessagePassing(MessagePassing):
    """
    Implements a message-passing mechanism for belief propagation in a network.

    This model simulates how beliefs spread over a network based on neighbors' 
    beliefs and an individual's openness to external influence.

    Attributes:
        linear (torch.nn.Linear): A linear transformation applied to aggregated messages.
        num_timesteps (int): Number of propagation steps in the message-passing process.
    """

    def __init__(self, in_channels=1, out_channels=1, num_timesteps: int = 5):
        """
        Initializes the belief propagation model.

        Args:
            in_channels (int): Dimensionality of input node features. Defaults to 1.
            out_channels (int): Dimensionality of transformed node features. Defaults to 1.
            num_timesteps (int, optional): Number of message-passing iterations. Defaults to 5.
        """
        super().__init__(aggr='mean')  # Use mean aggregation of messages
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.num_timesteps = num_timesteps

    def forward(self, x, edge_index, alpha_matrix=None, confidence_bound=None, r=0.2, N_max=50000,
               gamma=0.1, lambda_m=2, verbose=False):
        """
        Forward pass for belief propagation.

        Args:
            x (torch.Tensor): Node feature matrix of shape (num_nodes, in_channels).
            edge_index (torch.Tensor): Graph connectivity in COO format.
            alpha_matrix (torch.Tensor, optional): Openness values for each node. Shape: (num_nodes, 1).

        Returns:
            torch.Tensor: Updated node features after belief propagation.
            list[torch.Tensor]: List of node embeddings over time.
        """
        # Initialize alpha_matrix (openness) if not provided
        if alpha_matrix is None:
            alpha_matrix = torch.rand((x.size(0), 1))  # Random openness values
        if confidence_bound is None:
            confidence_bound = torch.rand((x.size(0), 1)) * 0.5 # Confidence bound between 0 and 0.5

        # Store belief embeddings at each timestep
        embeddings_over_time = [x.clone().detach()]
        edge_indices_over_time = [edge_index.clone()]
        feature_snapshots = []

        node_ids = torch.arange(x.size(0))  # Initial node IDs
        node_histories = {int(i): [] for i in node_ids.tolist()}  # Dict of lists
        i_row, i_col = edge_index
        init_deg = degree(i_row, num_nodes=x.size(0), dtype=torch.float).view(-1, 1)

        for i, nid in enumerate(node_ids):
            node_features = torch.cat([x[i], alpha_matrix[i], confidence_bound[i], init_deg[i]])
            node_histories[int(nid)].append(node_features.tolist())
            
        # Message passing for multiple timesteps
        for _ in tqdm(range(self.num_timesteps), desc="Belief Propagation Steps"):
            if verbose:
                connections = edge_index.shape[1]
                pop_size = x.shape[0]
                print(f"Timestep {_}: Network Size: {pop_size} | Connections {connections}")
            
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            
            row, col = edge_index
            deg = degree(row, num_nodes=x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # Update normalization
            
            x = self.propagate(edge_index, x=x, norm=norm, alpha_matrix=alpha_matrix,
                              confidence_bound=confidence_bound)
            edge_index = self.update_edges(edge_index, x, confidence_bound)
            edge_index, x, alpha_matrix, confidence_bound, node_ids = self.node_churn(edge_index, x, 
                                                                            alpha_matrix, 
                                                                            confidence_bound,
                                                                            gamma, node_ids)
            edge_index, x, alpha_matrix, confidence_bound, node_ids = self.add_nodes(edge_index, x,
                                                                           alpha_matrix,
                                                                           confidence_bound,
                                                                           r, N_max, lambda_m, node_ids)

            embeddings_over_time.append(x.clone().detach())
            edge_index_no_loop = remove_self_loops(edge_index.clone())[0]
            edge_indices_over_time.append(edge_index_no_loop)
            new_row, new_col = edge_index_no_loop
            node_degree = degree(new_row, num_nodes=x.size(0), dtype=torch.float).view(-1, 1)
            
            timestep_features = torch.cat([x, alpha_matrix, confidence_bound], dim=1)
            feature_snapshots.append(timestep_features.clone().detach())

            for i, nid in enumerate(node_ids):
                if int(nid) not in node_histories:
                    node_histories[int(nid)] = []
                node_features = torch.cat([x[i], alpha_matrix[i], confidence_bound[i], node_degree[i]])
                node_histories[int(nid)].append(node_features.tolist())
                
        return x, embeddings_over_time, edge_indices_over_time, feature_snapshots, node_histories

    def message(self, x_j, norm):
        """
        Message function that applies degree normalization to incoming messages.

        Args:
            x_j (torch.Tensor): Features of neighboring nodes.
            norm (torch.Tensor): Degree normalization factor.

        Returns:
            torch.Tensor: Normalized messages.
        """
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, x, alpha_matrix, confidence_bound):
        """
        Update function that determines how beliefs change based on neighbors' beliefs.

        Args:
            aggr_out (torch.Tensor): Aggregated messages from neighbors.
            x (torch.Tensor): Current node beliefs.
            alpha_matrix (torch.Tensor): Openness values for each node.

        Returns:
            torch.Tensor: Updated belief values.
        """
        # Compute similarity between a node's current belief and the aggregated belief from neighbors
        similarity = torch.cosine_similarity(aggr_out, x, dim=-1, eps=1e-6).unsqueeze(-1)

        # Compute updated belief: a node is influenced by neighbors' beliefs based on similarity
        updated_belief = similarity * self.linear(aggr_out) + (1 - similarity) * x

        # Soft thresholding using a sigmoid function to gradually reduce influence
        influence_weight = torch.sigmoid((similarity - confidence_bound.view(-1, 1)) * 10)

        # Final belief update incorporating openness (alpha_matrix)
        final_belief = torch.tanh(alpha_matrix * updated_belief + (1 - alpha_matrix) * x)
        # return torch.tanh(alpha_matrix * updated_belief + (1 - alpha_matrix) * x)

        result = influence_weight * final_belief + (1 - influence_weight) * x
        
        return result

    def logistic_growth(self, num_nodes, r, N_max):
        lambda_t = r * num_nodes * (1 - (num_nodes / N_max))
        return np.random.poisson(lambda_t)

    def barabasi_albert_attachment(self, edge_index, num_nodes, new_nodes, lambda_m):
        degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float) + 1e-6
        probs = degrees / degrees.sum()

        new_edges = []
        for new_node in range(num_nodes, num_nodes + new_nodes):
            m = max(1, np.random.poisson(lambda_m))
            neighbors = np.random.choice(num_nodes, size=min(m, num_nodes), replace=False, p=probs.numpy())

            for neighbor in neighbors:
                new_edges.append([new_node, neighbor])
                new_edges.append([neighbor, new_node])

        if new_edges:
            new_edges = torch.tensor(new_edges, dtype=torch.long).T
            edge_index = torch.cat([edge_index, new_edges], dim=1)

        return edge_index

    def add_nodes(self, edge_index, x, alpha_matrix, confidence_bound, r, N_max, lambda_m, node_ids):
        num_nodes = x.size(0)

        new_nodes = self.logistic_growth(num_nodes, r, N_max)
        if new_nodes == 0:
            return edge_index, x, alpha_matrix, confidence_bound

        new_x = torch.rand((new_nodes, x.size(1))) * 2 - 1  
        new_confidence_bound = torch.rand((new_nodes, 1)) * 0.5
        new_alpha_matrix = torch.rand((new_nodes, 1))

        x = torch.cat([x, new_x], dim=0)
        confidence_bound = torch.cat([confidence_bound, new_confidence_bound], dim=0)
        alpha_matrix = torch.cat([alpha_matrix, new_alpha_matrix], dim=0)

        edge_index = self.barabasi_albert_attachment(edge_index, num_nodes, new_nodes, lambda_m)

        new_node_ids = torch.arange(max(node_ids).item() + 1, max(node_ids).item() + 1 + new_nodes)
        node_ids = torch.cat([node_ids, new_node_ids], dim=0)

        return edge_index, x, alpha_matrix, confidence_bound, node_ids

    def node_churn(self, edge_index, x, alpha_matrix, confidence_bound, gamma, node_ids):
        num_nodes = x.size(0)
        degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

        p_leave = gamma * (1 - degrees / (degrees.max() + 1e-6))
        remove_mask = torch.bernoulli(1 - p_leave).bool()

        x = x[remove_mask]
        alpha_matrix = alpha_matrix[remove_mask]
        confidence_bound = confidence_bound[remove_mask]

        mask_idx = torch.arange(num_nodes)[remove_mask]
        node_map = -torch.ones(num_nodes, dtype=torch.long)
        node_map[mask_idx] = torch.arange(len(mask_idx))

        row, col = edge_index
        valid_edges = remove_mask[row] & remove_mask[col]
        edge_index = edge_index[:, valid_edges]
        edge_index = node_map[edge_index]

        node_ids = node_ids[remove_mask]

        return edge_index, x, alpha_matrix, confidence_bound, node_ids
        
    def update_edges(self, edge_index, x, confidence_bound):
        row, col = edge_index
        belief_difference = torch.abs(x[row] - x[col])
        
        # Remove edges
        keep_edges = belief_difference < confidence_bound[row]
        edge_index = edge_index[:, keep_edges.squeeze()]

        num_nodes = x.size(0)

        # Randomly select a subset of nodes to attempt new connections
        num_active_nodes = torch.randint(1, num_nodes // 5 + 1, (1,)).item()
        active_nodes = torch.randperm(num_nodes)[:num_active_nodes]

        new_edges = []
        
        for node in active_nodes:
            belief_diff = torch.abs(x - x[node])
            valid_candidates = (belief_diff < confidence_bound[node]).nonzero(as_tuple=True)[0]

            if valid_candidates.numel() == 0:
                continue

            max_attempts = torch.randint(1, max(2, num_nodes // 10) + 1, (1,)).item()
            num_new_edges = torch.randint(1, min(max_attempts, valid_candidates.numel()) + 1, (1,)).item()
            chosen = valid_candidates[torch.randperm(valid_candidates.numel())[:num_new_edges]]

            for target in chosen:
                p_connect = 1 - (belief_diff[target] / confidence_bound[node]).item()  # Higher belief similarity → higher chance
                if torch.rand(1).item() < p_connect:  # Stochastic connection
                    new_edges.append([node, target.item()])
        
        if not new_edges:
            return edge_index
            
        new_edges = torch.tensor(new_edges).T
        
        return torch.cat([edge_index, new_edges], dim=1)