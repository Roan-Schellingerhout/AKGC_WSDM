import networkx as nx

# Global placeholders in the worker process memory
_H = None
_G = None

def init_worker(H_shared, G_shared):
    """
    This runs ONCE when each worker process starts.
    It sets the global graphs for that specific process.
    """
    global _H, _G
    _H = H_shared
    _G = G_shared

def extract_balanced_subgraph(source, target, max_neighbors=15, hub_threshold=500):
    # Uses the global _H and _G instead of passed arguments
    blacklist = {'xsd:integer', 'xsd:string', 'xsd:decimal', '0', '1', 'none'}
    
    if (not source in _H) or (not target in _G):
        return None
    
    try:
        # --- STAGE 1: THE SPINE ---
        try:
            path = nx.bidirectional_shortest_path(_H, source, target)
            final_nodes = set(path)
        except nx.NetworkXNoPath:
            final_nodes = {source, target}

        # --- STAGE 2: THE FLESH ---
        for seed in [source, target]:
            if seed not in _H: continue
            
            potential_neighbors = [
                n for n in _H[seed] 
                if _H.degree(n) < hub_threshold 
                and str(n).lower() not in blacklist
            ]
            
            sorted_neighbors = sorted(potential_neighbors, key=lambda x: _H.degree(x))
            final_nodes.update(sorted_neighbors[:max_neighbors])

        # --- STAGE 3: THE BRIDGE ---
        sub = _G.subgraph(final_nodes).copy()
        if not nx.has_path(sub.to_undirected(), source, target):
            sub.add_edge(source, target, edge_type="gets_recommended")
            
        return sub
    except Exception:
        return None

def extraction_worker(pair):
    """Now only takes the pair. Graphs are pulled from global scope."""
    source, target = pair
    return extract_balanced_subgraph(source, target)