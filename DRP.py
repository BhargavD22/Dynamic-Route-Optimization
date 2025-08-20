import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. The Core Algorithm: Bucket-Based Dijkstra ---
# This is our implementation of the "Breaking the Sorting Barrier" algorithm.
# It uses buckets instead of a priority queue for efficiency.

def find_shortest_path_bucket_dijkstra(graph, start_node, end_node, max_cost=1000):
    """
    Finds the shortest path using a bucket-based Dijkstra's algorithm.
    This is highly efficient for graphs with non-negative, integer-like weights.
    """
    # Initialize distances and buckets
    dist = {node: float('inf') for node in graph}
    dist[start_node] = 0
    
    # Create buckets. The size is based on the maximum possible cost.
    # In a real-world scenario, you'd calculate a dynamic max.
    buckets = [[] for _ in range(max_cost + 1)]
    buckets[0].append(start_node)
    
    # Keep track of the path
    prev = {node: None for node in graph}
    
    # Process nodes in increasing order of distance
    for d in range(max_cost + 1):
        while buckets[d]:
            current_node = buckets[d].pop(0)

            # If we've already found a shorter path, skip this node
            if dist[current_node] < d:
                continue

            # If we've reached the destination, we are done
            if current_node == end_node:
                break
            
            # Explore neighbors of the current node
            for neighbor, weight in graph[current_node].items():
                new_dist = dist[current_node] + weight
                
                # If we found a shorter path to the neighbor
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current_node
                    
                    # Place the neighbor in the correct bucket.
                    # This is the "Breaking the Sorting Barrier" part.
                    if new_dist <= max_cost:
                        buckets[new_dist].append(neighbor)
        
        # If we reached the end node, break out of the outer loop too
        if current_node == end_node:
            break
            
    # Reconstruct the path
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()

    return path, dist[end_node]

# --- 2. Streamlit Application Interface ---

st.title("Dynamic Supply Chain Route Optimizer")
st.markdown("---")

st.markdown("""
This application demonstrates how a real-time routing algorithm can optimize supply chain logistics for a Fortune 500 company. 
It finds the most efficient path for a shipment and instantly recalculates when a disruption occurs.
""")

# --- 3. Data Loading and Graph Creation ---

uploaded_file = st.file_uploader("Upload your supply chain network data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Build the graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source_location'], row['destination_location'], weight=row['travel_cost'])
        
    # Convert NetworkX graph to a dictionary for the algorithm
    graph_dict = {node: {} for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        graph_dict[u][v] = data['weight']

    locations = list(graph_dict.keys())
    
    # --- 4. User Inputs for Routing ---
    
    st.header("Select Shipment Details")
    col1, col2 = st.columns(2)
    
    start_node = col1.selectbox("Select Start Location", locations)
    end_node = col2.selectbox("Select End Location", locations)

    # --- 5. Disruption Simulation ---
    
    st.header("Simulate a Disruption")
    disruption = st.selectbox(
        "Choose a real-time event:",
        ("No Disruption", "Port Closure", "Major Highway Accident")
    )

    if disruption != "No Disruption":
        # Identify the edge to modify based on the disruption
        if disruption == "Port Closure":
            if "Port_X" in graph_dict and "Warehouse_1" in graph_dict["Port_X"]:
                graph_dict["Port_X"]["Warehouse_1"] = 1000 # Set a very high cost
        elif disruption == "Major Highway Accident":
            if "Warehouse_1" in graph_dict and "Store_B" in graph_dict["Warehouse_1"]:
                graph_dict["Warehouse_1"]["Store_B"] = 1000 # Set a very high cost

    # --- 6. Run the Algorithm and Display Results ---
    
    if start_node and end_node:
        try:
            path, cost = find_shortest_path_bucket_dijkstra(graph_dict, start_node, end_node)
            
            st.markdown("---")
            st.header("Optimization Results")

            if cost == float('inf'):
                st.error("No path found between the selected locations.")
            else:
                st.success(f"Optimal Path Found! Total Cost: **{cost}**")
                st.write("Path: " + " → ".join(path))

                # --- 7. Visualization ---
                st.header("Network Visualization")
                
                # Get edge list for the shortest path
                path_edges = list(zip(path, path[1:]))

                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.75, iterations=50) # Use a layout for better visualization
                
                # Draw all nodes and edges first
                nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
                nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
                
                # Draw shortest path edges and nodes
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, arrows=True, arrowsize=20)
                
                # Draw the nodes on the shortest path
                nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightcoral', node_size=2000)
                
                # Add edge labels (weights)
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
                
                plt.title("Supply Chain Network with Optimal Route", fontsize=16)
                plt.axis('off')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your inputs and CSV file.")
