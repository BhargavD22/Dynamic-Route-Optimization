import streamlit as st
import pandas as pd
import networkx as nx
# The pyvis library will create an interactive network graph
from pyvis.network import Network
import streamlit.components.v1 as components

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

                # --- 7. Visualization with Pyvis ---
                st.header("Network Visualization")
                
                # Create a Pyvis network object
                net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", cdn_resources="in_line")
                net.set_edge_smooth('dynamic')
                
                # Add nodes and edges from the NetworkX graph
                for node in G.nodes():
                    color = "skyblue"
                    size = 15
                    if node == start_node:
                        color = "lightgreen"
                        size = 30
                    elif node == end_node:
                        color = "lightcoral"
                        size = 30
                    elif node in path:
                        size = 20
                        
                    net.add_node(node, label=node, title=node, color=color, size=size)
                
                for u, v, d in G.edges(data=True):
                    color = "gray"
                    width = 1
                    if (u, v) in zip(path, path[1:]) or (v, u) in zip(path, path[1:]):
                        color = "red"
                        width = 3
                        
                    net.add_edge(u, v, value=d['weight'], title=f"Cost: {d['weight']}", color=color, width=width)
                
                # Set physics options for animation
                net.set_options("""
                var options = {
                  "nodes": {
                    "borderWidth": 2
                  },
                  "edges": {
                    "arrows": {
                      "to": {
                        "enabled": true
                      }
                    },
                    "color": {
                      "inherit": false
                    },
                    "smooth": false
                  },
                  "physics": {
                    "barnesHut": {
                      "gravitationalConstant": -2000,
                      "centralGravity": 0.3,
                      "springLength": 100,
                      "springConstant": 0.05
                    },
                    "minVelocity": 0.75
                  }
                }
                """)
                
                # Save and display the network as an HTML file
                net.save_graph('network.html')
                
                # Display the HTML in Streamlit
                HtmlFile = open("network.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height=600)
                
        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your inputs and CSV file.")
