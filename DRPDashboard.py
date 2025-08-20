import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
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

# Title of the Dashboard
st.title("Network Optimization Dashboard")

# --- 3. Data Loading and Graph Creation ---

st.sidebar.header("Data Uploader")
uploaded_file = st.sidebar.file_uploader("Upload your supply chain network data (CSV)", type="csv")

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Initial graph setup
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['source_location'], row['destination_location'], weight=row['travel_cost'])
        
    locations = list(G.nodes())
else:
    st.info("Please upload a CSV file to begin.")
    st.stop()


# --- 4. Sidebar Controls ---

st.sidebar.header("Route & Scenario Options")
source_location = st.sidebar.selectbox("Select Source Location", locations)
destination_location = st.sidebar.selectbox("Select Destination Location", locations)

# Filter by Travel Cost Threshold
max_cost = st.sidebar.slider(
    "Filter by Max Edge Cost",
    min_value=0,
    max_value=int(df['travel_cost'].max()),
    value=int(df['travel_cost'].max())
)

# Toggle Optimization Scenarios
scenario = st.sidebar.selectbox(
    "Toggle Optimization Scenarios",
    ("No Disruption", "Port Closure", "Major Highway Accident")
)

# Build the graph based on the user's filter and scenario
filtered_df = df[df['travel_cost'] <= max_cost].copy()

if scenario == "Port Closure":
    if ('Port_X' in G and 'Warehouse_1' in G['Port_X']):
        filtered_df.loc[
            (filtered_df['source_location'] == 'Port_X') & (filtered_df['destination_location'] == 'Warehouse_1'), 
            'travel_cost'
        ] = 1000
elif scenario == "Major Highway Accident":
    if ('Warehouse_1' in G and 'Store_B' in G['Warehouse_1']):
        filtered_df.loc[
            (filtered_df['source_location'] == 'Warehouse_1') & (filtered_df['destination_location'] == 'Store_B'), 
            'travel_cost'
        ] = 1000

# Convert the filtered DataFrame to a graph
filtered_G = nx.DiGraph()
for _, row in filtered_df.iterrows():
    filtered_G.add_edge(row['source_location'], row['destination_location'], weight=row['travel_cost'])

graph_dict = {node: {} for node in filtered_G.nodes()}
for u, v, data in filtered_G.edges(data=True):
    graph_dict[u][v] = data['weight']

# --- 5. Main Panel Layout ---

# Shortest Path & Cost
st.header("1. Interactive Network Graph")
path, cost = find_shortest_path_bucket_dijkstra(
    graph_dict, 
    source_location, 
    destination_location, 
    max_cost=int(filtered_df['travel_cost'].max()) + 1000
)

# Create a Pyvis network object
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", cdn_resources="in_line")
net.set_edge_smooth('dynamic')

# Add nodes and edges to the network
for node in filtered_G.nodes():
    color = "rgba(135, 206, 235, 0.8)"
    size = 15
    if node == source_location:
        color = "rgba(144, 238, 144, 1)"
        size = 35
    elif node == destination_location:
        color = "rgba(240, 128, 128, 1)"
        size = 35
    elif node in path:
        size = 20
    
    net.add_node(node, label=node, title=node, color=color, size=size)

for u, v, d in filtered_G.edges(data=True):
    color = "gray"
    width = 1
    is_in_path = (u, v) in zip(path, path[1:])
    
    if is_in_path:
        color = "red"
        width = 4
        
    net.add_edge(u, v, value=d['weight'], title=f"Cost: {d['weight']}", color=color, width=width)

# Disable physics for a stable, non-volatile graph
net.set_options("""
var options = {
  "physics": {
    "enabled": false
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 200,
    "dragNodes": true,
    "zoomView": true
  },
  "nodes": {
    "font": { "color": "#eeeeee" },
    "borderWidth": 2,
    "shape": "dot"
  },
  "edges": {
    "arrows": { "to": { "enabled": true } },
    "color": { "inherit": false },
    "smooth": { "enabled": true, "type": "dynamic" }
  }
}
""")

# Save and display the network
net.save_graph('network.html')
HtmlFile = open("network.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, height=600)

# --- 6. Shortest Path and Cost Summary ---
st.header("2. Shortest Path and Cost")
if cost == float('inf'):
    st.error("No path found between the selected locations with the current filter and scenario.")
else:
    st.success(f"Optimal Path Found! Total Cost: **{cost}**")
    st.write("Path: " + " → ".join(path))

# --- 7. Top Costly Routes (Bar Chart) ---
st.header("3. Top Costly Routes")
costly_routes_df = filtered_df.sort_values(by='travel_cost', ascending=False).head(10)
costly_routes_df['route'] = costly_routes_df['source_location'] + ' → ' + costly_routes_df['destination_location']
st.bar_chart(costly_routes_df, x='route', y='travel_cost')

# --- 8. Optimization Summary Table ---
st.header("4. Optimization Summary Table")
summary_data = {
    "Metric": ["Shortest Path Cost", "Source Location", "Destination Location", "Current Scenario"],
    "Value": [cost, source_location, destination_location, scenario]
}
summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

# --- 9. Scenario Comparison (Optional) ---
# This section can be built out further if you need more complex comparisons.
# For now, it's a place to add a note or simple explanation.
st.header("5. Scenario Comparison (Optional)")
st.info("The selected scenario affects the network. You can change the scenario in the sidebar to see how the optimal route changes in real-time.")

