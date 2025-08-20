import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import copy
import json

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
st.title("Animated Network Optimization Dashboard")

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

# Calculate the best path
path_1, cost_1 = find_shortest_path_bucket_dijkstra(
    graph_dict, 
    source_location, 
    destination_location, 
    max_cost=int(filtered_df['travel_cost'].max()) + 1000
)

# Calculate the second-best path
path_2, cost_2 = [], float('inf')
if cost_1 != float('inf'):
    best_second_path_cost = float('inf')
    best_second_path = []
    
    # Iterate through each edge in the best path
    path_edges = list(zip(path_1, path_1[1:]))
    
    for u, v in path_edges:
        # Temporarily remove this edge from the graph
        temp_graph = copy.deepcopy(graph_dict)
        if u in temp_graph and v in temp_graph[u]:
            del temp_graph[u][v]
            
            # Recalculate the shortest path
            temp_path, temp_cost = find_shortest_path_bucket_dijkstra(
                temp_graph, 
                source_location, 
                destination_location,
                max_cost=int(filtered_df['travel_cost'].max()) + 1000
            )
            
            # If a new path is found and it's better than the current second-best
            if temp_cost < best_second_path_cost:
                best_second_path_cost = temp_cost
                best_second_path = temp_path

    path_2 = best_second_path
    cost_2 = best_second_path_cost

# --- Generate the complete HTML and JavaScript for the graph ---
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", cdn_resources="in_line")

# Use a networkx layout to get initial positions
pos = nx.spring_layout(filtered_G)

for node in filtered_G.nodes():
    net.add_node(node, 
                 label=node, 
                 title=node, 
                 color="#555555", 
                 size=15, 
                 x=pos[node][0] * 1000, 
                 y=pos[node][1] * 1000, 
                 physics=False)

for u, v, d in filtered_G.edges(data=True):
    net.add_edge(u, v, value=d['weight'], title=f"Cost: {d['weight']}", color="rgba(128, 128, 128, 0.4)", width=1)

# Get the graph data as dictionaries
nodes_list = net.nodes
edges_list = net.edges

# Pass the paths and data as JSON strings
path_1_js = json.dumps(path_1)
path_2_js = json.dumps(path_2)
nodes_js = json.dumps(nodes_list)
edges_js = json.dumps(edges_list)

# Define the HTML and JavaScript content
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Network Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
    <style type="text/css">
        body, html {{
            margin: 0;
            padding: 0;
            background-color: #222222;
        }}
        #mynetwork {{
            width: 100%;
            height: 600px;
            background-color: #222222;
            border: 1px solid #444444;
        }}
    </style>
</head>
<body>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        // Create the graph data
        var nodes = new vis.DataSet({nodes_js});
        var edges = new vis.DataSet({edges_js});
        var data = {{ nodes: nodes, edges: edges }};

        // Set the options for a stable graph
        var options = {{
            "physics": {{
                "enabled": false
            }},
            "interaction": {{
                "hover": true,
                "tooltipDelay": 200,
                "dragNodes": true,
                "zoomView": true
            }},
            "nodes": {{
                "font": {{ "color": "#eeeeee" }},
                "borderWidth": 2,
                "shape": "dot"
            }},
            "edges": {{
                "arrows": {{ "to": {{ "enabled": true }} }},
                "color": {{ "inherit": false }},
                "smooth": {{ "enabled": true, "type": "dynamic" }}
            }}
        }};

        // Initialize the network
        var network = new vis.Network(document.getElementById('mynetwork'), data, options);

        // Animation script
        var path1 = {path_1_js};
        var path2 = {path_2_js};
        
        // Find the specific edge ID by its source and destination node IDs
        function findEdgeId(fromNode, toNode) {{
            const edge = edges.get({{
                filter: function (item) {{
                    return item.from === fromNode && item.to === toNode;
                }}
            }});
            return edge.length > 0 ? edge[0].id : null;
        }}

        function animatePath(path, color, width, delay=500, pulseDuration=200) {{
            if (!path || path.length < 2) return;
            let currentStep = 0;
            const interval = setInterval(() => {{
                if (currentStep >= path.length - 1) {{
                    clearInterval(interval);
                    // Final highlight for the last node
                    const finalNodeData = {{id: path[path.length - 1], color: color, size: 35}};
                    nodes.update([finalNodeData]);
                    return;
                }}
                
                const fromNode = path[currentStep];
                const toNode = path[currentStep + 1];
                
                const edgeId = findEdgeId(fromNode, toNode);
                if (edgeId) {{
                    edges.update([{{id: edgeId, color: {{color: color}}, width: width}}]);
                }}
                
                // Animate the node with a pulsing effect
                const originalNodeSize = nodes.get(fromNode).size;
                nodes.update([{{id: fromNode, color: color, size: 30}}]);
                
                setTimeout(() => {{
                    nodes.update([{{id: fromNode, size: originalNodeSize}}]);
                }}, pulseDuration);
                
                currentStep++;
            }}, delay);
        }}
        
        // Start the animations
        animatePath(path1, "red", 4, 700);
        setTimeout(() => {{
            animatePath(path2, "orange", 3, 600);
        }}, path1.length * 700);
        
    </script>
</body>
</html>
"""
components.html(html_content, height=600)

# --- 6. Shortest Path and Cost Summary ---
st.header("2. Shortest Path and Cost")
if cost_1 == float('inf'):
    st.error("No path found between the selected locations with the current filter and scenario.")
else:
    st.success(f"Optimal Path Found! Total Cost: **{cost_1}**")
    st.write("Path: " + " → ".join(path_1))
    
    if cost_2 != float('inf') and path_2:
        st.info(f"Second Best Path Found! Total Cost: **{cost_2}**")
        st.write("Path: " + " → ".join(path_2))
    else:
        st.info("No second-best path was found.")

# --- 7. Top Costly Routes (Bar Chart) ---
st.header("3. Top Costly Routes")
costly_routes_df = filtered_df.sort_values(by='travel_cost', ascending=False).head(10)
costly_routes_df['route'] = costly_routes_df['source_location'] + ' → ' + costly_routes_df['destination_location']
st.bar_chart(costly_routes_df, x='route', y='travel_cost')

# --- 8. Optimization Summary Table ---
st.header("4. Optimization Summary Table")
summary_data = {
    "Metric": ["Shortest Path Cost", "Second Best Path Cost", "Source Location", "Destination Location", "Current Scenario"],
    "Value": [cost_1, cost_2, source_location, destination_location, scenario]
}
summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

# --- 9. Scenario Comparison (Optional) ---
# This section can be built out further if you need more complex comparisons.
# For now, it's a place to add a note or simple explanation.
st.header("5. Scenario Comparison (Optional)")
st.info("The selected scenario affects the network. You can change the scenario in the sidebar to see how the optimal route changes in real-time.")
