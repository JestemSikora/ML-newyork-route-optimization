import osmnx as ox
import matplotlib.pyplot as plt


# Street network for whole borough
G = ox.graph_from_place("Manhattan, New York, USA", network_type="drive")

G_proj = ox.project_graph(G)

# First Node
address = "Sutton Place, Manhattan, New York, USA"
target_point = ox.geocode(address) 
target_node_1 = ox.nearest_nodes(G, target_point[1], target_point[0])

# Second Node
address = "Upper East Side, Manhattan, New York, USA"
target_point = ox.geocode(address) 
target_node_2 = ox.nearest_nodes(G, target_point[1], target_point[0])

print(f'first node: {target_node_1}, second node: {target_node_2}')


routes = ox.routing.k_shortest_paths(G, target_node_1, target_node_2, k=3, weight="length")
fig, ax = ox.plot.plot_graph_routes(
    G,
    list(routes),
    route_colors="y",
    route_linewidth=4,
    node_size=0,
)
