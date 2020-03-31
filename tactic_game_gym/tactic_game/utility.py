import numpy as np
import math
def compass_to_rgb(h, s=0.5, v=1):#Thanks https://python-forum.io/Thread-Python-function-to-get-colorwheel-RGB-values-from-compass-degree
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return [r, g, b]
def get_n_colors(n=2):
	#n is self.sides
	output = []
	max_angle = 360
	interval = max_angle / n
	angles = [interval*i for i in range(1, n+1)]
	for i in range(len(angles)):
		angle = angles[i]
		rgb_color = compass_to_rgb(angle)
		output.append(rgb_color)
	output = np.asarray(output, dtype=np.uint8)
	return output
def get_network(ids, num_subs):
    #smaller ids lead to higher values
    #Example graph
    #0 - 1 - 5
    #      - 6
    #      - 7
    #      - 8
    #  - 2
    #  - 3
    #  - 4 - 17
    #      - 18
    #      - 19
    #      - 20
    #Graph size: 1 + x + x^2 + x^3 ... + x^n = (x^n+1 - 1)/(x-1)
    #Thus, graph rank x is given by log base x (graph_size*(x-1)+1) - 1 however, remove - 1 so there are no rank 0
    def get_graph_rank(graph_size, num_subs=num_subs):
        graph_rank =  math.log(graph_size*(num_subs-1)+1, num_subs)
        graph_rank = math.ceil(graph_rank)
        return graph_rank
    output = []
    graph_rank = get_graph_rank(len(ids), num_subs)
    for i in range(len(ids)):
        rank = graph_rank - get_graph_rank(i+1, num_subs) + 1
        superior = i // num_subs
        if i % num_subs == 0:
            superior -= 1
        if superior == -1:
            superior = 0
        if i*num_subs+1 >= len(ids):
            player_net = {"superior": int(ids[superior]), "subordinates": None, "rank": rank}
            output.append(player_net)
            continue
        subordinates = list(range(i*num_subs+1, (i+1)*num_subs+1 if (i+1)*num_subs+1 <= len(ids) else len(ids)))
        player_net = {"superior": int(ids[superior]), "subordinates": [int(ids[subordinate]) for subordinate in subordinates], "rank": rank}
        output.append(player_net)
    return output