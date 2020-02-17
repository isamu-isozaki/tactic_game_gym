import numpy as np
def get_n_colors(self, n=2):
	#n is self.sides
	output = []
	max_color = 256**3
	interval = max_color / n
	colors = [interval*i for i in range(1, n+1)]
	bases = [256**2, 256, 1]
	for i in range(len(colors)):
		color = colors[i]
		rgb_color = []
		for j in range(3):
			color_comp = min(color // bases[j],256)
			color_comp -= 1
			color_comp = int(color_comp)
			rgb_color.append(color_comp)
			color -= color_comp*bases[j]
		output.append(rgb_color)
	output = np.asarray(output)
	return output