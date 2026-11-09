import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	rows, cols = new_shape
	reshaped_matrix = []
	flat = [num for row in a for num in row]
	if rows * cols == len(flat):
		for i in range(rows):
			row = []
			for j in range(cols):
				row.append(flat[i*cols + j])
			reshaped_matrix.append(row)
	return reshaped_matrix

