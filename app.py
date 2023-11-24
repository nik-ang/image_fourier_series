import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class image(object):
	def __init__(self, directory: str, name: str):
		self.name = name
		self.image = cv.imread(directory, cv.IMREAD_GRAYSCALE)

	def fourier_transform2D(self):
		fourier_transformed = np.fft.fft2(self.image)
		fourier_transformed = np.fft.fftshift(fourier_transformed)
		return fourier_transformed

	def resize(self, new_width: int):
		dimensions = None
		(h, w) = self.image.shape[:2]
		ratio = h / w
		dimensions = (new_width, int(new_width * ratio))
		return cv.resize(self.image, dimensions)
	
	def fourier_series2D(self, max_width):
		f = np.fft.fft2(self.resize(max_width))
		f = np.fft.fftshift(f)

		N = self.resize(max_width).shape
		n_coords = (np.arange(0, N[0]), np.arange(0, N[1]))
		relative_bw = (1, 1)

		band_slice = (
			int(relative_bw[0] * N[0]),
			int(relative_bw[1] * N[1])
		)
		k_tensor = (
			np.arange(-int(band_slice[0] / 2), int(band_slice[0] / 2)),
			np.arange(-int(band_slice[1] / 2), int(band_slice[1] / 2))	
		)
		nk_mesh = (
			np.meshgrid(n_coords[0], k_tensor[0], indexing="xy"),
			np.meshgrid(n_coords[1], k_tensor[1], indexing="xy")
		)
		
		exp_tensor = (
			np.exp(1j * 2*np.pi / N[0] * nk_mesh[0][0] * nk_mesh[0][1]),
			np.exp(1j * 2*np.pi / N[1] * nk_mesh[1][0] * nk_mesh[1][1])
		)

		slice_indices = (
			int((N[0] + band_slice[0]) / 2),
			int((N[1] + band_slice[1]) / 2)
		)

		f_k_tensor = f[-slice_indices[0] : slice_indices[0], -slice_indices[1] : slice_indices[1]]
		plt.imshow(np.log(np.abs(f_k_tensor)))
		plt.show()

		x_f = np.tensordot(f_k_tensor, exp_tensor[0], axes=(0, 0))
		x_f = np.tensordot(x_f, exp_tensor[1], axes=(0, 0))

		plt.imshow(np.real(x_f), cmap="gray")
		plt.show()

		return np.real(x_f) / (N[0] * N[1])
		
		
 
	def show(self, max_width = 1000):
		cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
		cv.imshow(self.name, self.resize(max_width))
		cv.waitKey(0)
		cv.destroyAllWindows()

	def show_series(self, max_width = 1000):
		series = self.fourier_series2D(max_width)
		cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
		cv.imshow(self.name, series)
		cv.waitKey(0)
		cv.destroyAllWindows()



def main():
	photo = image('mri.jpg', "mri picture")
	photo.show()
	photo.show_series()
	


if __name__ == '__main__':
	main()