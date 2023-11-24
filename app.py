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
	
	def fourier_series2D(self, max_width: int, f: np.ndarray = None):

		if f is None:
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
			np.meshgrid(n_coords[0], k_tensor[0], indexing="ij"),
			np.meshgrid(n_coords[1], k_tensor[1], indexing="ij")
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

		x_nm = np.tensordot(f_k_tensor, exp_tensor[0], axes=(0, 1))
		x_nm = np.tensordot(x_nm, exp_tensor[1], axes=(0, 1))
		x_nm = x_nm / (N[0] * N[1])

		return np.real(x_nm) 
		
 
	def show(self, max_width = 1000):
		cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
		cv.imshow(self.name, self.resize(max_width))
		cv.waitKey(0)
		cv.destroyAllWindows()

	def show_series_as_image(self, max_width = 1000):
		series = self.fourier_series2D(max_width)
		cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
		cv.imshow(self.name, series.astype(np.uint8))
		cv.waitKey(0)
		cv.destroyAllWindows()

	def plot_fourier_series(self, max_width = 1000):
		fourier_transform = np.fft.fft2(self.resize(max_width))
		fourier_transform = np.fft.fftshift(fourier_transform)
		fourier_series = self.fourier_series2D(max_width=max_width, f=fourier_transform)

		plt.style.use("dark_background")
		fig = plt.figure()

		ax1 = plt.subplot(221)
		ax1.imshow(self.resize(max_width), cmap='gray')
		ax1.set_axis_off()
		ax1.set_title("Original Image")

		ax2 = plt.subplot(222)
		ax2.imshow(np.log(np.abs(fourier_transform)))
		ax2.set_title("Spectrum")

		ax3 = plt.subplot(2,2, (3, 4))
		ax3.imshow(fourier_series, cmap="gray")
		ax3.set_axis_off()
		ax3.set_title("Fourier Series")

		fig.tight_layout(pad=0.5)
		plt.show()



def main():
	photo = image('mri.jpg', "mri picture")
	#photo.show()
	#photo.show_series_as_image(1000)
	photo.plot_fourier_series()
	


if __name__ == '__main__':
	main()