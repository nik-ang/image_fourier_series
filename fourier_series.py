import numpy as np
import matplotlib.pyplot as plt

PERIOD = 4
DUTY_CYCLE = 0.5
STEP = 0.01

def generate_square_signal(time_length: float, time_step: float, 
						   period: float, duty_cycle: float, amplitude: float):
	t = np.arange(0, time_length, time_step)
	x = np.where(np.mod(t, period) < period * duty_cycle, amplitude, - amplitude)

	return (t, x)

def fourier_transform(signal: np.ndarray):
	f = np.fft.fft(signal)
	f = np.fft.fftshift(f)
	return f

def fourier_series(signal: np.ndarray, relative_bw: float = 1):
		
	f = fourier_transform(signal)
	N = signal.shape[0]
	n_array = np.arange(0, N)
	k_array = np.arange(-int(N/2), int(N/2))

	if relative_bw > 1 or relative_bw < 0: relative_bw = 1

	print("relative bandwidth used:", relative_bw)
	x_f = np.zeros(N)
	lower_slice = int(N/2) - int(relative_bw/2 * N)
	upper_slice = int(N/2) + int(relative_bw/2 * N)
	print(k_array[lower_slice : upper_slice])
	for k in k_array[lower_slice : upper_slice]:
		x_f = x_f + (f[int(N/2) + k] * np.exp(1j * (2*np.pi / N) * k * n_array))
	return np.real(x_f) / N

def fourier_series2(signal: np.ndarray, relative_bw: float = 1):

	if relative_bw > 1 or relative_bw < 0: relative_bw = 1
	N = signal.shape[0]
	band_slice = int(relative_bw * N)
	
	f = fourier_transform(signal)
	n_array = np.arange(0, N)
	k_array = np.arange(-int(band_slice / 2), int(band_slice / 2))

	print("relative bandwidth used:", relative_bw)
	
	nk_mesh = np.meshgrid(n_array, k_array, indexing="ij")
	e_tensor = np.exp(1j * 2*np.pi / N * nk_mesh[0]*nk_mesh[1])
	slice_index = int((N + band_slice) / 2)
	f_k = f[-slice_index : slice_index]

	x_n = np.tensordot(f_k, e_tensor, axes=(0, 1))
	x_n = x_n / N
	return np.real(x_n)


def main():
	(t, x) = generate_square_signal(20, STEP, PERIOD, DUTY_CYCLE, 1)

	shape_length = int(PERIOD / STEP)

	print("array length {}".format(t.shape[0]))
	print("shape length {}".format(shape_length))

	x_f = fourier_series2(x, relative_bw=0.1)
	#x_f = fourier_series(x, relative_bw=0.1)

	plt.plot(t, x_f, label="Fourier Series")
	plt.plot(t, x, label="Signal")
	plt.legend(loc='upper right')
	plt.title("Fourier Series")
	plt.show()

if __name__ == "__main__":
	main()
