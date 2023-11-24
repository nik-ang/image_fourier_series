# image_fourier_series
Decomposes an image into fourier 2D fourier components and reconstructs it via 2D Fourier Series using a chosen relative bandwidth.
100% of the bandwidth recorvers the original image

## How it works
Reads an image as a 2D Numpy Array, which we will handle as a (pseudo)tensor $I^{hw}$. The axis represent the height and width axis of the image.

## Signals and System Theory: 1D Discrete Fourier Series
The Discrete Fourier Transform of a sequence is given by

$$
	f[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N}nk}
$$

And the inverse discrete fourier transform is the fourier series.

$$
	x[n] = \frac{1}{N} \sum_{k = 0}^{N-1} f[k] e^{j \frac{2\pi}{N}nk}
$$

Which can be written as a tensor product (using Einstein's Summation Notation)

$$
	x^n = \frac{1}{N} f^k E^{nk}
$$

Where $E^{nk}$ is a tensor such that for any given indices $\alpha \beta$:

$$
	E^{\alpha \beta} = e^{e^{j \frac{2\pi}{N}\alpha\beta}}
$$

To get a fourier series using only part of the bandwidth we can just limit the frequencies k we use.

$f^k$ we get using numpy's fourier transform function

```python
f = np.fft.fft(signal)
f = np.fft.fftshift(f)
```

It's worth mentioning that since we are working with complex fourier series, we are also using negative frequencies. The defined bandwidth contains the negative frequencies as well. That is the reason we are shifting $f$ with numpy.

To create $E^{nk} we use a mesh. For that we generate all values of $n$ and $k$ we need, given a limited bandwidth. We define a with as integer now

```python
band_slice = int(relative_bw * N)
```
And with that we generate our n and k arrays for the mesh

```python
n_array = np.arange(0, N)
k_array = np.arange(-int(band_slice / 2), int(band_slice / 2))
```

We create the mesh

```python
nk_mesh = np.meshgrid(n_array, k_array, indexing="ij")
```

And use it to create the tensor with the help of numpy. The first component of the mesh are the n-values and the second are the k-values

```python
e_tensor = np.exp(1j * 2*np.pi / N * nk_mesh[0]*nk_mesh[1])
```

We don't forget to slice our used frequencies

```python
slice_index = int((N + band_slice) / 2)
f_k = f[-slice_index : slice_index]
```

And now we can finally multiply the tensors

$$
	x^n = \frac{1}{N} f^k E^{nk}
$$

```python
x_n = np.tensordot(f_k, e_tensor, axes=(0, 1))
x_n = x_n / N
```

## 2D Discrete Fourier Series

The 2D Fourier series of a 2D Array is

$$
	x[n,m] = \frac{1}{N_1 N_2} \sum_{\mu = 0}^{N - 1} \sum_{\nu = 0}^{M - 1} f[\mu, \nu] e^{j \frac{2\pi}{N}n\mu} e^{j \frac{2\pi}{M}m\nu}
$$

As a tensor product

$$
	x^{nm} = \frac{1}{N_1 N_2} f^{\mu \nu} E^{n\mu m\nu}
$$

The problem with this is that allocating a 4D array requires Petabytes of memor, which is why the $E$ tensor has to be treated as the outer product of two tensors of second order

$$
	x^{nm} = \frac{1}{N_1 N_2} f^{\mu \nu} \Alpha^{n\mu}\Beta^{m\nu}
$$

First, we find the product

$$
	f^{\mu \nu}\Alpha^{n\mu} = \Gamma^{\nu n}
$$

Then

$$
	\Gamma^{\nu n} \Beta^{m\nu} = \Chi^{nm}
$$

and finally

$$
	x^{nm} = \frac{1}{N_1 N_2} \Chi^{nm}
$$

The generation of the meshes and tensors is analogue to the 1D version but using tuples of meshes. The final tensor product is

```python
x_nm = np.tensordot(f_k_tensor, exp_tensor[0], axes=(0, 1))
x_nm = np.tensordot(x_nm, exp_tensor[1], axes=(0, 1))
x_nm = x_nm / (N[0] * N[1])
```

