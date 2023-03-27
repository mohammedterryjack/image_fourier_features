from numpy import concatenate, ndarray
from numpy.fft import fft2
from numpy.linalg import norm
from PIL import Image


def image_similarity(image_a:Image, image_b: Image) -> float:
  return cosine_similarity(
    a=featurise_image_data(image=image_a), 
    b=featurise_image_data(image=image_b)
  )

def featurise_image_data(image: ndarray, max_image_size: int = 50) -> ndarray:
  grayscale_image = image.convert("L")
  thumbnail_image = grayscale_image.resize((max_image_size, max_image_size))
  fourier_image = fft2(thumbnail_image)
  fourier_features_complex = fourier_image.reshape(-1)
  fourier_features = concatenate([fourier_features_complex.real, fourier_features_complex.imag])
  return fourier_features

def cosine_similarity(a: ndarray, b: ndarray) -> float:
  return (a @ b.T) / (norm(a) * norm(b))
