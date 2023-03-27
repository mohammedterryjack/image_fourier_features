# image_fourier_features
Fast Fourier Transform to extract Image Features

```python
from image_features import Image, image_similarity

image_blender_blue = Image.open("images/blender_blue.jpg") 
image_blender_orange = Image.open("images/blender_orange.jpg") 
image_other = Image.open("images/other.jpeg") 
```

```python
image_similarity(image_blender_blue, image_blender_orange)
```
![](images/blender_blue.jpg)
![](images/blender_orange.jpg)
>>> 0.9838861100641383

```python
image_similarity(image_blender_blue, image_other)
```
![](images/blender_blue.jpg)
![](images/other.jpg)
>>> 0.7792162468558569
