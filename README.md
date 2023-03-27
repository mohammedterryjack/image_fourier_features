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
> 0.9838861100641383

![](images/blender_blue.jpg)
![](images/blender_orange.jpg)


```python
image_similarity(image_blender_blue, image_other)
```
> 0.7792162468558569

![](images/blender_blue.jpg)
![](images/other.jpeg)
