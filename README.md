# Summary

This is a basic procedural console program written in CUDA C. First, it collects values representing the bounds of a section of the complex plane that will be seen in the generated image, and so these values can be used to zoom in or out of a given Julia set. It then accepts user input for defining which Julia set is generated, and then subsequently collects transform values for choosing a color scheme for the generated set. The program then saves the color data to disk on the Windows desktop with a user-defined file name.

If you are not familiar with Julia sets or how they are generated and colored using the method employed by this program, have a look at these links:

https://en.wikipedia.org/wiki/Julia_set

https://www.youtube.com/watch?v=2AZYZ-L8m9Q

If you run into any issues compiling or running the program or have any questions – just let me know!

-Chris

<br>

# Sample Input and Output

Complex plane bounds:
  
  - Left Bound = -2.13
  
  - Right Bound = 2.13
  
  - Upper Bound = 1.2
  
  - Lower Bound = -1.2

<br>

Julia set selection:
  
  - Real Component = 0.0
  
  - Imaginary Component = 0.67

<br>

Color gradient transform functions:
  
  - d (red) = -13.3
  
  - k (red) = 0.045
  
  - a (red) = -130
  
  - c (red) = 160

 <br>

  - d (green) = 0
  
  - k (green) = 0.045
  
  - a (green) = -80
  
  - c (green) = 40

<br>

  - d (blue) = 29.3
  
  - k (blue) = 0.1
  
  - a (blue) = 60
  
  - c (blue) = 80

<br>

File name:
  
  - 0.0 + 0.67i

<br>

Output:

![0.0 + 0.67i](https://github.com/RealTimeChris/Julia-Set-Generator-CUDA/blob/main/Images/0.0%20%2B%200.67i.jpg)

<br>

# Notes
- Be sure that the character set is Unicode in the Visual Studio project settings.

- You may need to change the CUDA worker group Grid and Block dimensions to suit the specs of your hardware, as shown here; (This is controlled using the dim3-typed variables named “threadsPB” and “grid” in the code)
	https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications

