# White Balance Color Augmenter

## What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance

*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1</sup> and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>1,2</sup>
<br></br><sup>1</sup>York University  <sup>2</sup>Samsung Research

#### [Project page](http://cvil.eecs.yorku.ca/projects/public_html/wb_emulation/index.html)

<br>
<img src="https://drive.google.com/uc?export=view&id=1hFq00SmUo4-xUZFJQAqMRQNkwO6NIu6w" style="width: 350px; max-width: 100%; height: auto" title="Click for the larger version." />

Our augmentation method can accurately emulate realistic color constancy degradation. Existing color augmentation methods often generate unrealistic colors which rarely happen in reality (e.g., green skin or purple grass). More importantly, the visual appearance of existing color augmentation techniques does not well represent the color casts produced by incorrect WB applied onboard cameras, as shown below.

<img src="https://drive.google.com/uc?export=view&id=1xDF4mjD9AyIEAgbVT38ygRvw0K_o3MN-" style="width: 350px; max-width: 100%; height: auto" title="Click for the larger version." />


### Quick start
#### 1. Matlab:
 1. Run `install_.m`
 2. Try our demos: 
    * `demo_single_image` to process signle image
    * `demo_batch` to process an image directory
    * `demo_WB_color_augmentation` to process an image directory and repeating the corresponding ground truth files for our generated images
    * `demo_GUI` (located in `GUI` directory) for a GUI interface 

#### MIT License

### Publication
Mahmoud Afifi and Michael S. Brown. What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance. International Conference on Computer Vision (ICCV), 2019.



