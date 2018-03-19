**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data.JPG
[image2]: ./output_images/Hog_Visualisation.JPG
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/testResults.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image10]: ./output_images/heatMap.JPG
[image11]: ./output_images/boundingbox.JPG
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Sixth code cell of the IPython notebook `P5-Vehicle-Detection.ipynb`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Once the features were extracted witht the right parameters, I moved on to train the data set.

#### 2. Explain how you settled on your final choice of HOG parameters.

There are a number of parameters that a HOG function expects. Below were my final chosen parameters 

| Parameter       | Value |
|-----------------|-------|
| Orientations    | 9     |
| Cells Per Block | 2     |
| Pixels Per Cell | 16    |
| Color Space     | YUV   |

I tried other combination of color spaces like `YCrCb` , `LUV` and `HLS`. I found `YUV` colospace to give to accuracy of 98% better than any of the mentioned colorspaces when tested on testing data.

**Combinations Tried** :

| S. No | Orientations | Cells Per Block | Pixels Per Cell | Color Space | HOG Channel |
|-------|--------------|-----------------|-----------------|-------------|-------------|
| 1     | 9            | 2               | 8               | LUV         | ALL         |
| 2     | 9            | 2               | 8               | LUV         | 1           |
| 3     | 11           | 2               | 16              | HLS         | 1,2         |
| 4     | 9            | 2               | 16              | YUV         | ALL         |
| 4     | 6            | 4               | 16              | YUV         | 0           |
| 5     | 12           | 4               | 16              | YCrCb       | ALL         |
| 5     | 12           | 4               | 16              | YCrCb       | 0           |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After stacking all the features extracted from vehicle and non vehicle dataset, the next part is to train the classifier. I used Spatial features and HOG features to train the data set. I ommitted the color features as it wasn't adding to any accuracy increase and to reduce the processing time.

Before training the model with the data, there's some preprocessing to be done. The data is divided into `Training` and `Test` data.
Also before splitting it was made sure that the data is properly shuffled. Furthermore, Data was normalized using Standard Scaler function of Sklearn.preprocessing class. **An important point to note here is that we do not fit the test set because by doing so we already give our classifier a sneek peak into data** 

Now the data is ready for the model. The final step is to train the classifier. I used `LinearSVC` with default parameteres. I tried with the other kernel `rbf` but but it drastically increased the processing time. 

I was able to achieve a test accuracy of 98.22 %. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search in method `slide_window `. This method returns the windows list. I decided to use the following windows after some hit and trials. Following are tweaks I used that helped reduce the processing time:

1. The sliding windows need not to be used on the upper half of image. Scanning of cars in sky doesn't make sense yet! .

2. The cars near the chosen horizon will be small and will increase in size as we move from the horizon towards the car.

3. It makes sense to search for 64x64 car in only one window starting from horizon and increase the window size as we move from horizon towards car.

4. The window sizes close to the car are chosen with lesser overlapping than the windows near the horizon.

The final windows selected were-

| Window Size | Overlap | Y Start | Y Stop |
|-------------|---------|---------|--------|
| 64x64       | 85      | 400     | 464    |
| 80x80       | 80      | 400     | 480    |
| 96x96       | 70      | 400     | 612    |
| 128x128     | 50      | 400     | 660    |
| 160x160     | 50      | 400     | 670    |

![alt text][image3]


---

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 windows scales using YUV 3-channel HOG features plus spatially binned color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

I used the technique of `Heat Map` to plot the final bounding boxes around the car. I defined a function `add_heat` that increments the pixel value of an black image the size of the original image at the location of each detected window.

![alt text][image10]

Finally the next point is to draw bounding boxes on the final image. The `scipy.ndimage.measurements.label()` function collects spatially contiguous areas of the heatmap and assigns each a label:

![alt text][image11]


The final implentation worked very well as seen from the test images above. 


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

The video pipeline is slighlty differently handled than the test images pipeline. 

Inorder to remove false positives. I used an averaging approach in the video pipeline that sums all the heats coming in the past 10 frames and then I apply threshold using the function `apply_threshold`. This function blacks out the pixels who have value less than threshold value.



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

  1. 

