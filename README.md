# ear-project

### 1. Problem Definition

To retrieve the correct child ID in the first position, if the child is in the
database, from images taken from the past.

We divide it into the following sub-questions:

1. Whether or not the left and the right ears are symmetric?
2. What are the unique features among the children?
3. What are the growth pattern?
4. ...

### 2. Methods

For sub-question 1,

1. Transform the template using different in-/off-plane rotation angles and scaling factors.
2. For each image, find the transformed template that matches the image best.
3. For each image, find the region of interest (ROI) defined by the template and standardize the ROI.
4. Compute the average of the two copies of the same ear during the same visit.
5. Compute the matching scores (normalized correlation coefficient and mutual information) between each averaged left ear and all others within the same visit.
6. Compare 1) the probabilistic distribution of the matching score between one left ear and its associated right ear; 2) the probabilistic distribution of the maximum matching score within the same visit, and 3) the probabilistic distribution of the average matching scores within the same visit.

Note that for the following id-visit#, the left and right ears are swapped.

- HE3-visit1
- S5E-visit1
- 3CI-visit2
- EUD-visit2

And for the following id-visit#, the image frame is rotated manually.

- 3VA-visit1
- BL2-visit1
- 3VA-visit2
- 4Q9-visit2
- DNF-visit2
- JB6-visit2
