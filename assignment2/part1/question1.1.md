1.

 **i.** Run the given code with net_type=`Net1', and vary different conv_type from `valid', `replicate', `reflect', `circular', `sconv' and `fconv'. Report the validation and test scores in the form of a table.

**ii.** Look at the data samples from the train and test sets provided in *README.md*. Based on the structure of the images, guess the pattern of class label 0 and class label 1.  How do the samples in the train set differ from the ones in the test set?



i: Net1 Results

| Conv Type | Validation Mean (%) | Validation Std (%) | Test Mean (%) | Test Std (%) |
| --------- | ------------------- | ------------------ | ------------- | ------------ |
| valid     | 100.00              | 0.0000             | 0.09          | 0.1136       |
| replicate | 98.30               | 0.4025             | 93.11         | 2.1116       |
| reflect   | 100.00              | 0.0000             | 0.00          | 0.0000       |
| circular  | 98.94               | 1.8975             | 81.30         | 3.5228       |
| sconv     | 98.88               | 0.2135             | 6.53          | 0.3551       |
| fconv     | 87.60               | 1.0964             | 88.75         | 0.4153       |



ii.

The pattern for label 0: One red dot and one green dot. The green dot is to the right of the red dot.

The pattern for label 1: One red dot and one green dot. The green dot is to the left of the red dot.


Sample distribution differences:

In the train set, samples with Label 0 (Green right of Red) feature dots in the upper part of the image, and samples with Label 1 (Green left of Red) feature dots in the lower part of the image.

However, in the test set, suchspatial distribution is reversed.Samples with Label 0 (Green right of Red) feature dots in the lower part of the image, and samples with Label 1 (Green left of Red) feature dots in the upper part of the image.





1b

**i.** What is the difference between conv_typeconv_type "valid", "sconv" and "fconv"? Why do the test accuracies of conv_type=conv_type= "valid", "sconv" and "fconv" (i.e., acc_valid,acc_sconvacc_valid,acc_sconv and acc_fconvacc_fconv) follow the order -- acc_valid<acc_sconv<acc_fconvacc_valid<acc_sconv<acc_fconv?

Hint: Pay attention to the "kernel\_size" and "stride" used in the Conv2D layers.}

**ii.** Why is the test accuracy of conv_type=conv_type= "reflect" less than "fconv"?

**iii.** Why is the test accuracy of conv_type=conv_type= "replicate" more than "fconv"?


1c
i. Run the given code with 
net_type
=
“Net2”
net_type=“Net2”
 and vary 
conv_type
conv_type
 across “valid”, “replicate”, “reflect”, “circular”, “sconv”, and “fconv”. Report the validation and test scores in a table.
ii. Do the test accuracies of each 
conv_type
conv_type
 under 
net_type
=
“Net2”
net_type=“Net2”
 increase or decrease relative to their corresponding 
conv_type
conv_type
 counterparts in “Net1”?
iii. Explain the reason behind the observed changes in test accuracy.


i: Net2 Results

| Conv Type | Validation Mean (%) | Validation Std (%) | Test Mean (%) | Test Std (%) |
|-----------|---------------------|--------------------|--------------|--------------|
| valid     |            100.00 |           0.0000 |        0.00 |      0.0000 |
| replicate |            100.00 |           0.0000 |        0.00 |      0.0000 |
| reflect   |            100.00 |           0.0000 |        0.00 |      0.0000 |
| circular  |            100.00 |           0.0000 |        0.00 |      0.0000 |
| sconv     |            100.00 |           0.0000 |        0.00 |      0.0000 |
| fconv     |            100.00 |           0.0000 |        0.00 |      0.0000 |

ii: Comparison of Test Accuracy (Net2 vs Net1)

| Conv Type | Net1 Test (%) | Net2 Test (%) | Change (%) | Direction |
|-----------|---------------|---------------|------------|----------|
| valid     |         0.09 |         0.00 |     -0.09 | Decreased |
| replicate |        93.11 |         0.00 |    -93.11 | Decreased |
| reflect   |         0.00 |         0.00 |     +0.00 | No change |
| circular  |        81.30 |         0.00 |    -81.30 | Decreased |
| sconv     |         6.53 |         0.00 |     -6.53 | Decreased |
| fconv     |        88.75 |         0.00 |    -88.75 | Decreased |