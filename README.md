# ML_Project (SPINEWISE)

Multivariant Time Series clustering

Built a model to try to cluster human movements detected by sensors worn behind the neck (_n) and on a belt (_r)

The pipeline uses a few different methods to filter the input data from an accelerometer, gyroscope, magnetometer, and a calculation device.  

The accelerometer gyroscope provide insights on movements in the x,y,z plains.

The effectiveness of the methods used in the pipeline were tested using labelled movement data provided from the client.
Using mostly the data from the accelerometer, gyroscope, and output from the calculation device on the belt.  

These methods were then further used on unlabelled movement data in attempts to cluster unlablled data into distinct movements.  

Our pipeline works in a more modular manner, to give freedom to the user to choose the features they would like to examine, preprocessing filters
they find give the most clarity in the data before clustering, and the option to use our clustering methods or insert other clustering methods in 
our clustering file. 

Given the size, and variety of the data inputs, we found that this modularity was key for better and more distinctive clustering.


Data Preprocessing:

Hillbert Filter from sklearn

Haar filter from Pytw

Symbolic Aggregession(SAX) from Tslearn

Principal Component Analysis(PCA) from sklearn

Clustering Methods:

K_means

DBSCAN

Results can be plotted after each cluster method

Function main.py contains the pipeline.  It takes in input from the user as well as a '.csv' file, the input from the user determines which filtering, clustering

## Skills and Tools

## Results

First we were able to associate one or 2 clusters with the labels from the labelled data, depending on the chosen input features.

After this, we ran our methods on unlabelled data to try and find new clusters of movements.  

[! image](link ) labeled classes

## License

Free license

## Github link:

https://github.com/SpineWiseTeam4/ML_Project


## Contributers:

Amanuel Zeredawit: junior ML engineer at Becode

https://github.com/AmanuelZeredawit

Maysa AbuSharekh: junior ML engineer at Becode

https://github.com/maysahassan

Sam Fooks: junior ML engineer at Becode

https://github.com/samuelfooks

## Acknowledgments

BeCode Arai4 AI coaches(Chrysanthi and Louis)









	

