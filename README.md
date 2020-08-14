# location sequence prediction (work in progress)
# the idea
is to predict locations e.g. of debris in a debris field based on the sequence, direction and distance distribution of debris in previously analysed debris fields  
# the goal
the goal is to train a model that will predict the next possible locations.
Find a way to make predictions more accurate as more debris locations are uncovered.
# the generated debris field training data
the training data generator generates x half circular virtual (very simplified) debris fields as cartesian cordinates, then determines the convex hull.
All coordinates are connected by a minimum spanning tree, and the edges sorted in sequences starting from each point of the convex hull, following the next adjacent edge.
This results in n sequences of edges per simulated field.

# TODO
Train model .... 
Get useful predictions




