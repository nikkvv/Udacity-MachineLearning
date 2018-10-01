""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    xmin = float(min(arr))
    xmax = float(max(arr))
    scaled_arr = [(x-xmin)/(xmax-xmin) for x in arr]
    return scaled_arr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)

