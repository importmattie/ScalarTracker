import numpy as np

# scalar: the field being tracked (currently looks for mins though)
# mask: an array with 1s and 0s, where 1s represent a "hit" to track
# lons,lats: describing the longitude and latitude in x and y
# timeBefore: the number of days before a "hit" to track
# timeAfter: the number of days after a "hit" to track
# radius: the number of grid cells to search
# min/max X/Y Int0: outlining the subset region to search for the event on day0
# trajX,trajY: tracking the location of the storm each day

def findPath(scalar,mask,lons,lats,timeBefore,timeAfter,
             radius,minXInt0,maxXInt0,minYInt0,maxYInt0,
             trajX,trajY):

    # NOTE: future versions should only track positive anomalies...
    # leave it to the user to make sure the data passed is positive
    # the sign can always be added back later

    lonInterval = lons[1]-lons[0]
    latInterval = lats[1]-lats[0]

    totalTime = len(mask)
    scalarAnom = np.zeros((totalTime,len(lats),len(lons)))

    # currently optimized for months with 30 day length
    # should be changed to be more broadly applicable?
    # or require the anomaly as the input? <- this
    for tt in range(0,totalTime/30):
        scalarAnom[tt*30:tt*30+30,:,:] = (scalar[tt*30:tt*30+30,:,:]-
                                          scalar[tt*30:tt*30+30,:,:].mean(0))

    for ii in range(0,totalTime):
        if(mask[ii] == 1):

            minValueIndicesDay0 = getMinIndices(scalarAnom,ii,
                                                minYInt0,maxYInt0,
                                                minXInt0,maxXInt0)

            # (If you do the math, the location of day 0 is always the same as the positive
            # number of days before. i.e. if 2 days before, day 0 will be the 2nd index of
            # the array, starting from 0th.)
            # (If daysBefore = 0, then day0 is at index 0 and we're simply tracking after
            # a given date.

            lastX = minValueIndicesDay0[1]+minXInt0
            lastY = minValueIndicesDay0[0]+minYInt0
            trajX[timeBefore].append(lons[lastX])
            trajY[timeBefore].append(lats[lastY])
            # Arbitrarily requiring that we must match sign with the original anomaly if
            # we are to continue
            minScalar = 0.0

            fillNANs = 0
            for tt in range(1,timeBefore+1):
                if(fillNANs == 0):
                    minScalar = 0.0
                    needNextPoint = 1
                    iterations = 0
                    while(needNextPoint == 1):
                        # Get the minimum and consider it a candidate for tt
                        # days before day0
                        tempMinValueIndices = getMinIndices(scalarAnom,ii-tt,
                                                            max((lastY-radius),0),
                                                            min((lastY+radius),len(lats)-1),
                                                            max((lastX-radius),0),
                                                            min((lastX+radius),len(lons)-1))
                        newLastXCand = tempMinValueIndices[1]+max((lastX-radius),0)
                        newLastYCand = tempMinValueIndices[0]+max((lastY-radius),0)
                        # Make sure the point we find in the box is within the radius
                        # (i.e. exclude the grid cells outside of the circle with radius r
                        # Better future solution - perform a mask ahead of time on those
                        # points outside the radius? A subroutine to generate the mask?
                        distance = ((newLastXCand-lastX)**2+(newLastYCand-lastY)**2)**0.5
                        if(distance < radius):
                            # If true, we found our candidate
                            if(scalarAnom[ii-tt,newLastYCand,newLastXCand] < minScalar):
                                lastX = newLastXCand
                                lastY = newLastYCand
                                trajX[timeBefore-tt].append(lons[lastX])
                                trajY[timeBefore-tt].append(lats[lastY])
                            # If not, no more negative values...give up and write all nans
                            # back in time
                            else:
                                trajX[timeBefore-tt].append(np.nan)
                                trajY[timeBefore-tt].append(np.nan)
                                fillNANs = 1
                            # Either way, we're done with this loop
                            needNextPoint = 0
                        else:
                            # rule out this point being the minimum going forward
                            scalarAnom[ii-tt,newLastYCand,newLastXCand] = 1.e6
                            # failsafe in case we somehow get stuck with large radius
                            # give up and write nans
                            if(iterations > 20):
                                trajX[timeBefore-tt].append(np.nan)
                                trajY[timeBefore-tt].append(np.nan)
                                fillNANs = 1
                                needNextPoint = 0
                        iterations = iterations + 1
                else:
                    trajX[timeBefore-tt].append(np.nan)
                    trajY[timeBefore-tt].append(np.nan)

            # reset - now counting forward from 0
            lastX = minValueIndicesDay0[1]+minXInt0
            lastY = minValueIndicesDay0[0]+minYInt0

            # Same loop, but +tt instead of -tt
            # Moving to a function in future versions
            fillNANs = 0
            for tt in range(1,timeAfter+1):
                if(fillNANs == 0):
                    minScalar = 0.0
                    needNextPoint = 1
                    iterations = 0
                    while(needNextPoint == 1):
                        tempMinValueIndices = getMinIndices(scalarAnom,ii+tt,
                                                            max((lastY-radius),0),
                                                            min((lastY+radius),len(lats)-1),
                                                            max((lastX-radius),0),
                                                            min((lastX+radius),len(lons)-1))
                        newLastXCand = tempMinValueIndices[1]+max((lastX-radius),0)
                        newLastYCand = tempMinValueIndices[0]+max((lastY-radius),0)
                        distance = ((newLastXCand-lastX)**2+(newLastYCand-lastY)**2)**0.5
                        if(distance < radius):
                            if(scalarAnom[ii+tt,newLastYCand,newLastXCand] < minScalar):
                                lastX = newLastXCand
                                lastY = newLastYCand
                                trajX[timeBefore+tt].append(lons[lastX])
                                trajY[timeBefore+tt].append(lats[lastY])
                            else:
                                trajX[timeBefore+tt].append(np.nan)
                                trajY[timeBefore+tt].append(np.nan)
                                fillNANs = 1
                            needNextPoint = 0
                        else:
                            scalarAnom[ii+tt,newLastYCand,newLastXCand] = 1.e6
                            if(iterations > 20):
                                trajX[timeBefore+tt].append(np.nan)
                                trajY[timeBefore+tt].append(np.nan)
                                fillNANs = 1
                                needNextPoint = 0
                        iterations = iterations + 1
                        
                else:
                    trajX[timeBefore+tt].append(np.nan)
                    trajY[timeBefore+tt].append(np.nan)

# Return a tuple with the x and y location in the scalar
# of the minimum
def getMinIndices(scalarAnom,time,minYInt,maxYInt,minXInt,maxXInt):

    minValueIndices = ()

    if(minYInt != maxYInt or minXInt != maxXInt):
        minValueIndices = np.unravel_index(np.argmin(scalarAnom[time,minYInt:maxYInt+1,
                                                                minXInt:maxXInt+1]),
                                           ((maxYInt-minYInt+1),(maxXInt-minXInt+1)))
    else:
        minValueIndices = (0,0)

    return minValueIndices
