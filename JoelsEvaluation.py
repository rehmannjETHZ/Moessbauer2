import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import root

mypath = os.getcwd()
VelCalPaths = []
VelCalSpd = []
for i in range(5):
    VelCalPaths += [mypath + '\Measurements\VelCal%d.txt' %(i+3)]
    VelCalSpd += [(i+3)]
CutoffPaths = []
for i in range(8):
    CutoffPaths += [mypath + '\Measurements\CutoffVel50%d.txt' %(i+1)]
NoCutoffPaths = []
for i in range(8):
    NoCutoffPaths += [mypath + '\Measurements\keinCutoff%d.txt' %(i+1)]

#data = np.loadtxt(VelCalPaths[1], skiprows=0)
#print(data)

#adapted code from 'mikuszefski' on stack overflow

def hwhm(x0, gam):
    return gam

def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def multi_lorentz( x, params ):
    off = params[0]
    paramsRest = params[1:]
    assert not ( len( paramsRest ) % 3 )
    return off + sum( [ lorentzian( x, *paramsRest[ i : i+3 ] ) for i in range( 0, len( paramsRest ), 3 ) ] )

def res_multi_lorentz( params, xData, yData ):
    diff = [ multi_lorentz( x, params ) - y for x, y in zip( xData, yData ) ]
    return diff



#Calibration 
    
Fe57 = np.array([-5.04948, -2.8167, -0.5800, 1.1007, 3.3376, 5.5754])
NumberOfFiles = np.size(VelCalPaths)
SecondPeaks = np.zeros((2, NumberOfFiles)) #Store the position of the peaks furthest appart from the middle but still visible in all files.
SecondHWHM = np.zeros((2, NumberOfFiles))
for l in range(NumberOfFiles):
    path = VelCalPaths[l]
    print('******************************************')
    print(path)
    yData = np.trim_zeros(np.loadtxt(path, skiprows=0))
    j = 0
    while yData[j] ==0:
        j+=1
    
    k = 1
    while yData[-k] ==0:
        k+=1
    
    yData = yData[5+j:-5-k]
    xData = np.linspace(0, np.size(yData), np.size(yData)) + 5+j
    yData = yData / max(yData)
    
    generalWidth = 1
    
    yDataLoc = yData
    startValues = [ max( yData ) ]
    counter = 0
    
    while max( yDataLoc ) - min( yDataLoc ) > .07:
        counter += 1
        print(counter)
        if counter > 10: ### max 10 peak...emergency break to avoid infinite loop
            break

        minP = np.argmin( yDataLoc )
        minY = yData[ minP ]
        x0 = xData[ minP ]
        startValues += [ x0, minY - max( yDataLoc ), generalWidth ]
        popt, ier = leastsq( res_multi_lorentz, startValues, args=( xData, yData ) )
        yDataLoc = [ y - multi_lorentz( x, popt ) for x,y in zip( xData, yData ) ]
    
    peaks = popt[1::3]
    print(popt)
    HWHM = np.zeros(np.size(peaks))
    for i in range(np.size(popt))[1::3]:
        HWHM[int((i-1)/3)] = hwhm(popt[i], popt[i+2])
        print('peak ', int((i-1)/3)+1,' ', popt[i], 'hwhm ', hwhm(popt[i], popt[i+2]))
    
    p = np.argsort(peaks)
    sortedPeaks = np.array(peaks)[p]
    sortedHWHM = np.array(HWHM)[p]
    print('Sorted peaks: ', sortedPeaks)
    print('Sorted HWHM: ', sortedHWHM)
    print(int(counter/2-2))
    SecondPeaks[0,l] = sortedPeaks[int(counter/2-2)]
    SecondPeaks[1,l] = sortedPeaks[int(counter/2+1)]
    
    SecondHWHM[0,l] = sortedHWHM[int(counter/2-2)]
    SecondHWHM[1,l] = sortedHWHM[int(counter/2+1)]
    
    testData = [ multi_lorentz(x, popt ) for x in xData ]
    
    fig = plt.figure()
    ax = fig.add_subplot( 1, 1, 1 )
    ax.plot( xData, yData )
    ax.plot( xData, testData )
    
    fig.savefig(path + '.pdf')
    plt.show()
    
    #Velocity calibration
    if l==2:
        calibrationPeaks = sortedPeaks
    
        
        
print('*******************************************')
print('Find zeros using varying speeds')   
print(SecondPeaks)
plt.figure()
linpar1 = np.polyfit(VelCalSpd, SecondPeaks[0, :], 1)
linpar2 = np.polyfit(VelCalSpd, SecondPeaks[1, :], 1)

line1 = np.poly1d(linpar1)
line2 = np.poly1d(linpar2)
#find root
def f(x):
    return line1(x) - line2(x)

zero = root(f, 230)
frameZero = line1(zero.x[0])
x_range = range(3, 12, 1)


fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1 )
ax.plot(x_range, line1(x_range))
ax.plot(x_range, line2(x_range))
ax.errorbar(VelCalSpd, SecondPeaks[0, :], yerr = SecondHWHM[0, :], fmt ='x')
ax.errorbar(VelCalSpd, SecondPeaks[1, :], yerr = SecondHWHM[1, :], fmt ='x')
plt.xlabel('*10 -> percent of max. range of velocity')
plt.ylabel('Normalized intensity peak position')
fig.savefig('Zero.pdf')

fig = plt.figure()
print('We have zero velocity at frame following method 1', frameZero)

calibrationPeaks = np.array([calibrationPeaks[0:3], np.array([frameZero]), calibrationPeaks[3:]])
Fe57 = np.array([Fe57[0:3], np.array([0]), Fe57[3:]])
calibrationPeaks = np.hstack(calibrationPeaks)
Fe57 = np.hstack(Fe57)


velcalpar = np.polyfit(calibrationPeaks, np.flip(Fe57), 1)
print('coefficients of the calibration function are:', velcalpar)
velcalibrator = np.poly1d(velcalpar)
plt.plot(calibrationPeaks, np.flip(Fe57),  '*', label = 'data from table')
plt.plot(calibrationPeaks, velcalibrator(calibrationPeaks), label = 'linear fit')
plt.legend(loc='best')
plt.xlabel('channels')
plt.ylabel('velocity [mm/s]')
plt.savefig('speedCalibration.pdf')
plt.show()

frameZero2 = root(velcalibrator, 230)
print('VelCalibration gives us zero at', frameZero2)
plt.show()

#Energy spectra

NumberOfFiles = np.size(NoCutoffPaths)


for l in range(NumberOfFiles):
    path2 = NoCutoffPaths[l]
    print('***************************************')
    print(path)
    y = np.loadtxt(path2, skiprows=0)
    x = np.linspace(0, np.size(y), np.size(y))
    
    fig = plt.figure()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot( 1, 1, 1 )
    ax.errorbar(x, y, yerr = np.sqrt(y), fmt='x', ecolor='green', label='Counts with counting statistics error')
    ax.set_xlabel('channels')
    ax.set_ylabel('counts')
    ax.legend(loc='best')
    
    fig.savefig(path2 + '.pdf')
    plt.show()

#
NumberOfFiles = np.size(CutoffPaths)

for l in range(NumberOfFiles):
    path = CutoffPaths[l]
    print('******************************************')
    print(path)
    yData = np.trim_zeros(np.loadtxt(path, skiprows=0))
    j = 0
    while yData[j] ==0:
        j+=1
    
    k = 1
    while yData[-k] ==0:
        k+=1
    
    yData = yData[5+j:-5-k]
    xData = velcalibrator(np.linspace(0, np.size(yData), np.size(yData)) + 5+j)
    yData = yData / max(yData)
    
    generalWidth = 0.02
    
    yDataLoc = yData
    startValues = [ max( yData ) ]
    counter = 0
    
    while max( yDataLoc ) - min( yDataLoc ) > .07:
        counter += 1
        print(counter)
        if counter > 6: ### max 10 peak...emergency break to avoid infinite loop
            break

        minP = np.argmin( yDataLoc )
        minY = yData[ minP ]
        x0 = xData[ minP ]
        startValues += [ x0, minY - max( yDataLoc ), generalWidth ]
        popt, ier = leastsq( res_multi_lorentz, startValues, args=( xData, yData ) )
        yDataLoc = [ y - multi_lorentz( x, popt ) for x,y in zip( xData, yData ) ]
    
    peaks = popt[1::3]
    print(popt)
    HWHM = np.zeros(np.size(peaks))
    for i in range(np.size(popt))[1::3]:
        HWHM[int((i-1)/3)] = hwhm(popt[i], popt[i+2])
        print('peak ', int((i-1)/3)+1,' ', popt[i], 'hwhm ', hwhm(popt[i], popt[i+2]))
    
    p = np.argsort(peaks)
    sortedPeaks = np.array(peaks)[p]
    sortedHWHM = np.array(HWHM)[p]
    print('Sorted peaks: ', sortedPeaks)
    print('Sorted HWHM: ', sortedHWHM)
    print(int(counter/2-2))

    
    testData = [ multi_lorentz(x, popt ) for x in xData ]
    
    fig = plt.figure()
    ax = fig.add_subplot( 1, 1, 1 )
    ax.plot( xData, yData )
    ax.plot( xData, testData )
    plt.xlabel('velocity [mm/s]')
    plt.ylabel('Normalized counts')
    
    fig.savefig(path + '.pdf')
    plt.show()
    