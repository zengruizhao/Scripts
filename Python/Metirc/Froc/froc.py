import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import euclidean

def computeConfMatElements(thresholded_proba_map, ground_truth, allowedDistance):
    
    if allowedDistance == 0 and type(ground_truth) == np.ndarray:
        P = np.count_nonzero(ground_truth)
        TP = np.count_nonzero(thresholded_proba_map*ground_truth)
        FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))
        N = ground_truth.size - P
        P_pre = np.count_nonzero(thresholded_proba_map)
    else:
    
        #reformat ground truth to a list  
        if type(ground_truth) == np.ndarray:
            #convert ground truth binary map to list of coordinates
            labels, num_features = ndimage.label(ground_truth)
            list_gt = ndimage.measurements.center_of_mass(ground_truth, labels, range(1,num_features+1))   
        elif type(ground_truth) == list:        
            list_gt = ground_truth        
        else:
            raise ValueError('ground_truth should be either of type list or ndarray and is of type ' + str(type(ground_truth)))
        
        #reformat thresholded_proba_map to a list
        labels, num_features = ndimage.label(thresholded_proba_map)
        list_proba_map = ndimage.measurements.center_of_mass(thresholded_proba_map, labels, range(1,num_features+1)) 
         
        #compute P, TP and FP  
        FP = 0
        TP = 0
        P = len(list_gt)
        #compute FP
        for point_pm in list_proba_map:
            found = False
            for point_gt in list_gt:                           
                if euclidean(point_pm,point_gt) < allowedDistance:
                    found = True
                    break
            if found == False:
                FP += 1
        #compute TP
        for point_gt in list_gt:
            for point_pm in list_proba_map:                           
                if euclidean(point_pm,point_gt) < allowedDistance:
                    TP += 1
                    break
                                 
    return P,TP,FP,N, P_pre
        
def computeFROC(proba_map, ground_truth, allowedDistance, nbr_of_thresholds=40, range_threshold=None):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #allowedDistance: Integer. euclidian distance distance in pixels to consider a detection as valid (anisotropy not considered in the implementation)  
    #nbr_of_thresholds: Interger. number of thresholds to compute to plot the FROC
    #range_threshold: list of 2 floats. Begining and end of the range of thresholds with which to plot the FROC  
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #FPavg_list_treshold: list of average FP over the set of images for increasing thresholds
    
    #verify that proba_map and ground_truth have the same shape
    if proba_map.shape != ground_truth.shape:
        raise ValueError('Error. Proba map and ground truth have different shapes.')
        
    #rescale ground truth and proba map between 0 and 1
    proba_map = proba_map.astype(np.float32)
    proba_map = (proba_map - np.min(proba_map)) / (np.max(proba_map) - np.min(proba_map))
    if type(ground_truth) == np.ndarray:
        ground_truth = ground_truth.astype(np.float32)    
        ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth))
    
    #define the thresholds
    if range_threshold == None:
        threshold_list = (np.linspace(np.min(proba_map),np.max(proba_map),nbr_of_thresholds)).tolist()
    else:
        threshold_list = (np.linspace(range_threshold[0],range_threshold[1],nbr_of_thresholds)).tolist()
    
    sensitivity_list_treshold = []
    FPavg_list_treshold = []
    precision_list_treshod = []
    #loop over thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        FP_list_proba_map = []
        precision_list_proba_map = []
        #loop over proba map
        for i in range(len(proba_map)):
                       
            #threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(proba_map[i]))
            thresholded_proba_map[proba_map[i] >= threshold] = 1
            
            #save proba maps
#            imageio.imwrite('thresholded_proba_map_'+str(threshold)+'.png', thresholded_proba_map)                   
                   
            #compute P, TP, and FP for this threshold and this proba map
            P,TP,FP, N, P_pre= computeConfMatElements(thresholded_proba_map, ground_truth[i], allowedDistance)
            
            #append results to list
            FP_list_proba_map.append(FP)##
            #check that ground truth contains at least one positive
            if (type(ground_truth) == np.ndarray and np.nonzero(ground_truth) > 0) or (type(ground_truth) == list and len(ground_truth) > 0):
                sensitivity_list_proba_map.append(TP*1./P)
                precision_list_proba_map.append(TP*1./P_pre)
            
        
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map))    
        precision_list_treshod.append(np.mean(precision_list_proba_map))
    sensitivity_list_treshold.append(0)
    precision_list_treshod.append(1)
    return sensitivity_list_treshold, FPavg_list_treshold, threshold_list, precision_list_treshod

def plotFROC(FPavg_list,x,threshold_list,save_path,write_thresholds, y):
    plt.figure()
    #x = (x - np.min(x)) / (np.max(x) - np.min(x))
    # x = 1-np.exp(-np.array(x))
    y = np.array(y)
    x = np.array(x)
    # print x,y
    # y = (y-x)/(1.-x)
    # x = -np.log(1.-x)
    # print x,y
    plt.plot(x, y, '-')##
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ## FAUC
    auc = 0
    for i in range(1, len(x)):
        auc = auc + (y[i] + y[i-1]) * (x[i-1] - x[i])/2
    #auc = auc + y[i]*x[i]/2
    plt.title('P-R curve, AUPR = ' + str(auc))
    #round thresholds
    threshold_list = [ '%.2f' % elem for elem in threshold_list ]
    
    #annotate thresholds
    if write_thresholds:
        xy_buffer = None
        for i, xy in enumerate(zip(x, y)):
            if xy != xy_buffer:                                    
                plt.annotate(str(threshold_list[i]), xy=xy, textcoords='data')
                xy_buffer = xy
    
    plt.savefig(save_path)
    print 'AUC = ', auc
    #plt.show()
    
def computeAndPlotFROC(proba_map,ground_truth, allowedDistance, nbr_of_thresholds, range_threshold, save_path, write_thresholds):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #allowedDistance: Integer. euclidian distance distance in pixels to consider a detection as valid (anisotropy not considered in the implementation)  
    #nbr_of_thresholds: Interger. number of thresholds to compute to plot the FROC
    #range_threshold: list of 2 floats. Begining and end of the range of thresholds with which to plot the FROC  
    #write_thresholds: Boolean. Write thresholds on the FROC plot 
    #save_path: string. Saving path of the FROC plot.
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #FPavg_list_treshold: list of average FP over the set of images for increasing thresholds
        
    sensitivity_list, FPavg_list, threshold_list, precision_list_treshod = computeFROC(proba_map,ground_truth, allowedDistance, nbr_of_thresholds, range_threshold)
    print 'computed FROC'
    plotFROC(FPavg_list,sensitivity_list,threshold_list,save_path, write_thresholds, precision_list_treshod)
    print 'plotted FROC'
    
    
    
    
    
