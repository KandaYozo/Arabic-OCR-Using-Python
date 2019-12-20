import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_template
from scipy import stats

def ShowImageCV2(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ShowImagePlt(images,titles=None):
    try:
        n_ims = len(images)
        if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
        fig = plt.figure()
        n = 1
        for image,title in zip(images,titles):
            a = fig.add_subplot(1,n_ims,n)
            if image.ndim == 2: 
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
            n += 1
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
        plt.show()
    except:
        print('\nCheck square braces [ ]\n')

def correct_skew(thresh):
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
	    angle = -(90 + angle)
    else:
	    angle = -angle
    
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def threshold(image):
    thresholded = np.copy(image)
    _, thresholded = cv2.threshold(thresholded, 127, 255, cv2.THRESH_BINARY_INV)
    # thresholded = thresholded // 255
    return thresholded

def VerticalProjection(image):
    return (np.sum(image, axis = 0))//255

def HorizontalProjection(image):
    return (np.sum(image, axis = 1))//255

#Algo 1 Region####################################################################################################

def globalPeaks(HP):
    global_peaks = []
    global_peaks_indices = []
    region = []
    region_start = 0
    region_end = 0
    flag = 0
    regions_list = [] # 2D Array
    region_indices = []
    local_max = 0
    local_max_index = 0
    for index,value in enumerate(HP):
        # Ana 0 w gy mn 0
        if value == 0 and flag == 0:
            region = []
            region_start = 0
            region_end = 0

        # Ana msh 0 w gy mn 0 yb2a ana awel wa7ed
        elif value != 0 and flag == 0:
            flag = 1
            region_start = index
            region.append(value)
            if value > local_max:
                local_max = value
                local_max_index = index

        # Ana msh 0 w gy mn msh 0 wely ba3dy msh 0 yb2a ana fel nos
        elif value != 0 and HP[index+1] != 0:
            flag = 1
            region.append(value)
            if value > local_max:
                local_max = value
                local_max_index = index

        # Ana msh 0 w gy mn msh 0 wely ba3dy 0 yb2a ana fel a5er
        elif value != 0 and HP[index+1] == 0:
            region.append(value)
            region_end = index
            
            regions_list.append(region)
            region_indices.append((region_start,region_end))
            
            global_peaks.append(local_max)
            global_peaks_indices.append(local_max_index)
            
            flag = 0
            local_max = 0
            local_max_index = 0
            region = []
            region_end = 0
            region_start = 0
    return global_peaks, global_peaks_indices

def drawLines(TR, MPL):
    TR_copy = np.copy(TR)
    for i in MPL:
        TR_copy[i,:] = 255
    return TR_copy


def extractLines(TR):
    lines = []
    width = TR.shape[1]
    TR_copy = np.copy(TR)
    number_of_labels, labels, stats_per_region, centroids = cv2.connectedComponentsWithStats(TR_copy, 4, cv2.CV_32S)
    for i in range(1,number_of_labels):
        area = stats_per_region[i, cv2.CC_STAT_AREA]
        if area <= width:
            continue
        left = stats_per_region[i,cv2.CC_STAT_LEFT]
        top = stats_per_region[i,cv2.CC_STAT_TOP]
        width = stats_per_region[i,cv2.CC_STAT_WIDTH]
        height = stats_per_region[i,cv2.CC_STAT_HEIGHT] + 1 # + 1 to get dots
        lines.append(TR_copy[top:top+height,left:left+width])
        # cv2.rectangle(TR_copy, (left,top), (left+width, top+height), (255,255,255), 1)
        # ShowImagePlt([TR_copy])
    return lines
'''
# HABD
def remove_line(line):
    line_copy = np.copy(line)
    HP = HorizontalProjection(line_copy)
    MP = np.max(HP)
    BLI = np.where(HP == MP)[0][0]
    for i in range(line_copy.shape[1]):
        if line_copy[BLI-1,i] == 0 and line_copy[BLI+1,i] == 0:
            line_copy[BLI,i] = 0
# HABD DONE
'''

def line_segmentation(TR): # Algo 1
    HP    = []
    MP    = []
    MPL   = []
    TR_copy = np.copy(TR)
    lines_with_baseline = []
    lines_without_baseline = []
    regions_indices = []
    
    HP = HorizontalProjection(TR)
    MP, MPL = globalPeaks(HP)
    TR = drawLines(TR, MPL)
    lines_with_baseline = extractLines(TR)
    # HABD
    # lines_without_baseline = remove_line(lines_with_baseline[0])
    # DONE

    # Get original lines without horizontal line
    # TODO: El performance etfasha5
    for line in lines_with_baseline:
        result = match_template(TR_copy, line)
        ij = np.unravel_index(np.argmax(result), result.shape)
        left, top = ij[::-1]
        height, width = line.shape
        lines_without_baseline.append(TR_copy[top:top+height,left:left+width])
        # cv2.rectangle(TR_copy, (left,top), (left+width, top+height), (255,255,255), 1)
        # ShowImageCV2(TR_copy)

    '''
    Using region indices
    for i in regions_indices:
        lines_with_baseline.append(TR[i[0]:i[1],:])
    '''
    return lines_with_baseline, lines_without_baseline

#Algo 2 Region####################################################################################################

class SepReg():
    # lw msheet 3la gap_indices 7atal3 el gap nafso
    # lw msheet 3la min w max 7aral3 el SW
    minColumnIndex: int = 0
    maxColumnIndex: int = 0
    length: int = 0
    SW = []
    gap = []
    segmentType = ""
    gap_indices = []
    gap_length: int = 0
    penSize = 0
    def setPenSize(self):
        PS = 0
        VP = VerticalProjection(self.SW)
        print(VP)
        HP = HorizontalProjection(self.SW)
        if np.max(HP) > np.max(VP):
            PS = stats.mode(VP[VP > 0])[0][0]
            print(PS)
        else:
            PS = stats.mode(HP[HP > 0])[0][0]
        self.penSize = PS

def MainBinaryLineImage(line, oline):
    line_copy = np.copy(line)
    number_of_labels, labels, stats_per_region, centroids = cv2.connectedComponentsWithStats(line_copy, 4, cv2.CV_32S)
    sizes = stats_per_region[:,-1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2,number_of_labels):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    MBLI = np.zeros(labels.shape)
    MBLI[labels == max_label] = 255
    logical_and = np.logical_and(np.array(MBLI),np.array(oline))
    bitwise_and = np.zeros(logical_and.shape)
    bitwise_and[logical_and == False] = 0
    bitwise_and[logical_and == True] = 255
    MBLI = bitwise_and
    return MBLI

def mergeContinuousIndices(lst): 
    res = [[lst[0]]]
    for i in range(1, len(lst)):
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]])
    SR_array = []
    for i in res:
        SR = SepReg()
        SR.gap_indices = i
        SR_array.append(SR)
    return SR_array

def words_line_segmentation(DBLI, MBLI): # Algo 2
    VP = []
    SI = []
    WSL = []
    VP = VerticalProjection(MBLI)
    SI = (np.where(np.copy(VP) == 0))[0]
    SR = mergeContinuousIndices(SI)

    for i in range(0,len(SR)-1):
        SR[i].minColumnIndex = np.max(SR[i].gap_indices) + 1
        SR[i].maxColumnIndex = np.min(SR[i+1].gap_indices)
        SR[i].SW = MBLI[:,SR[i].minColumnIndex:SR[i].maxColumnIndex]
        SR[i].setPenSize()
        SR[i].gap_length = len(SR[i+1].gap_indices)

    for i in range(0,len(SR)-1):
        # TODO: Pen Size is strangely calculated in some cases
        # Ask about stats.mode if more than one value is the MFV
        if SR[i].gap_length > ((SR[i].penSize + SR[i+1].penSize)/2):
            SR[i].segmentType = "wordSegment"
            print("Word")
        else:
            SR[i].segmentType = "subWordSegment"
            print("SubWord")
            np.append(SR[i+1].SW, SR[i].SW, axis=1)
            diacriticsSegment = DBLI[:,np.min(SR[i].minColumnIndex):np.max(SR[i+1].maxColumnIndex)]
            bodySegment = MBLI[:,np.min(SR[i].minColumnIndex):np.max(SR[i+1].maxColumnIndex)]
            WSL.append(diacriticsSegment + bodySegment)
            ShowImagePlt([diacriticsSegment + bodySegment])
    return WSL

        # print(SR[i].length)
        # print(SR[i].gap_indices)
        # print(SR[i+1].gap_indices)

        # El foro2at fel gomla
        # SR[i].SW = MBLI[:,SR[i].gap_indices[0]:SR[i].gap_indices[-1]]
        # ShowImagePlt([SR[i].SW])


#Algo 3 Region####################################################################################################


# Main############################################################################################################

def main():
    image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    image = threshold(image)
    # ShowImageCV2(image)

    lines_with_baseline, lines_without_baseline = line_segmentation(image)
    # for i in range(len(lines_with_baseline)):
    #     ShowImagePlt([lines_with_baseline[i]])
    #     ShowImagePlt([lines_without_baseline[i]])

    # for i in lines_with_baseline:
    #     ShowImagePlt([i])
    line  = lines_with_baseline[0]
    oline = lines_without_baseline[0]
    # ShowImagePlt([line])
    MBLI = MainBinaryLineImage(line,oline)
    DBLI = oline - MBLI
    WSL = words_line_segmentation(DBLI, MBLI)

    # for i in WSL:
    #     ShowImagePlt([i])
    #     _,thresh = cv2.threshold(i, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    #     rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,5))
    #     dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
        # # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # # dilation = cv2.dilate(i, rect_kernel, iterations = 1)
        # edges = cv2.Canny(i, 0, 1)
        # h,w = i.shape
        # vis = np.zeros((h,w), np.float32)
        # i2 = cv2.CreateMat(h, w, cv2.CV_32SC1)
        # i2 = cv2.fromarray(i2)
        # cv2.CvtColor(vis, i2, cv2.CV_GRAY2BGR)
        # edges = cv2.Canny(vis,127,255)
        # # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # # dilation = cv2.dilate(i, rect_kernel, iterations = 1)
        # # im, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # cv2.drawContours(im, contours, -1, (0,255,0), 3)
        # ShowImagePlt([dilation])

main()

'''
Above Baseline
test = line_copy[0:BLI,:]
'''

# def remove_dots():
#     test = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
#     test = np.array(test)
#     test = threshold(test)
#     opening = cv2.morphologyEx(test, cv2.MORPH_OPEN, (0.5,0.5))
#     ShowImagePlt([test,opening])

# remove_dots()