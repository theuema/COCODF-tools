import cv2
import copy

from pathlib import Path

'''
    :Plots ground truth object position projections of image: specified by `img_path`, generates plotted image and saves the result to `out`
    :Plots ground truth projections of image: specified by `img_path`, generates plotted image and directly shows the result if `show` is set `True`
'''
def image_plot_gtruth_projections(img_path, projections, colors, out=None, show: bool=False):
    if out is None and show is False:
        return
    
    # saving condition for results
    save_img = False if out is None else True
    if save_img: save_path = str(Path(out) / Path(img_path).name) 

    # print information
    print('Annotating image {} with number of projections: {}'.format(Path(img_path).name, len(projections)))
    
    # write results
    img = cv2.imread(img_path)
    id_plotted_img = copy.deepcopy(img)

    # plot gtruth projection
    for i, projection in enumerate(projections):
        X = projection['X']
        Y = projection['Y']
        ID = projection['ID']
        cv2.circle(id_plotted_img, (X, Y), radius=8, color=colors[i], thickness=-1)

        pos = (X, Y - 8)
        tl = 1 # line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        fs = tl / 2.5 # font scale
        cv2.putText(id_plotted_img, ID, pos, fontFace=0, fontScale=fs, color=colors[i], thickness=tf, lineType=cv2.LINE_AA) # place category ID next to projection

    if save_img:
        cv2.imwrite(save_path, id_plotted_img)
    
    if show: # show image stages
        cv2.imshow('Raw image', img)
        cv2.waitKey(0)
        cv2.imshow('Plotted image', id_plotted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return