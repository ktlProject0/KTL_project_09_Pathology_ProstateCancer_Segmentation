import model_torch
import modules_torch
import _utils_torch
from openslide import OpenSlide
from _utils_torch import *
import loss

def preprocessing(img_list):

    img = []
    mask= []
    
    for idx in tqdm(range(len(img_list))):
        img_slide = OpenSlide(img_list[idx])
        
        try:
            mask_dir = img_list[idx].replace('img', 'label').replace('.tiff', '_mask.tiff')
            mask_slide = OpenSlide(mask_dir)
            mask_img = sitk.ReadImage(mask_dir)
            mask_img = sitk.GetArrayFromImage(mask_img)
        except:
            continue
        
        if mask_img.max() < 3:
            continue
        
        print("ImageFile name:", img_list[idx])
        print("Mask File name:", mask_dir)
        print("Slide dimensions:", img_slide.dimensions)
        print("Slide level count:", img_slide.level_count)
        print("Slide level dimensions:", img_slide.level_dimensions)
        print("Slide level downsamples:", img_slide.level_downsamples)
        print('\n')
        
        size = 512
        for i in range(10):
            x, y = np.random.randint(0, img_slide.level_dimensions[0][0]-size), np.random.randint(0, img_slide.level_dimensions[0][1]-size)
            img_patch = img_slide.read_region((x,y), 0, (size, size))
            img_patch = img_patch.convert("RGB")
            img_patch = np.array(img_patch).astype(np.float32)
            
            if img_patch.mean() > 230:
                continue
    
            img_patch -= img_patch.min()
            img_patch /= img_patch.max()    
            
            mask_patch = mask_slide.read_region((x,y), 0, (size, size))
            mask_patch = np.array(mask_patch)[...,0]
            mask_patch[mask_patch<3] = 0
            mask_patch[mask_patch>2] = 1
            
            img.append(img_patch)
            mask.append(mask_patch)
            
        img_slide.close()
        mask_slide.close()
    
    img = np.array(img)
    mask = np.array(mask)

    X_train = img[:int(np.round(len(img)*0.7)),...]
    Y_train = mask[:int(np.round(len(img)*0.7)),...]
    
    X_test = img[int(np.round(len(img)*0.7)):,...]
    Y_test = mask[int(np.round(len(img)*0.7)):,...]
    
    return X_train, Y_train, X_test, Y_test