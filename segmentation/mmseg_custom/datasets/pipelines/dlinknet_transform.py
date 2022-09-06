import cv2
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomHueSaturationValue(object):

    def __init__(self,hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15),
                                    prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob
        
    def __call__(self, results):
        image = results['img']
        if np.random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.randint(self.hue_shift_limit[0], self.hue_shift_limit[1]+1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            #image = cv2.merge((s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        results['img'] = image
        return results


@PIPELINES.register_module()
class RandomShiftScaleRotate(object):

    def __init__(self,shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-0, 0),
                                        prob=0.5, borderMode=cv2.BORDER_CONSTANT):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.aspect_limit = aspect_limit
        self.rotate_limit = rotate_limit
        self.borderMode = borderMode
        self.prob = prob

    def __call__(self, results):
        image = results['img']
        if np.random.random() < self.prob:
            height, width, channel = image.shape

            angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])
            scale = np.random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1])
            aspect = np.random.uniform(1 + self.aspect_limit[0], 1 + self.aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * width)
            dy = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)

            

            results['img'] = cv2.warpPerspective(results['img'], mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=self.borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))

            for key in results.get('seg_fields', []):                        
                results[key] = cv2.warpPerspective(results[key], mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=self.borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))

            
        return results


@PIPELINES.register_module()
class RandomRotate90(object):
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, results):

        if np.random.random() < self.prob:
            results['img']=np.rot90(results['img'])
            for key in results.get('seg_fields', []):
                results[key] = np.rot90(results[key])

        return results

@PIPELINES.register_module()
class Normalize_Ann(object):

    def __init__(self) -> None:
        pass

    def __call__(self, results):
        
        for key in results.get('seg_fields', []):   
            pass
        
        return results