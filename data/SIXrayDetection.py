import torch.utils.data as data
import os.path as osp
import cv2
import numpy as np
import torch

MY_CLASSES = (  # always index 0
    'core', 'coreless')


class SIXrayAnnotationTransform(object):

    def __init__(self) -> None:
        self.class_to_ind = dict(
            zip(MY_CLASSES, range(len(MY_CLASSES))))

    def __call__(self, class_index, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target:
            obj_info = obj.split(" ")
            if obj_info[1] == "不带电芯充电宝":
                type_ = "coreless"
            else:
                type_ = "core"

            pts = [2, 3, 4, 5]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(obj_info[pt])
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[type_]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class SIXrayDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 images_set_file=None,
                 transform=None,
                 target_transform=SIXrayAnnotationTransform(),
                 ):
        self.root = root
        self.images_set_file = images_set_file
        self.transform = transform
        self.target_transform = target_transform
        self.name = "SIXRay"
        self.ids = list()
        # VOC_ROOT = osp.join("./", "data/SIXRay/")
        class_dirs = [osp.join(self.root, "core_3000"), osp.join(self.root, "coreless_3000")]
        for path in class_dirs:
            anno = osp.join(path, 'Annotation')
            import os
            files = os.listdir(anno)
            for filename in files:
                filename = filename
                self.ids.append(filename.replace(".txt", ""))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        # target = ET.parse(self._annopath % img_id).getroot()
        type_ = "core"
        if img_id.startswith("coreless"):
            type_ = "coreless"
        img = cv2.imread(osp.join(self.root, type_ + "_3000", 'Image', "%s.jpg" % img_id))
        target = open(osp.join(self.root, type_ + "_3000", 'Annotation', "%s.txt" % img_id),
                      encoding="utf-8").readlines()

        height, width, channels = img.shape

        target = self.target_transform(img_id, target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        type_ = "core"
        if img_id.startswith("coreless"):
            type_ = "coreless"
        return cv2.imread(osp.join(self.root, type_ + "_3000", 'Image', "%s.jpg" % img_id))


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        type_ = "core"
        if img_id.startswith("coreless"):
            type_ = "coreless"
        target = open(osp.join(self.root, type_ + "_3000", 'Annotation', "%s.txt" % img_id),
                      encoding="utf-8").readlines()

        gt = self.target_transform(img_id, target, 1, 1)
        return img_id, gt


if __name__ == "__main__":
    a = SIXrayDetection(osp.join("C:\\Work\\Git\\ssd.pytorch", "data/SIXRay/"))
