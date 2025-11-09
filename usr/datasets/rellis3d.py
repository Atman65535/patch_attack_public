import os.path as osp
import os

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS
from typing import Optional, Union, Dict, Sequence, List, Callable

"""
Dataset calss derived from BaseSegDataset
"""

#print("rellis3d registered")

@DATASETS.register_module()
class Rellis3DDataset(BaseSegDataset):
    # TODO finish the metainfo here
    METAINFO = dict(
        ## rellis difinition refer to label2color.ipynb
        color_palette={
            0: {"color": [0, 0, 0],  "name": "void"},
            1: {"color": [108, 64, 20],   "name": "dirt"},
            3: {"color": [0, 102, 0],   "name": "grass"},
            4: {"color": [0, 255, 0],  "name": "tree"},
            5: {"color": [0, 153, 153],  "name": "pole"},
            6: {"color": [0, 128, 255],  "name": "water"},
            7: {"color": [0, 0, 255],  "name": "sky"},
            8: {"color": [255, 255, 0],  "name": "vehicle"},
            9: {"color": [255, 0, 127],  "name": "object"},
            10: {"color": [64, 64, 64],  "name": "asphalt"},
            12: {"color": [255, 0, 0],  "name": "building"},
            15: {"color": [102, 0, 0],  "name": "log"},
            17: {"color": [204, 153, 255],  "name": "person"},
            18: {"color": [102, 0, 204],  "name": "fence"},
            19: {"color": [255, 153, 204],  "name": "bush"},
            23: {"color": [170, 170, 170],  "name": "concrete"},
            27: {"color": [41, 121, 255],  "name": "barrier"},
            31: {"color": [134, 255, 239],  "name": "puddle"},
            33: {"color": [99, 66, 34],  "name": "mud"},
            34: {"color": [110, 22, 138],  "name": "rubble"}},
        # refer to rellis3d.benchmarks.GSCNN-master.train.py
        label_mapping = {0: 0,
                     1: 0,
                     3: 1,
                     4: 2,
                     5: 3,
                     6: 4,
                     7: 5,
                     8: 6,
                     9: 7,
                     10: 8,
                     12: 9,
                     15: 10,
                     17: 11,
                     18: 12,
                     19: 13,
                     23: 14,
                     27: 15,
                     31: 16,
                     33: 17,
                     34: 18},
            ### here no use but easier initialize 

        # cityscapes definition.
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        )  # 19 classes
    
    # refer to mmseg.datasets.basesegdataset BaseSegDataset
    def __init__(self,
                 data_root: Optional[str] = "data/rellis3d",
                 img_suffix:str='',
                 seg_map_suffix:str='',
                 ann_file = 'train.lst',
                 data_prefix=dict(
                             img_suffix='',
                             seg_map_suffix=''
                 ),
                 **kwargs):
        kwargs.pop("data_prefix",None)
        
        super().__init__(data_root=data_root,
                         img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         ann_file=ann_file,
                         data_prefix=data_prefix,
                         **kwargs)
    
    @classmethod
    def get_label_map(cls,
                      new_classes:None = None
                      ) -> Union[Dict, None]:
        '''
        overwrite of base class method get_label_map
        the original one returns a list about label mapping
        here we don't use new_classes, but return a fixed list
        '''
        return cls.METAINFO.get('label_mapping')
    
    def _update_palette(self):
        # assert self._metainfo.get('palette', []) is [], \
        # "palette should be empty\ndon't put anything in metainfo !"
        return self.METAINFO.get('palette')
    
    def load_data_list(self) -> List[dict]:
        '''
        Load annotation from lst file 
        return a list of data info
        '''
        data_list = []
        img_dir = self.data_prefix.get('img_path', None) # should be empty here
        ann_dir = self.data_prefix.get('seg_map_path', None)

        assert img_dir == '' or img_dir == None, f"img_dir should be None ! Check your config !"
        assert ann_dir == '' or ann_dir == None, f"ann_dir should be None ! Check your config !"

        assert self.img_suffix == '' or self.img_suffix == None, "img_suffix should be None for the list file already given the full path !"
        assert self.seg_map_suffix == '' or self.seg_map_suffix == None, \
            "seg_map_suffix should be None for the list file already given the full path !"
                
        _, suffix = osp.splitext(self.ann_file)
        assert osp.isfile(self.ann_file) and suffix == ".lst", "R U sure that ann_file is a .lst file ?"
        
        img_dir = self.data_root
        ann_dir = self.data_root # here just for logic 

        with open(self.ann_file, 'r') as f_in:
            for line in f_in:
                img_name, ann_name = line.strip().split(' ')
                data_info = dict(
                    img_path = osp.join(img_dir, img_name), # obviously here we append img_name to root
                    seg_map_path = osp.join(ann_dir, ann_name)
                )
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label # whether to ignore label zero
                data_info['seg_fields'] = []
                data_list.append(data_info)

        return data_list