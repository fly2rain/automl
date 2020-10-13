# coding=utf-8
"""
# Oct 2, 2019
# written by Feiyun Zhu (i.e., Jerry ZHU) @ Sam's Club Innovative Team
#
# Function:　
#          1. Summarize all the useful codes that are commonly used..
#       (*)2. Find a place to refresh the latest codes.
"""
from __future__ import division
import os
import cv2
import sys
import random
from tqdm import tqdm
import numpy as np
from math import ceil
import multiprocessing as mp
import matplotlib.colors as mcolors


def add_current_package_path(module_path: str=""):
    if not module_path:
        module_path = os.path.dirname(os.path.abspath(__file__))
    else:
        if not os.path.exists(module_path):
            print(f"--- Module_path: {module_path}, does not exist! ------------")
            return
        module_path = os.path.expanduser(module_path)
    sys.path.append(module_path)


################################################################################
# The global constant variables @ Nov 11, 2019
################################################################################
class CFG:
    # Global vars: Interested_types
    INTERESTED_CFG_TYPES = (".cfg",)
    INTERESTED_TXT_TYPES = (".txt",)
    INTERESTED_IMG_TYPES = (".jpg", ".jpeg", ".png", ".bmp")
    INTERESTED_VID_TYPES = (".mp4", ".avi", ".mkv", ".mov")

    STRANGE_SYMBOLS = ("!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "+", ":",
                       "<", ">", " ", "")  # , "\uf02200\uf02200")

    # Global vars: colors
    BLUE = (255, 144, 30)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    PINK = (255, 51, 255)
    WHITE = (255, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    ORANGE = (0, 128, 255)

    # Global vars: Bases
    BASE_WIDTH = 1000
    BASE_HEIGHT = 1000

    BASE_SPACING_1 = 10
    BASE_SPACING_2 = 30
    BASE_SPACING_3 = 50
    BASE_SPACING_4 = 80

    # Small eps
    EPS = 1e-20


# ## Multiple thread
# class RunMultipleProcess:
#     def __init__(self, pool_size: int=0,  ):
#
#
#     def
#
#     def run_multiple_thread():
#         if self.multi_thread:
#             pool = ThreadPool()
#             pool.map(self.convert_one_list_image(), batch_images)
#         else:
#             for idx, batch in tqdm(enumerate(batch_images)):
#                 print(f"------------ convert {idx}-batch -----------------")
#                 self.convert_one_list_image(batch)


################################################################################
# Get the file lists @ Nov 11, 2019
################################################################################
class GetFileLists:
    def __init__(self, src: str, interested_type="image", del_hidden: bool = False):
        self.os_proc = OsProcess()
        self.src = self.os_proc.standardize_given_path(src)
        self.del_hidden = ConvertFormats.str_2_bool(del_hidden)

        if isinstance(interested_type, str):
            self.interested_types = self.get_interested_fileTypes(interested_type)
        elif isinstance(interested_type, (tuple, list)):
            self.interested_types = interested_type
        else:
            raise ValueError(f"interested_type: {interested_type}, should only be str, tuple or list!")

    def get_file_list(self) -> list:
        """ Get file list in the given path. """
        list_files = []
        if self.del_hidden:  # delete the hidden files in the folder.
            self.os_proc.delete_hidden_files(self.src)

        for root, _, file_names in os.walk(self.src):
            for file_name in file_names:
                _, extension = os.path.splitext(file_name)
                if extension.lower() not in self.interested_types:
                    continue

                path = os.path.join(root, file_name)
                list_files.append(path)
        return list_files

    def get_file_list_del_empty_files(self) -> list:
        """ Delete the empty files in the src and get the file list"""
        rst = []
        list_files = self.get_file_list()
        for file_name in list_files:
            if os.stat(file_name).st_size == 0:
                os.remove(file_name)
            else:
                rst.append(file_name)
        return rst

    @classmethod
    def get_interested_fileTypes(cls, fileType="image") -> tuple:
        """ choose interested file Types """
        if "." in fileType:  # means that fileType is a type of format, like ".jpg" or ".txt" etc.
            return fileType,
        elif fileType == "image":
            return CFG.INTERESTED_IMG_TYPES
        elif fileType == "txt":
            return CFG.INTERESTED_TXT_TYPES
        elif fileType == "video":
            return CFG.INTERESTED_VID_TYPES
        elif fileType == "cfg":
            return CFG.INTERESTED_CFG_TYPES
        else:
            raise ValueError(f"fileType: {fileType}, should be '.txt', '.jpg' or 'image', 'txt', 'video', 'cfg'\n")


################################################################################
# Convert the formats into the required formats @ Nov 11, 2019
################################################################################
class ConvertFormats:
    @classmethod
    def str_2_bool(cls, v: (str, bool)) -> bool:
        """ Convert str to bool, i.e., yes or no.
        Args:
            v: the input string.
        Return:
            yes, true, t, y, 1 ---> True
            no, false, f, n, 0, " ", None ---> No
        """
        if isinstance(v, bool):
            return v

        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', " ", "None"):
            return False

    @classmethod
    def batch_2_float(cls, xx):
        if isinstance(xx, (tuple, list)):
            return [float(i) for i in xx]
        else:
            return float(xx)

    @classmethod
    def str_2_tuple_len(cls, s: str, LEN: int, sep: str = "$") -> tuple:
        if not s: return ()

        out = cls.str_2_tuple(s, sep)
        if len(out) != LEN:
            raise ValueError(f'The number of inputs != LEN (i.e., {LEN})')
        return out

    @classmethod
    def str_2_tuple(cls, s: str, sep: str = "$") -> tuple:
        """ convert str to a list of int values. This is to process WH and xyxy """
        if not s:
            return ()
        elif isinstance(s, tuple):
            return s
        elif isinstance(s, list):
            return tuple(s)
        elif not isinstance(s, str):
            return ()

        sep = sep if sep in s else " "
        out = s.strip().split(sep)
        out = [int(x.strip()) for x in out]
        return tuple(out)


################################################################################
# The basic os Process @ Nov 11, 2019
################################################################################
class OsProcess(object):
    @classmethod
    def crapped_file_or_not(cls, f: str) -> bool:
        # To find the given file is crapped or not.
        # Two cases: #1) the file is not exist,  #2) the file's size is 0
        rst = not f or not os.path.exists(f) or os.stat(f).st_size == 0
        return rst

    @classmethod
    def delete_hidden_files(cls, src: str) -> None:
        """ delete the hidden files in the folder.
        :param src: the input src folder.
        :return: None
        """
        cls.delete_certainPattern_files(src, pattern=".*")

    @classmethod
    def delete_certainPattern_files(cls, src: str, pattern: str = ".*") -> None:
        """ Delete the files with certain pattern in the folder.
        :param src: the input src folder
        :param pattern: the pattern in the filename
        :return: None
        """
        if not os.path.exists(src):
            print(f"Path doesn't exit: ({src})\n")
            return

        commandline = f"find {src} -name '{pattern}' -delete"
        os.system(commandline)

    @classmethod
    def check_make_dir(cls, path: str) -> None:
        """ To check the give path exists or not. If not, create the path.
        Args:
            path: the given path to verify and to create.
        """
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # if the last element is os.sep, remove it. We want standard path input
    @classmethod
    def standardize_given_path(cls, path: str) -> str:
        """ To standardize the given path. Some given path is like "tmp/a/b/" or
            "tmp/a/b", this function is to standardize the given path, i.e, convert
            "tmp/a/b/" to "tmp/a/b".
        Args:
            path: the input given path.
        """
        # return os.path.dirname(path)
        if not path: return ""

        path = path.strip()
        path = os.path.expanduser(path)
        idx = path.rfind(os.sep)
        if idx + 1 == len(path) or not path[idx + 1:]:
            return path[:idx]
        else:
            return path

    @classmethod
    def standardize_strange_symbol_4_filename(cls, src: str, strange_symbols: tuple = CFG.STRANGE_SYMBOLS) -> str:
        dst = src
        for x in strange_symbols:
            dst = dst.replace(x, "_")
        return dst

    @classmethod
    def standardize_strange_symbol_4_filename_mkdir_rename(cls, filename: str,
                                                           strange_symbols: tuple = CFG.STRANGE_SYMBOLS) -> str:
        # create the new filename for the given filename
        out_name = cls.standardize_strange_symbol_4_filename(filename, strange_symbols)

        # get the path for the filename and mkdir for it if it does not exist.
        tmp_path = os.path.dirname(out_name)
        cls.check_make_dir(tmp_path)

        # rename the filename
        os.rename(filename, out_name)
        return out_name

    @classmethod
    def remove_ext_4_filename(cls, filename: str) -> str:
        # # same ss the function below "get_basename_without_ext"
        # if not filename: return ""
        # while filename.find(".") > -1:
        #     filename, _ = os.path.splitext(filename)
        # return filename
        return cls.get_basename_without_ext(filename)

    @classmethod
    def get_basename_without_ext(cls, filename: str) -> str:
        # same ss the function above "get_basename_without_ext"
        if not filename: return ""

        while "." in filename:
            filename, _ = os.path.splitext(filename)
        return filename

    @classmethod
    def change_ext_4_filename(cls, filename: str, extension: str = "") -> str:
        basename = cls.get_basename_without_ext(filename)
        return basename + extension

    @classmethod
    def get_outName_4_givenFile_fullPath_only(cls, filename: str, src: str, dst: str, extension: str = "") -> str:
        """ Given the file_path, generate the corresponding output files with the full_path fashion.
        Args:
              1. filename: the input filename, including the full directory.
              2. src: the source directory.
              3. dst: the destination directory.'
              4. extension: the new extensions, e.g., ".jpg". If extension is None, then use the
                 original extension. Otherwise, use the provided extension name.
        """
        if not filename: return ""

        # 1. replace the src in the filename with dst.
        out_name = filename.replace(src, dst, 1)
        if extension:  # check extension is empty or not
            out_name = cls.change_ext_4_filename(out_name, extension)
        return out_name

    @classmethod
    def get_outName_4_givenFile_fullPath(cls, filename: str, src: str,
                                         dst: str, extension: str = ""):
        if not filename: return ""

        # 1. get the out_name for a given file
        out_name = cls.get_outName_4_givenFile_fullPath_only(filename, src, dst, extension)

        # 2. to check the out_path exists or not
        out_path = os.path.dirname(out_name)
        cls.check_make_dir(out_path)
        # idx = out_name.rfind(os.sep)
        # if idx > -1:
        #     tmp_path = out_name[:idx]
        #     cls.check_make_dir(tmp_path)
        return out_name

    @classmethod
    def get_outName_4_givenFile_fullPath_mkdir(cls, filename: str, src: str,
                                               dst: str, extension: str = ""):
        """ Same as cls.get_outName_4_givenFile_fullPath and just rewrite the code. """
        return cls.get_outName_4_givenFile_fullPath(filename, src, dst, extension)

    @classmethod
    def get_outName_4_givenFile_noFullPath(cls, filename: str, dst: str, extension: str = "") -> str:
        """ Given the file_path, generate the corresponding output files with the full_path fashion.
        Args:
            1. filename: the input filename, including the full directory.
            2. src: the source directory.
            3. dst: the destination directory.'
            4. extension: the new extensions, e.g., ".jpg". If extension is None, then use the
               original extension. Otherwise, use the provided extension name.
        """
        if not filename: return ""

        out_name = os.path.basename(filename)
        if extension:
            out_name = cls.change_ext_4_filename(out_name, extension)
        out_name = os.path.join(dst, out_name)
        return out_name

    @classmethod
    def get_outName_4_givenFile_noFullPath_mkdir(cls, filename: str, dst: str, extension: str = "") -> str:
        cls.check_make_dir(dst)
        return cls.get_outName_4_givenFile_noFullPath(filename, dst, extension)

    @classmethod
    def get_outName_4_givenFile(cls, filename: str, src: str, dst: str,
                                full_path: bool = True, extension: str = "") -> str:
        """ Get the out_name 4 the given file """
        if full_path:
            out_name = cls.get_outName_4_givenFile_fullPath_only(filename, src, dst, extension)
        else:
            out_name = cls.get_outName_4_givenFile_noFullPath(filename, dst, extension)
        return out_name

    @classmethod
    def get_outName_4_givenFile_mkdir(cls, filename: str, src: str, dst: str,
                                      full_path: bool = True, extension: str = "") -> str:
        """ Get the out_name 4 the given file and mkdir the out directory """
        if full_path:
            out_name = cls.get_outName_4_givenFile_fullPath_mkdir(filename, src, dst, extension)
        else:
            out_name = cls.get_outName_4_givenFile_noFullPath_mkdir(filename, dst, extension)
        return out_name

    @classmethod
    def get_outName_4_givenFile_standardize(cls, filename: str, src: str, dst: str,
                                            full_path: bool = True, extension: str = "") -> str:
        """ Get the out_name 4 the given file  and standardize it and mkdir. """
        if full_path:
            out_name = cls.get_outName_4_givenFile_fullPath_only(filename, src, dst, extension)
        else:
            out_name = cls.get_outName_4_givenFile_noFullPath(filename, dst, extension)
        out_name = cls.standardize_strange_symbol_4_filename(out_name)
        return out_name

    @classmethod
    def get_outName_4_givenFile_standardize_mkdir(cls, filename: str, src: str, dst: str,
                                                  full_path: bool = True, extension: str = "") -> str:
        """ Get the out_name 4 the given file  and standardize it and mkdir. """
        out_name = cls.get_outName_4_givenFile_standardize(filename, src, dst, full_path, extension)
        out_path = os.path.dirname(out_name)
        cls.check_make_dir(out_path)
        return out_name

    @classmethod
    def get_filename_2_compare(cls, path: str, src: str, full_path: bool = True, with_ext: bool = True):
        if not path: return ""

        out_path = path
        if not with_ext:
            out_path = cls.get_basename_without_ext(out_path)

        if full_path:
            out_path = cls.get_outName_4_givenFile_fullPath_only(out_path, src, "")
        else:
            out_path = cls.get_outName_4_givenFile_noFullPath(out_path, "")
        return out_path


################################################################################
# The basic process on lists @ Nov 11, 2019
################################################################################
class ListProcess:
    @classmethod
    def randomly_select_n_unq_elements_4_list(cls, nums, n):
        if len(nums) < n + 1:
            return nums

        out = random.sample(nums, n)
        return sorted(out)

    @classmethod
    def split_list_2_lists(cls, list_files: list, batch_size: int = 1024) -> list:
        """ Split the list of images into multiple batches """
        # set random seed and sort the image list
        np.random.seed(0)
        list_files.sort()

        # Corner case for the batch lists ---------------------
        num_img = len(list_files)
        if num_img < batch_size:
            return [list_files]

        # set batch_size and cut the image idx to batches -----
        bt_size = int(num_img / 32)
        batch_size = batch_size if batch_size < bt_size else bt_size
        bi = np.floor(np.arange(num_img) / batch_size).astype(np.int)

        # set the image idx to batches
        nb = bi[-1] + 1
        out_batch = [[] for _ in range(nb)]
        for i in range(num_img):
            idx = bi[i]
            out_batch[idx].append(list_files[i])
        return out_batch


################################################################################
# Basic Image Visualizing Class
################################################################################
class BBoxVisualize:
    def __init__(self):
        pass

    @classmethod
    def get_color_lists(cls, num_color: int = 10):
        color_base = dict(mcolors.BASE_COLORS)
        color_base.pop('k', None)
        color_base = [[int(x) for x in np.array(mcolors.to_rgba(color)[:3]) * 255] for name, color in
                      color_base.items()]

        color_tab = dict(mcolors.TABLEAU_COLORS)
        color_tab.pop("k", None)
        color_tab = [[int(x) for x in np.array(mcolors.to_rgba(color)[:3]) * 255] for name, color in color_tab.items()]

        colors = color_base + color_tab
        while len(colors) < num_color:
            colors += colors
        return colors

    @classmethod
    def apply_box(cls, image, bbox, color, thickness=2, ratio=1):
        thickness *= ceil(ratio)
        # y1, x1, y2, x2 = bbox
        x1, y1, x2, y2 = bbox

        # rgb_color = tuple([int(c * 255) for c in color])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image

    @classmethod
    def apply_line(cls, image, start_point, end_point, color, thickness=1, ratio=1):
        thickness *= ceil(ratio)
        cv2.line(image, start_point, end_point, color=color, thickness=thickness)
        return image

    @classmethod
    def apply_text_left(cls, image, bbox, color, text, font_scale=0.5, thickness=1, ratio=1):
        rgb_color = tuple([int(c * 255) for c in color])
        font_scale *= ratio
        thickness *= ceil(ratio)
        if len(bbox) == 4:
            # y1, x1, y2, x2 = bbox
            x1, y1, x2, y2 = bbox
            cv2.putText(image, '{}'.format(text), (x1 + 5, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_TRIPLEX, font_scale, rgb_color, thickness)
        if len(bbox) == 2:
            x, y = bbox
            cv2.putText(image, '{}'.format(text), (x, y), cv2.FONT_HERSHEY_TRIPLEX,
                        font_scale, color, thickness)
        return image

    @classmethod
    def apply_text_upper_left(cls, image, bbox, color, text, font_scale=0.5,
                              thickness=1, ratio=1):
        font_scale *= ratio
        thickness *= ceil(ratio)
        # text_wind_size = lambda tex: cv2.getTextSize(tex, cv2.FONT_HERSHEY_DUPLEX, fontScale, thickness)[0]
        text_length, text_height = cls.text_size(text)
        xs, ys, _, _ = bbox
        # ys, xs, _, _ = bbox
        # xs = xs + 5
        ys = ys - text_height - 5
        ye, xe = ys + text_height * ceil(ratio), xs + text_length * ceil(ratio)
        cv2.rectangle(image, (xs, ys), (xe, ye + 5), color=color, thickness=-1)
        cv2.putText(image, text, (xs, ye), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0),
                    thickness)
        return image

    @classmethod
    def apply_text_center(cls, image, y, x, color, text, font_scale=0.5, thickness=1, ratio=1):
        font_scale *= ratio
        thickness *= ceil(ratio)
        w, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, fontScale=font_scale,
                               thickness=thickness)[0]
        ys = y
        _, width_img = image.shape[:2]
        xs = width_img // 2 - w // 2
        cv2.putText(image, "{}".format(text), (xs, ys), cv2.FONT_HERSHEY_TRIPLEX, font_scale,
                    color, thickness)
        return image

    @classmethod
    def apply_contour(cls, image, mask, bbox, color, thickness=3, ratio=1):
        thickness *= ceil(ratio)
        # new_mask = np.where(mask, 255, 0).astype(np.uint8)
        # y1, x1, y2, x2 = bbox
        x1, y1, x2, y2 = bbox
        new_mask = -mask[y1:y2, x1:x2].astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(new_mask, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_TC89_KCOS)
        cv2.drawContours(image[y1:y2, x1:x2], contours, -1, color, thickness)
        return image

    @classmethod
    def text_size(cls, text, font_scale=0.5, thickness=1, ratio=1):
        font_scale *= ratio
        thickness *= ceil(ratio)
        width, height = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, fontScale=font_scale,
                                        thickness=thickness)[0]
        return width, height


################################################################################
# Basic Image Processing Class
################################################################################
class BasicImgProcess(object):
    @classmethod
    def crop_roi(cls, frame, xyxy_int: tuple):
        if not xyxy_int:
            return frame
        elif not isinstance(xyxy_int, (tuple, list)):
            raise ValueError("xyxy ROI is not a (tuple or list)")
        elif len(xyxy_int) != 4:
            raise ValueError("xyxy ROI label does not have 4 values, i.e., (x1,y1,x2,y2)")

        x1, y1, x2, y2 = xyxy_int
        height, width, _ = frame.shape
        # process x1, y1, x2, y2
        x1 = min(max(x1, 0), width)
        x2 = min(max(x2, 0), width)
        y1 = min(max(y1, 0), height)
        y2 = min(max(y2, 0), height)

        # crop frame and return the result.
        return frame[y1:y2, x1:x2]

    @classmethod
    def extract_pixel_feature(cls, frame, height=90, width=120):
        """ Extract pixel level feature for the given frame. """
        feature = cv2.resize(frame, (width, height))
        feature = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
        return feature

    @classmethod
    def estimate_sharp(cls, img):
        """ Estimate the sharp score for the given image. """
        # if not img: None
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharp_map = cv2.Laplacian(img, cv2.CV_32F)
        score = np.var(sharp_map)
        return score


################################################################################
# Yolo based classes
################################################################################
class BBoxLabel(object):
    def __init__(self):
        pass

    @classmethod
    def put_into_range(cls, num, low_t=0.0, high_t=1.0):
        if isinstance(num, (float, int)):
            return min(max(num, low_t), high_t)
        elif isinstance(num, (tuple, list)):
            return [min(max(low_t, x), high_t) for x in num]
        else:
            raise ValueError(f"The input {num} is not float, int, tuple or list.\n")

    @classmethod
    def xywh_2_xyxy(cls, x: float, y: float, w: float, h: float) -> tuple:
        x, y, w, h = cls.put_into_range((x, y, w, h))
        w_2, h_2 = w / 2.0, h / 2.0

        x1, x2 = x - w_2, x + w_2
        y1, y2 = y - h_2, y + h_2

        x1, y1, x2, y2 = cls.put_into_range((x1, y1, x2, y2))
        return x1, y1, x2, y2

    @classmethod
    def xyxy_2_xywh(cls, x1: float, y1: float, x2: float, y2: float) -> tuple:
        x1, y1, x2, y2 = cls.put_into_range((x1, y1, x2, y2))

        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2.0, y1 + h / 2.0
        return x, y, w, h

    # @classmethod
    # def put_into_range_xyxy(cls, bbox_xyxy, low_t=0.0, high_t=1.0):
    #     return cls.put_into_range(bbox_xyxy, low_t, high_t)
    @classmethod
    def put_into_range_xywh(cls, bbox_xywh):
        bbox_xyxy = cls.xywh_2_xyxy(*bbox_xywh)
        bbox_xywh = cls.xyxy_2_xywh(*bbox_xyxy)
        return bbox_xywh


class YoloLabel(BBoxLabel):
    __slots__ = ["x", "y", "w", "h", "xywh", "xyxy"]

    def __init__(self, x: float, y: float, w: float, h: float):
        super().__init__()
        self.x, self.y, self.w, self.h = self.put_into_range_xywh((x, y, w, h))
        self.xywh = x, y, w, h
        self.xyxy = self.xywh_2_xyxy(*self.xywh)  # ((x1,y1), (x2,y2))

    def get_obj_area(self):
        return self.w * self.h

    def get_wh_ratio(self):
        ratio_1 = self.w / (self.h + CFG.EPS)
        ratio_2 = self.h / (self.w + CFG.EPS)
        return min(ratio_1, ratio_2)

    def get_standard_bbox_xywh(self):
        return self.xywh

    def get_standard_bbox_xyxy(self):
        return self.xyxy


################################################################################
# Yolo based classes with txt and images
################################################################################
class YoloLabelImage(YoloLabel):
    __slots__ = ["x", "y", "w", "h", "xywh", "xyxy"]

    def __init__(self, x: float, y: float, w: float, h: float, img=None):
        super().__init__(x, y, w, h)
        self.img = img
        if self.img is not None:
            self.height, self.width, _ = self.img.shape
        else:
            self.height = self.width = None

    def xywh_2_xyxy_int(self):
        if self.height is None or self.width is None:
            raise ValueError(f"Warning None value for height or width: \n\t\t-the height is {self.height}",
                             f"\n\t\t-the width is {self.width}")

        x1, y1, x2, y2 = self.get_standard_bbox_xyxy()

        x1, x2 = round(x1 * self.width), round(x2 * self.width)
        y1, y2 = round(y1 * self.height), round(y2 * self.height)
        return x1, y1, x2, y2

    def cropped_obj_with_padding(self, padding=10):
        """ Crop logo from item image according to given xy corner coordinates
        Args:
            cropped logo image
        """
        if not self.height or not self.width or self.img is None:
            raise ValueError("Warning None value for img or height or width: \
                \n\t\t-the input img is {},\
                \n\t\t-the height is {},\
                \n\t\t-the width is {}".format(self.img, self.height, self.width))

        x1, y1, x2, y2 = self.xywh_2_xyxy_int()

        x1 = x1 if x1 - padding < 0 else x1 - padding
        y1 = y1 if y1 - padding < 0 else y1 - padding
        x2 = x2 if x2 + padding > self.width else x2 + padding
        y2 = y2 if y2 + padding > self.height else y2 + padding

        cropped_img = self.img[y1:y2, x1:x2]
        cropped_bbox = (x1, y1, x2, y2)
        return cropped_img, cropped_bbox


if __name__ == "__main__":
    pass
    # 1. Testing the BasicFileProcess class
    # fprocess = BasicFileProcess()
    # xx = fprocess.get_outName_4_givenFile_fullPath("a/b/c/efg.txt", "a/b", 'c/d', ".jpg")
    # yy = fprocess.get_outName_4_givenFile_noFullPath("a/b/c/efg.txt", "a/b", 'c/d', ".jpg")
    # print(xx)
    # print(yy)

    # txt_name = ["123/abc.mp4.mp4", "xyz.a.b.c", "123.jpg.png"]
    # # out_name = fprocess.get_outName_4_givenFile_fullPath(txt_name, "123", "dfg", extension=".jpg")
    # # out_name = fprocess.remove_ext_4_filename(txt_name)
    # for x in txt_name:
    #     y = fprocess.get_basename_without_ext(x)
    #     z = fprocess.remove_ext_4_filename(x)
    #     print(x, y, z)

    # xx = range(100)
    # yy = fprocess.randomly_select_n_unq_elements_4_list(xx, 5)
    # print(yy)
    # fprocess.delete_certainPattern_files("abc")
