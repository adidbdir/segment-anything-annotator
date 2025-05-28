import sys
import functools
import cv2
import glob
import os
import os.path as osp
import imgviz
import html
import json
import math
import time
import argparse
import numpy as np
import tempfile
import torch
import base64
import csv
import uuid

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, QScrollArea, QDockWidget, QMessageBox, QLineEdit, QCheckBox, QSpinBox, QDateEdit
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.Qt import QSize
from qtpy.QtCore import Qt
from qtpy import QtCore
from qtpy import QtGui, QtWidgets
from canvas import Canvas
import utils
from utils.download_model import download_model

from labelme.widgets import ToolBar, UniqueLabelQListWidget, LabelDialog, LabelListWidget, LabelListWidgetItem, ZoomWidget
from labelme import PY2
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError


from shape import Shape

from PIL import Image

from collections import namedtuple
Click = namedtuple('Click', ['is_positive', 'coords'])

# from segment_anything import sam_model_registry, SamPredictor
# sys.path.append('../sam2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'external', 'sam2')))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import csv_exporter
from icecream import ic

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.crystallization_analysis import CrystallizationAnalysis

LABEL_COLORMAP = imgviz.label_colormap()

class MainWindow(QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(self, parent=None, global_w=1000, global_h=1800, model_type='vit_b', keep_input_size=True, max_size=1080, category_file='primary.txt',
                 save_mask=True, save_bbox=True, save_labels=True, image_directory=None):
        super(MainWindow, self).__init__(parent)
        self.resize(global_w, global_h)
        self.model_type = model_type
        self.keep_input_size = keep_input_size
        self.max_size = float(max_size)
        self.category_file = category_file
        self.global_w = global_w
        self.global_h = global_h

        # スケールバー関連の変数を初期化
        self.scale_mode = False  # スケールバー測定モード
        self.scale_points = []  # スケールバーの2点の座標
        self.scale_value = 0.0  # スケールの実際の長さ
        self.scale_unit = "μm"  # スケールの単位
        self.scale_factor = 1.0  # ピクセルから実際の長さへの変換係数
        self.scalebar_image = None  # スケールバー画像へのパス
        self.scale_set = False  # スケールが設定されたかどうか

        self.setWindowTitle('segment-anything-annotator')
        self.canvas = Canvas(self,
            epsilon=10.0,
            double_click='close',
            num_backups=10,
            app=self,
        )

        
        self._noSelectionSlot = False
        self.current_output_dir = 'output'
        os.makedirs(self.current_output_dir, exist_ok=True)
        self.current_output_filename = ''
        self.canvas.zoomRequest.connect(self.zoomRequest)

        self.memory_shapes = []
        self.image = []
        self.image_np = []
        self.masks = []
        self.sam_mask = []
        self.sam_mask_proposal = []
        self.image_encoded_flag = False
        self.min_point_dis = 4

        self.predictor = None

        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: self.scrollArea.verticalScrollBar(),
            Qt.Horizontal: self.scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. "
                "Press 'Esc' to deselect."
            )
        )
        self.labelDialog = LabelDialog(
            parent=self,
            labels=[],
            sort_labels=False,
            show_text_field=True,
            completion='contains',
            fit_to_content={'column': True, 'row': False},
        )

        self.labelList = LabelListWidget()
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)

        self.shape_dock = QDockWidget(
            self.tr("Polygon Labels"), self
        )
        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)

        self.category_list = [i.strip() for i in open('categories.txt', 'r', encoding='utf-8').readlines()]
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self.category_list,
            sort_labels=False,
            show_text_field=True,
            completion='contains',
            fit_to_content={'column': True, 'row': False},
        )
        self.zoom_values = {}
        self.video_directory = ''
        self.video_list = []
        self.video_len = len(self.video_list)

        self.img_list = []
        self.img_len = len(self.img_list)
        self.current_img_index = 0
        self.current_img = ''
        self.current_img_data = ''

        self.button_next = QPushButton('Next Image', self)
        self.button_next.clicked.connect(self.clickButtonNext)
        self.button_last = QPushButton('Last Image', self)
        self.button_last.clicked.connect(self.clickButtonLast)

        self.img_progress_bar = QProgressBar(self)
        self.img_progress_bar.setMinimum(0)
        self.img_progress_bar.setMaximum(1)
        self.img_progress_bar.setValue(0)
        self.button_proposal1 = QPushButton('Proposal1', self)
        self.button_proposal1.clicked.connect(self.choose_proposal1)
        self.button_proposal1.setShortcut('1')
        self.button_proposal2 = QPushButton('Proposal2', self)
        self.button_proposal2.clicked.connect(self.choose_proposal2)
        self.button_proposal2.setShortcut('2')
        self.button_proposal3 = QPushButton('Proposal3', self)
        self.button_proposal3.clicked.connect(self.choose_proposal3)
        self.button_proposal3.setShortcut('3')
        self.button_proposal4 = QPushButton('Proposal4', self)
        self.button_proposal4.clicked.connect(self.choose_proposal4)
        self.button_proposal4.setShortcut('4')
        self.button_proposal_list = [self.button_proposal1, self.button_proposal2, self.button_proposal3, self.button_proposal4]
        
        self.class_on_flag = True
        self.class_on_text = QLabel("Class On", self)
        
        # secondary のグループ分けを行う
        self.grouping_complete = False  # グループ分けフラグを追加
        self.grouped_segments = []  # グループ分けされたセグメントを保持するリスト
        self.group_id = 1
        self.default_label = None  # default_label を初期化

        # time
        self.s = time.time()

        #naive layout
        self.scrollArea.move(int(0.02 * global_w), int(0.08 * global_h))
        self.scrollArea.resize(int(0.75 * global_w), int(0.7 * global_h))
        self.shape_dock.move(int(0.79 * global_w), int(0.08 * global_h))
        self.shape_dock.resize(int(0.2 * global_w), int(0.7 * global_h))
        self.button_next.move(int(0.18 * global_w), int(0.85 * global_h))
        self.button_next.resize(int(0.1 * global_w),int(0.04 * global_h))
        self.button_last.move(int(0.01 * global_w), int(0.85 * global_h))
        self.button_last.resize(int(0.1 * global_w),int(0.04 * global_h))
        self.class_on_text.move(int(0.01 * global_w), int(0.9 * global_h))
        self.img_progress_bar.move(int(0.01 * global_w), int(0.8 * global_h))
        self.img_progress_bar.resize(int(0.3 * global_w),int(0.04 * global_h))
        
        self.button_proposal1.resize(int(0.17 * global_w),int(0.14 * global_h))
        self.button_proposal1.move(int(0.33 * global_w), int(0.8 * global_h))
        self.button_proposal2.resize(int(0.17 * global_w),int(0.14 * global_h))
        self.button_proposal2.move(int(0.50 * global_w), int(0.8 * global_h))
        self.button_proposal3.resize(int(0.17 * global_w),int(0.14 * global_h))
        self.button_proposal3.move(int(0.67 * global_w), int(0.8 * global_h))
        self.button_proposal4.resize(int(0.17 * global_w),int(0.14 * global_h))
        self.button_proposal4.move(int(0.84 * global_w), int(0.8 * global_h))
        
        # add:隠しでパラメータ保持用の QLineEdit を作成（初期値は空）
        self.date_edit = QtWidgets.QLineEdit()
        self.experimenter_edit = QtWidgets.QLineEdit()
        self.impurity_type_edit = QtWidgets.QLineEdit()
        self.impurity_conc_edit = QtWidgets.QLineEdit()
        self.seed_size_edit = QtWidgets.QLineEdit()
        self.crystal_time_edit = QtWidgets.QLineEdit()
        self.suspension_density_edit = QtWidgets.QLineEdit()
        self.image_scaler_edit = QtWidgets.QLineEdit()
        
        self.zoomWidget = ZoomWidget()
        self.image_directory = image_directory

        action = functools.partial(utils.newAction, self)
        
        experimentParamsAction = action(
            self.tr("Experiment Params"),
            self.showExperimentParamsDialog,
            None,
            "objects",
            self.tr("Set Experiment Parameters"),
            enabled=True,
        )
        GroupSeg = action(
            self.tr("Group Seg"),
            lambda: self.clickGroupSeg(),
            'g',
            "objects",
            self.tr("Group Seg"),
            enabled=True,
        )
        categoryFile = action(
            self.tr("Category File"),
            lambda: self.clickCategoryChoose(),
            'c',
            "objects",
            self.tr("Category File"),
            enabled=True,
        )
        imageDirectory = action(
            self.tr("Image Directory"),
            lambda: self.clickFileChoose(),
            'None',
            "objects",
            self.tr("Image Directory"),
            enabled=True,
        )
        LoadSAM = action(
            self.tr("Load SAM"),
            lambda: self.clickLoadSAM(),
            'None',
            "objects",
            self.tr("Load SAM"),
            enabled=True,
        )
        AutoSeg = action(
            self.tr("AutoSeg"),
            lambda: self.clickAutoSeg(),
            'None',
            "objects",
            self.tr("AutoSeg"),
            enabled=False,
        )
        promptSeg = action(
            self.tr("Accept"),
            lambda: self.addSamMask(),
            'a',
            "objects",
            self.tr("Accept"),
            enabled=False,
        )

        saveDirectory = action(
            self.tr("Save Directory"),
            lambda: self.clickSaveChoose(),
            'None',
            "objects",
            self.tr("Save Directory"),
            enabled=True,
        )

        createMode = action(
            self.tr("Manual Polygons"),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            'Ctrl+W',
            "objects",
            self.tr("Start drawing polygons"),
            enabled=True,
        )
        createPointMode = action(
            self.tr("Point Prompt"),
            lambda: self.toggleDrawMode(False, createMode="point"),
            'p',
            "objects",
            self.tr("Point Prompt"),
            enabled=True,
        )
        createRectangleMode = action(
            self.tr("Box Prompt"),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            'b',
            "objects",
            self.tr("Box Prompt"),
            enabled=True,
        )
        cleanPrompt = action(
            self.tr("Reject"),
            lambda: self.cleanPrompt(),
            'r',
            "objects",
            self.tr("Reject"),
            enabled=True,
        )
        
        self.switchClass = action(
            self.tr("Class On/Off"),
            lambda: self.clickSwitchClass(),
            'none',
            "objects",
            self.tr("Class On/Off"),
            enabled=True,
        )

        editMode = action(
            self.tr("Edit Polygons"),
            self.setEditMode,
            'e',
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            'ALT+s',
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=True,
        )

        undoLastPoint = action(
            self.tr("Undo last point"),
            self.canvas.undoLastPoint,
            'U',
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )

        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo"),
            self.undoShapeEdit,
            'Ctrl+U',
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )

        save = action(
            self.tr("&Save"),
            self.saveFile,
            'S',
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShape,
            'd',
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            'None',
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        reduce_point = action(
            self.tr("Reduce Points"),
            self.reducePoint,
            'None',
            "copy",
            self.tr("Reduce Points"),
            enabled=True,
        )            
        edit = action(
            self.tr("&Edit Label"),
            self.editLabel,
            'None',
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )
        

        self.actions = utils.struct(
            categoryFile=categoryFile,
            imageDirectory=imageDirectory,
            saveDirectory=saveDirectory,
            switchClass=self.switchClass,
            loadSAM=LoadSAM,
            #autoSeg=AutoSeg,
            promptSeg=promptSeg,
            cleanPrompt=cleanPrompt,
            createMode=createMode,
            createPointMode=createPointMode,
            createRectangleMode=createRectangleMode,
            editMode=editMode,
            undoLastPoint=undoLastPoint,
            undo=undo,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            reduce_point=reduce_point,
            save=save,
            onShapesPresent=(saveAs, hideAll, showAll),
            menu=(
                createMode,
                editMode,
                undoLastPoint,
                undo,
                save,
            )
        )

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        self.toolbar = self.addToolBar('Tool')
        self.toolbar.addAction(experimentParamsAction)
        self.toolbar.addAction(categoryFile)
        self.toolbar.addAction(imageDirectory)
        self.toolbar.addAction(saveDirectory)
        self.toolbar.addAction(self.switchClass)
        self.toolbar.addAction(LoadSAM)
        #self.toolbar.addAction(AutoSeg)
        self.toolbar.addAction(promptSeg)
        self.toolbar.addAction(cleanPrompt)
        self.toolbar.addAction(createMode)
        self.toolbar.addAction(createPointMode)
        self.toolbar.addAction(createRectangleMode)
        self.toolbar.addAction(editMode)
        self.toolbar.addAction(undoLastPoint)
        self.toolbar.addAction(undo)
        self.toolbar.addAction(delete)
        self.toolbar.addAction(edit)
        self.toolbar.addAction(duplicate)
        self.toolbar.addAction(reduce_point)
        self.toolbar.addAction(GroupSeg)
        self.toolbar.addAction(save)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} from the canvas."
                )
            ).format(
                #utils.fmtShortcut(
                #    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                #),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(True)

        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.canvas.actions = self.actions
    
        # 初期化時にカテゴリファイルを読み込む
        # 一つのラベルしかない場合は自動的にそのラベルを設定
        # acceptを押した際に自動でラベルが設定されるように
        # SAMの読みこみを実行する
        # 複数の場合はNoneに設定   
        if self.category_file and os.path.exists(self.category_file):
            try:
                with open(self.category_file, 'r') as f:
                    data = f.readlines()
                    self.category_list = [i.strip() for i in data]
                    self.category_list.sort()
                    if len(self.category_list) == 1:
                        self.default_label = self.category_list[0]  # 自動的にそのラベルを設定
                        self.class_on_flag = False
                        # print(f"カテゴリファイルを読み込みました: {self.category_file}")
                        self.clickLoadSAM()
                    else:
                        self.default_label = None  # 複数の場合はNoneに設定
            except Exception as e:
                print(f"カテゴリファイルの読み込みに失敗しました: {e}")
        
        # 保存設定の保持
        self.save_mask = save_mask
        self.save_bbox = save_bbox
        self.save_labels = save_labels
        
        # 保存設定用チェックボックス
        self.save_mask_checkbox = QtWidgets.QCheckBox(self.tr("Save Mask Images"), self)
        self.save_mask_checkbox.setChecked(self.save_mask)
        self.save_mask_checkbox.stateChanged.connect(self.updateSaveMaskSetting)
        
        self.save_bbox_checkbox = QtWidgets.QCheckBox(self.tr("Save BBox Images"), self)
        self.save_bbox_checkbox.setChecked(self.save_bbox)
        self.save_bbox_checkbox.stateChanged.connect(self.updateSaveBBoxSetting)
        
        # チェックボックスの配置
        self.save_mask_checkbox.move(int(0.01 * global_w), int(0.95 * global_h))
        self.save_bbox_checkbox.move(int(0.20 * global_w), int(0.95 * global_h))
        
        # スケールバー測定関連のUIコンポーネント
        self.scale_button = QPushButton('スケールバー測定', self)
        self.scale_button.clicked.connect(self.startScaleBarMode)
        self.scale_button.move(int(0.40 * global_w), int(0.95 * global_h))
        self.scale_button.resize(int(0.15 * global_w), int(0.03 * global_h))
        
        self.scale_status_label = QLabel("スケール: 未設定", self)
        self.scale_status_label.move(int(0.56 * global_w), int(0.95 * global_h))
        self.scale_status_label.resize(int(0.2 * global_w), int(0.03 * global_h))
        
        # スケールバー画像を自動的に探して表示するボタン
        self.find_scalebar_button = QPushButton('スケールバー画像を検索', self)
        self.find_scalebar_button.clicked.connect(self.findScaleBarImage)
        self.find_scalebar_button.move(int(0.77 * global_w), int(0.95 * global_h))
        self.find_scalebar_button.resize(int(0.2 * global_w), int(0.03 * global_h))
    
        # Add area threshold UI
        self.area_threshold = 100  # Default value
        self.area_threshold_label = QtWidgets.QLabel(self.tr("Area Threshold:"), self)
        self.area_threshold_spinbox = QtWidgets.QSpinBox(self)
        self.area_threshold_spinbox.setMinimum(0)
        self.area_threshold_spinbox.setMaximum(1000000) # Adjust max as needed
        self.area_threshold_spinbox.setValue(self.area_threshold)
        self.area_threshold_spinbox.setSingleStep(10)
        self.area_threshold_spinbox.valueChanged.connect(self.update_area_threshold)

        # Layout for area threshold (example placement, adjust as needed)
        # Assuming you want to place it near other settings like save checkboxes
        self.area_threshold_label.move(int(0.01 * global_w), int(0.92 * global_h)) # Adjust position
        self.area_threshold_spinbox.move(int(0.12 * global_w), int(0.92 * global_h)) # Adjust position
        self.area_threshold_spinbox.resize(int(0.07 * global_w), int(0.025 * global_h)) # Adjust size

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFile(self, _value=False):
        # assert not self.image.isNull(), "cannot save empty image"
        # if self.labelFile:
        #     # DL20180323 - overwrite when in directory
        #     self._saveFile(self.labelFile.filename)
        # elif self.output_file:
        #     self._saveFile(self.output_file)
        #     self.close()
        # else:
        #     self._saveFile(self.saveFileDialog())
        # self._saveFile(self.saveFileDialog())
        # print(self.current_output_filename)
        self._saveFile(self.current_output_filename)

    def _saveFile(self, filename):
        
        if self.save_labels:
            self.saveLabels(filename)
        # マスクとバウンディングボックス画像の保存
        if self.save_mask or self.save_bbox:
            filename_base = os.path.splitext(filename)[
                0
            ]  # 拡張子を除いたファイル名
            self.saveMaskAndBBoxImages(filename_base)

            self.setClean()

    def updateSaveMaskSetting(self, state):
        self.save_mask = (state == Qt.Checked)
        
    def updateSaveBBoxSetting(self, state):
        self.save_bbox = (state == Qt.Checked)

    def saveLabels(self, filename):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[[p.x(), p.y()] for p in s.points],
                    group_id=s.group_id,
                    description="",
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        imageData = base64.b64encode(self.current_img_data).decode("utf-8")
        save_data = {
            "version": "1.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": self.current_img,
            # "imageData": imageData,
            "imageHeight": self.raw_h,
            "imageWidth": self.raw_w,
        }

        with open(filename, 'w') as f:
            json.dump(save_data, f)
        return True

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)

    def saveFileDialog(self):
        caption = self.tr("Choose File")
        filters = self.tr("Label files")
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = os.path.basename(self.current_img)[:-4]
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def currentPath(self):
        #return osp.dirname(str(self.filename)) if self.filename else "."
        return "."

    def loadAnno(self, filename):
        with open(filename,'r') as f:
            data = json.load(f)
        for shape in data['shapes']:
            label = shape["label"]
            try:
                ttt = int(label)
                label = self.category_list[ttt]
            except:
                pass

            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            if not points:
                # skip point-empty shape
                continue
            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                flags=flags
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()
            self.addLabel(shape)
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    def clickButtonNext(self):
        # e = time.time()
        if self.actions.save.isEnabled():
            self.saveFile()
        if not self.grouping_complete:  # グループ分けが完了していない場合
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please complete grouping before moving to the next image."))
            return
        if self.current_img_index < self.img_len - 1:
            self.current_img_index += 1
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()

    def clickButtonLast(self):
        if self.actions.save.isEnabled():
            self.saveFile()
        if self.current_img_index > 0:
            self.current_img_index -= 1
            self.current_img = self.img_list[self.current_img_index]
            self.loadImg()


    def choose_proposal1(self):
        if len(self.sam_mask_proposal) > 0:
            self.sam_mask = self.sam_mask_proposal[0]
            self.canvas.setHiding()
            self.canvas.update()

    def choose_proposal2(self):
        if len(self.sam_mask_proposal) > 1:
            self.sam_mask = self.sam_mask_proposal[1]
            self.canvas.setHiding()
            self.canvas.update()
            
    def choose_proposal3(self):
        if len(self.sam_mask_proposal) > 2:
            self.sam_mask = self.sam_mask_proposal[2]
            self.canvas.setHiding()
            self.canvas.update()
            
    def choose_proposal4(self):
        if len(self.sam_mask_proposal) > 3:
            self.sam_mask = self.sam_mask_proposal[3]
            self.canvas.setHiding()
            self.canvas.update()    
            
    def loadImg(self):
        self.image = Image.open(self.current_img)
        self.image_np = np.array(self.image.convert("RGB"))
        self.raw_h, self.raw_w = cv2.imread(self.current_img).shape[:2]
        pixmap = QPixmap(self.current_img)
        self.canvas.loadPixmap(pixmap)
        self.img_progress_bar.setValue(self.current_img_index)

        # 自動的にズームレベルを調整して画像全体を表示
        self.adjustZoomToFitImage()

        img_name = os.path.basename(self.current_img)[:-4]
        self.current_output_filename = osp.join(self.current_output_dir, img_name + '.json')
        self.labelList.clear()
        if os.path.isfile(self.current_output_filename):
            self.loadAnno(self.current_output_filename)
        self.image_encoded_flag = False
        self.current_img_data = LabelFile.load_image_file(self.current_img)

    def adjustZoomToFitImage(self):
        """画像のサイズに表示を合わせるよう適切なズームレベルを設定する"""
            
        # スクロールエリアの表示可能サイズを取得
        view_width = self.scrollArea.width() - 2  # スクロールバーの幅を考慮
        view_height = self.scrollArea.height() - 2
        
        # 画像の実際のサイズを取得
        img_width = self.canvas.pixmap.width()
        img_height = self.canvas.pixmap.height()
        
        # 縦横比を維持しながら、画面内に収まるズーム値を計算
        width_ratio = float(view_width) / img_width
        height_ratio = float(view_height) / img_height
        
        # 小さい方の比率を使用して、画像全体が表示エリアに収まるようにする
        zoom_factor = min(width_ratio, height_ratio) * 100  # zoomWidgetは100倍の値を使用
        
        # 計算したズーム値を適用
        self.zoomWidget.setValue(int(zoom_factor))
        self.setZoom(int(zoom_factor))
        
        # 画像を中央に配置
        self.paintCanvas()

    def clickFileChoose(self):
        if self.image_directory is not None:
            directory = self.image_directory
        else:
            directory = QFileDialog.getExistingDirectory(self, 'choose target fold','.')
        if directory == '':
            return
        #self.img_list = glob.glob(directory + '/*.{jpg,png,JPG,PNG}')
        self.img_list = glob.glob(directory + '/*.jpg') + glob.glob(directory + '/*.png')
        self.img_list.sort()
        self.img_len = len(self.img_list)
        if self.img_len == 0:
            return
        self.current_img_index = 0
        self.current_img = self.img_list[self.current_img_index]
        self.img_progress_bar.setMinimum(0)
        self.img_progress_bar.setMaximum(self.img_len-1)
        
        self.loadImg()
        # ディレクトリ選択後、スケールが未設定ならスケールバー設定処理を開始
        if not self.scale_set:
            # スケールバー画像を検索して表示
            found = self.findScaleBarImage()
            if found:
                # スケールバー測定モードを開始
                self.startScaleBarMode()


    def clickSaveChoose(self):
        directory = QFileDialog.getExistingDirectory(self, 'choose target fold','.')
        if directory == '':
            return
        else:
            self.current_output_dir = directory
            os.makedirs(self.current_output_dir, exist_ok=True)
            self.loadImg()
            return directory


    def clickSwitchClass(self):
        if self.class_on_flag:
            self.class_on_flag = False
            self.class_on_text.setText('Class Off')
        else:
            self.class_on_flag = True
            self.class_on_text.setText('Class On')

    # スケールバー測定関連の関数
    def startScaleBarMode(self):
        """スケールバー測定モードを開始する"""
        # すでにスケールバー画像が読み込まれているか確認
        if self.scalebar_image is None:
            # スケールバー画像を自動検索
            self.findScaleBarImage()
            if self.scalebar_image is None:
                # スケールバー画像が見つからなかった場合、メッセージボックスを表示
                QMessageBox.warning(self, "警告", "スケールバー画像が見つかりませんでした。\n"
                                  "画像フォルダに'scalebar'または'scale'を含む名前の画像を配置してください。")
                return

        # スケールバー測定モードに入る
        self.scale_mode = True
        self.scale_points = []  # ポイントをリセット
        self.canvas.set_scale_mode(True)  # キャンバスにスケールモードを通知

        # ユーザーに指示を表示
        # QMessageBox.information(self, "スケールバー測定", "スケールバーの開始点と終了点を順番にクリックしてください。\n"
        #                        "スケールバーの実際の長さを入力するダイアログが表示されます。")

    def findScaleBarImage(self):
        """フォルダ内からスケールバー画像を探して読み込む"""
        if not self.img_list:
            QMessageBox.warning(self, "警告", "画像フォルダが選択されていません。\n"
                              "まず画像フォルダを選択してください。")
            return

        # フォルダ内のすべての画像をチェック
        img_dir = os.path.dirname(self.img_list[0])
        scalebar_candidates = []
        
        for img_path in glob.glob(img_dir + '/*.jpg') + glob.glob(img_dir + '/*.png'):
            img_filename = os.path.basename(img_path).lower()
            # 「スケールバー」または「scalebar」、「scale」を含むファイル名を探す
            if 'scalebar' in img_filename or 'scale' in img_filename or 'スケール' in img_filename:
                scalebar_candidates.append(img_path)
        
        if scalebar_candidates:
            # 最初に見つかったスケールバー画像を使用
            self.scalebar_image = scalebar_candidates[0]
            # スケールバー画像を読み込んで表示
            self.current_img = self.scalebar_image
            self.loadImg()
            # QMessageBox.information(self, "スケールバー画像", f"スケールバー画像を読み込みました: {os.path.basename(self.scalebar_image)}")
            return True
        else:
            self.scalebar_image = None
            return False

    def setScaleValue(self):
        """スケールバーの実際の長さを設定するダイアログを表示"""
        if len(self.scale_points) != 2:
            QMessageBox.warning(self, "警告", "スケールバーの2点が指定されていません。")
            return
        
        # 単位選択用のコンボボックス付きのダイアログを作成
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("スケール設定")
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # 値入力用のウィジェット
        form_layout = QtWidgets.QFormLayout()
        scale_value_edit = QtWidgets.QLineEdit()
        scale_value_edit.setValidator(QtGui.QDoubleValidator(0, 1000000, 5))  # 正の浮動小数点数のみ
        
        # 単位選択用のコンボボックス
        unit_combo = QtWidgets.QComboBox()
        unit_combo.addItems(["mm", "μm", "nm", "cm", "m"])
        unit_combo.setCurrentText("μm") # 初期値を "μm" に設定
        
        form_layout.addRow("測定値:", scale_value_edit)
        form_layout.addRow("単位:", unit_combo)
        layout.addLayout(form_layout)
        
        # ボタン
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # ダイアログを表示
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            try:
                scale_value = float(scale_value_edit.text())
                scale_unit = unit_combo.currentText()
                
                # 2点間の距離（ピクセル）を計算
                p1, p2 = self.scale_points
                pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                # スケール係数を計算（実際の長さ/ピクセル数）
                self.scale_factor = scale_value / pixel_distance
                self.scale_value = scale_value
                self.scale_unit = scale_unit
                self.scale_set = True
                
                # ステータス表示を更新
                self.scale_status_label.setText(f"スケール: {self.scale_factor:.6f} {scale_unit}/px")
                
                # スケールモードを終了
                self.scale_mode = False
                self.canvas.set_scale_mode(False)
                
                # 強制的に描画を更新し、マスクが表示されるようにする
                self.canvas.update()
                QtWidgets.QApplication.processEvents()  # イベントループを処理
                
                # 最初の画像に戻る（存在する場合）
                if self.img_len > 0:
                    self.current_img_index = 0
                    self.current_img = self.img_list[self.current_img_index]
                    self.loadImg()
                
            except ValueError:
                QMessageBox.warning(self, "エラー", "有効な数値を入力してください。")
                # Reset scale points and related canvas state
                self.scale_points = []
                if hasattr(self.canvas, 'scale_points'):
                    self.canvas.scale_points = [] # Clear points in canvas as well
                self.canvas.update() # Force repaint to clear the drawn points/line
        else: # Dialog was cancelled or closed
            # Reset scale points and related canvas state
            self.scale_points = []
            if hasattr(self.canvas, 'scale_points'):
                self.canvas.scale_points = []
            self.canvas.update() # Force repaint to clear the drawn points/line

    def showExperimentParamsDialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Experiment Parameters"))
        form_layout = QtWidgets.QFormLayout(dialog)
        
        # ダイアログ用のローカルな入力欄を作成し，既存の隠し QLineEdit の内容で初期化
        date_edit = QtWidgets.QDateEdit()
        date_edit.setDisplayFormat("yyyy/MM/dd")
        date_edit.setCalendarPopup(True)  # カレンダーポップアップを有効化
        date_edit.setDate(QtCore.QDate.currentDate())  # 現在の日付を初期値として設定
        
        # 既存の日付が設定されている場合は、それを表示
        if self.date_edit.text():
            try:
                # 既存の日付文字列を解析して設定
                existing_date = QtCore.QDate.fromString(self.date_edit.text(), "yyMMdd")
                if existing_date.isValid():
                    date_edit.setDate(existing_date)
            except:
                pass
        
        experimenter_edit = QtWidgets.QLineEdit(self.experimenter_edit.text())
        impurity_type_edit = QtWidgets.QLineEdit(self.impurity_type_edit.text())
        impurity_conc_edit = QtWidgets.QLineEdit(self.impurity_conc_edit.text())
        seed_size_edit = QtWidgets.QLineEdit(self.seed_size_edit.text())
        crystal_time_edit = QtWidgets.QLineEdit(self.crystal_time_edit.text())
        suspension_density_edit = QtWidgets.QLineEdit(self.suspension_density_edit.text())
        image_scaler_edit = QtWidgets.QLineEdit(self.image_scaler_edit.text())
        
        # スケール情報表示フィールドを追加
        scale_info = f"{self.scale_factor:.6f} {self.scale_unit}/px" if self.scale_set else "未設定"
        scale_info_label = QtWidgets.QLabel(scale_info)
        scale_info_label.setStyleSheet("font-weight: bold;")
        
        form_layout.addRow(self.tr("日付:"), date_edit)
        form_layout.addRow(self.tr("実験者:"), experimenter_edit)
        form_layout.addRow(self.tr("夾雑イオン種類:"), impurity_type_edit)
        form_layout.addRow(self.tr("夾雑イオン濃度（比）:"), impurity_conc_edit)
        form_layout.addRow(self.tr("種晶の大きさ:"), seed_size_edit)
        form_layout.addRow(self.tr("晶析時間:"), crystal_time_edit)
        form_layout.addRow(self.tr("懸濁密度:"), suspension_density_edit)
        # form_layout.addRow(self.tr("画像のスケーラー:"), image_scaler_edit)
        # form_layout.addRow(self.tr("スケール設定:"), scale_info_label)
        
        # OK/Cancel ボタン
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form_layout.addRow(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # 日付を指定された形式（yyMMdd）で保存
            formatted_date = date_edit.date().toString("yyMMdd")
            self.date_edit.setText(formatted_date)
            self.experimenter_edit.setText(experimenter_edit.text())
            self.impurity_type_edit.setText(impurity_type_edit.text())
            self.impurity_conc_edit.setText(impurity_conc_edit.text())
            self.seed_size_edit.setText(seed_size_edit.text())
            self.crystal_time_edit.setText(crystal_time_edit.text())
            self.suspension_density_edit.setText(suspension_density_edit.text())
            self.image_scaler_edit.setText(image_scaler_edit.text())

    def clickCategoryChoose(self):
        filename, _ = QFileDialog.getOpenFileName(self, "choose target file", ".")
        try:
            with open(filename, "r") as f:
                data = f.readlines()
                self.category_list = [i.strip() for i in data]
                self.category_list.sort()
                self.labelDialog = LabelDialog(
                    parent=self,
                    labels=self.category_list,
                    sort_labels=False,
                    show_text_field=True,
                    completion="contains",
                    fit_to_content={"column": True, "row": False},
                )
        except Exception as e:
            pass

    def clickLoadSAM(self):
        # download_model(self.model_type)
        self.sam = build_sam2(config_file='configs/sam2.1/sam2.1_hiera_l.yaml', ckpt_path='external/sam2/checkpoints/sam2.1_hiera_large.pt')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam.to(device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam)
        self.actions.loadSAM.setEnabled(False)
        #self.actions.autoSeg.setEnabled(True)
        self.actions.promptSeg.setEnabled(True)
    
    def clickAutoSeg(self):
        pass
    
    def getMaxId(self):
        max_id = -1
        for label in self.labelList:
            if label.shape().group_id != None:
                max_id = max(max_id, int(label.shape().group_id))
        return max_id
        
    def show_proposals(self, masks=None, flag=1):
        if flag != 1:
            img = cv2.imread(self.current_img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get button dimensions once, assuming all proposal buttons are the same size
            # If they can be different, this should be inside the loop
            if not self.button_proposal_list:
                return
            button_width = self.button_proposal_list[0].width()
            button_height = self.button_proposal_list[0].height()

            for msk_idx in range(masks.shape[0]): # Iterate through each mask proposal
                if msk_idx >= len(self.button_proposal_list): # Don't exceed available buttons
                    break

                current_button = self.button_proposal_list[msk_idx]
                tmp_mask_slice = masks[msk_idx] # This is a single mask (H, W)
                tmp_vis_for_button = img.copy()

                # Apply mask overlay to tmp_vis_for_button
                # Ensure tmp_mask_slice is boolean
                bool_mask_slice = tmp_mask_slice.astype(bool)

                # Overlay requires the mask to be broadcastable to the image's shape if image is 3-channel
                # and mask is 2D. We stack the boolean mask to 3 channels.
                if tmp_vis_for_button.ndim == 3 and bool_mask_slice.ndim == 2:
                    if tmp_vis_for_button.shape[:2] == bool_mask_slice.shape:
                        bool_mask_expanded = np.stack([bool_mask_slice]*3, axis=-1)
                        tmp_vis_for_button = np.where(
                            bool_mask_expanded,
                            0.5 * tmp_vis_for_button + 0.5 * np.array([30,30,220]),
                            tmp_vis_for_button
                        ).astype(np.uint8)
                    # else: print(f"Shape mismatch for overlay: img {tmp_vis_for_button.shape[:2]} mask {bool_mask_slice.shape}")
                # else: print(f"Vis/Mask ndim issue: vis {tmp_vis_for_button.ndim}, mask {bool_mask_slice.ndim}")

                # Resize the visualized image to fit the button, maintaining aspect ratio
                img_h, img_w = tmp_vis_for_button.shape[:2]
                if img_h == 0 or img_w == 0: continue # Skip if image is invalid

                aspect_ratio = img_w / img_h
                
                target_w = button_width
                target_h = int(target_w / aspect_ratio)

                if target_h > button_height:
                    target_h = button_height
                    target_w = int(target_h * aspect_ratio)
                
                if target_w <= 0 or target_h <= 0: continue # Skip if dimensions are invalid

                try:
                    resized_vis = cv2.resize(tmp_vis_for_button, (target_w, target_h), interpolation=cv2.INTER_AREA)
                except cv2.error as e:
                    # print(f"cv2.resize error: {e} with target_w={target_w}, target_h={target_h}")
                    continue

                q_image = QImage(resized_vis.data, resized_vis.shape[1], resized_vis.shape[0], resized_vis.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                current_button.setIcon(QIcon(pixmap))
                current_button.setIconSize(QSize(target_w, target_h))
                current_button.setText('') 
                current_button.setShortcut(str(msk_idx+1))

        else: # flag == 1 (clear proposals)
            for idx, button_proposal in enumerate(self.button_proposal_list):
                button_proposal.setIcon(QIcon()) # Clear icon
                button_proposal.setText('proposal{}'.format(idx+1))
                button_proposal.setIconSize(QSize(0,0))
                button_proposal.setShortcut(str(idx+1))

    def transform_input(self, image, box=None, points=None):
        if self.keep_input_size == True:
            return image, box, points
        else:
            h,w = image.shape[:2]
            scale_ratio = self.max_size / max(h,w)
            image = cv2.resize(image, (int(w*scale_ratio), int(h*scale_ratio)))
            if box is not None:
                box = box * scale_ratio
            if points is not None:
                points = points * scale_ratio
            return image, box, points
    
    def transform_output(self, masks, size):
        if self.keep_input_size == True:
            return masks
        else:
            h,w = size
            N = masks.shape[0]
            new_masks = np.zeros((N,h,w), dtype=np.uint8)
            for idx in range(N):
                new_masks[idx] = cv2.resize(masks[idx], (w,h))
            return new_masks

    def clickManualSegBBox(self):
        Box = self.canvas.currentBox
        if self.predictor is None or self.current_img == '' or Box == None:
            return
        # Use .copy() to ensure the array is contiguous with positive strides
        img = cv2.imread(self.current_img)[:,:,::-1].copy()
        rh, rw = img.shape[:2]
        input_box = np.array([Box[0].x(), Box[0].y(), Box[1].x(), Box[1].y()])
        img, input_box, _ = self.transform_input(img, box=input_box)
        if self.image_encoded_flag == False:
            self.predictor.set_image(img)
            self.image_encoded_flag = True
        masks_sam_raw, iou_prediction, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        # self.masks = masks_sam_raw # Store raw SAM masks if needed for other purposes

        # Transform raw masks to original image dimensions
        masks_sam_transformed = self.transform_output(masks_sam_raw.astype(np.uint8), (rh,rw))

        # Filter and reconstruct masks for display and shape creation
        # self.area_threshold is updated by the QSpinBox
        filtered_masks_for_show_and_shape = self.filter_and_reconstruct_masks(
            masks_sam_transformed, 
            self.area_threshold
        )

        # Show proposals using filtered masks
        self.show_proposals(filtered_masks_for_show_and_shape, 0) 
        
        self.sam_mask_proposal = []
        # Determine target_idx based on iou_prediction from unfiltered masks
        # This ensures the "best" proposal is still based on SAM's original confidence
        target_idx = np.argmax(iou_prediction) 

        for msk_idx in range(filtered_masks_for_show_and_shape.shape[0]):
            # Use the filtered mask to find contours for Shape objects
            mask_for_shape_creation = filtered_masks_for_show_and_shape[msk_idx].astype(np.uint8)
            
            # Ensure mask_for_shape_creation is C-contiguous
            mask_for_shape_creation_contiguous = np.ascontiguousarray(mask_for_shape_creation)
            
            points_list = cv2.findContours(mask_for_shape_creation_contiguous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            shape_type = 'polygon'
            tmp_sam_mask_shapes = [] # Stores Shape objects for the current proposal

            if not points_list: # If filtering removed all contours
                if msk_idx == target_idx:
                    self.sam_mask = []
                self.sam_mask_proposal.append([])
                continue

            for points in points_list:
                # Since the mask for `points_list` is already filtered,
                # we don't need to re-check area here unless there's a specific reason.
                # area = cv2.contourArea(points)
                # if area == 0: # Or some very small threshold if `filter_and_reconstruct_masks` might leave tiny artifacts
                #     continue

                pointsx = points[:,0,0]
                pointsy = points[:,0,1]

                shape = Shape(
                    label='Object', # Default label
                    shape_type=shape_type,
                    group_id=self.getMaxId() + 1,
                )
                for point_index in range(pointsx.shape[0]):
                    shape.addPoint(QtCore.QPointF(pointsx[point_index], pointsy[point_index]))
                shape.close()
                tmp_sam_mask_shapes.append(shape)
            
            if msk_idx == target_idx:
                self.sam_mask = tmp_sam_mask_shapes 
            self.sam_mask_proposal.append(tmp_sam_mask_shapes)

    def clickManualSegBox(self):
        ClickPos = self.canvas.currentPos
        ClickNeg = self.canvas.currentNeg
        if self.predictor is None or self.current_img == '' or (ClickPos == None and ClickNeg == None):
            return
        img = cv2.imread(self.current_img)[:,:,::-1].copy()
        rh, rw = img.shape[:2]

        input_clicks = []
        input_types = []
        if ClickPos != None:
            for pos in ClickPos:
                input_clicks.append([int(pos.x()), int(pos.y())])
                input_types.append(1)

        if ClickNeg != None:
            for neg in ClickNeg:
                input_clicks.append([int(neg.x()), int(neg.y())])
                input_types.append(0)
        if len(input_clicks) == 0:
            input_clicks = None
            input_types = None
        else:
            input_clicks = np.array(input_clicks)
            input_types = np.array(input_types)

        img, _, input_clicks = self.transform_input(img, points=input_clicks)

        if self.image_encoded_flag == False:
            self.predictor.set_image(img)
            self.image_encoded_flag = True
        masks_sam_raw, iou_prediction, _ = self.predictor.predict(
            point_coords=input_clicks,
            point_labels=input_types,
            multimask_output=True,
        )
        # self.masks = masks_sam_raw # Store raw SAM masks if needed for other purposes

        # Transform raw masks to original image dimensions
        masks_sam_transformed = self.transform_output(masks_sam_raw.astype(np.uint8), (rh,rw))

        # Filter and reconstruct masks for display and shape creation
        # self.area_threshold is updated by the QSpinBox
        filtered_masks_for_show_and_shape = self.filter_and_reconstruct_masks(
            masks_sam_transformed, 
            self.area_threshold
        )

        # Show proposals using filtered masks
        self.show_proposals(filtered_masks_for_show_and_shape, 0) 
        
        self.sam_mask_proposal = []
        # Determine target_idx based on iou_prediction from unfiltered masks
        # This ensures the "best" proposal is still based on SAM's original confidence
        target_idx = np.argmax(iou_prediction) 

        for msk_idx in range(filtered_masks_for_show_and_shape.shape[0]):
            # Use the filtered mask to find contours for Shape objects
            mask_for_shape_creation = filtered_masks_for_show_and_shape[msk_idx].astype(np.uint8)
            
            # Ensure mask_for_shape_creation is C-contiguous
            mask_for_shape_creation_contiguous = np.ascontiguousarray(mask_for_shape_creation)
            
            points_list = cv2.findContours(mask_for_shape_creation_contiguous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            shape_type = 'polygon'
            tmp_sam_mask_shapes = [] # Stores Shape objects for the current proposal

            if not points_list: # If filtering removed all contours
                if msk_idx == target_idx:
                    self.sam_mask = []
                self.sam_mask_proposal.append([])
                continue

            for points in points_list:
                # Since the mask for `points_list` is already filtered,
                # we don't need to re-check area here unless there's a specific reason.
                # area = cv2.contourArea(points)
                # if area == 0: # Or some very small threshold if `filter_and_reconstruct_masks` might leave tiny artifacts
                #     continue

                pointsx = points[:,0,0]
                pointsy = points[:,0,1]

                shape = Shape(
                    label='Object', # Default label
                    shape_type=shape_type,
                    group_id=self.getMaxId() + 1,
                )
                for point_index in range(pointsx.shape[0]):
                    shape.addPoint(QtCore.QPointF(pointsx[point_index], pointsy[point_index]))
                shape.close()
                tmp_sam_mask_shapes.append(shape)
            
            if msk_idx == target_idx:
                self.sam_mask = tmp_sam_mask_shapes 
            self.sam_mask_proposal.append(tmp_sam_mask_shapes)

    def filter_and_reconstruct_masks(self, masks_to_filter, area_threshold):
        if masks_to_filter is None or masks_to_filter.ndim != 3:
            return np.array([]) # Return empty if input is not as expected

        num_masks, h, w = masks_to_filter.shape
        # Ensure the output array is correctly initialized for boolean or uint8 masks
        filtered_reconstructed_masks = np.zeros_like(masks_to_filter, dtype=np.uint8)

        for i in range(num_masks):
            current_mask_slice = masks_to_filter[i]
            # Ensure current_mask_slice is C-contiguous and uint8 for findContours
            current_mask_contiguous = np.ascontiguousarray(current_mask_slice, dtype=np.uint8)
            
            contours, _ = cv2.findContours(current_mask_contiguous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours: # No contours found for this mask slice
                continue

            valid_contours = []
            for contour in contours: # Iterate through all found external contours for this mask slice
                area = cv2.contourArea(contour)
                if area >= area_threshold: # Apply threshold to ALL contours
                    valid_contours.append(contour)
            
            if valid_contours: # If any contours passed the threshold
                # Draw all valid contours onto the new mask for this index
                cv2.drawContours(filtered_reconstructed_masks[i], valid_contours, -1, (255), thickness=cv2.FILLED)
            # If no valid_contours, filtered_reconstructed_masks[i] remains all zeros, which is correct (empty mask)
                
        return filtered_reconstructed_masks

    def addSamMask(self):
        if len(self.sam_mask) > 0:
            label = self.default_label if self.default_label else 'Object'  # 自動設定ラベルを使用
            group_id = self.getMaxId() + 1
            if self.class_on_flag:
                xx = self.labelDialog.popUp(
                    text=label,
                    flags={},
                    group_id=group_id,
                )
                if len(xx) == 4:
                    label, _, group_id,_ = xx
                else:
                    label, _, group_id = xx
            if label is None:
                label = 'Object'
            if type(group_id) != int:
                group_id=self.getMaxId() + 1
            
            # 各マスクにラベルとグループIDを設定してaddLabelで追加
            for sam_mask in self.sam_mask:
                sam_mask.label = label
                sam_mask.group_id = group_id
                self.addLabel(sam_mask)
            
            # 現在の入力状態をクリア
            self.canvas.currentBox = None
            self.canvas.currentPos = None
            self.canvas.currentNeg = None
            
            # プロポーザルをクリア
            self.show_proposals()
            
            # ここでは sam_mask はクリアせず、最後に一括でクリアする
            # まずshapesに追加されたマスクをキャンバスにロード
            self.canvas.loadShapes([item.shape() for item in self.labelList])
            
            # UIの状態を更新
            self.actions.save.setEnabled(True)
            self.actions.editMode.setEnabled(True)
            
            # マスク表示確認のための強制描画更新
            self.canvas.update()
            QtWidgets.QApplication.processEvents()
            
            # マスク情報をクリア（描画が確実に行われた後）
            self.sam_mask = []
            self.sam_mask_proposal = []



    def cleanPrompt(self):
        self.canvas.currentBox = None
        self.canvas.currentPos = None
        self.canvas.currentNeg = None
        self.canvas.current = None
        self.sam_mask = []
        self.sam_mask_proposal = []
        self.show_proposals()
        self.canvas.setHiding()
        self.canvas.update()
        self.actions.editMode.setEnabled(True)



    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        if not text:
            previous_text = self.labelDialog.edit.text()
            xx = self.labelDialog.popUp(text)
            if len(xx) == 4:
                text, flags, group_id, _ = xx
            else:
                text, flags, group_id = xx
            if not text:
                self.labelDialog.edit.setText(previous_text)

        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
        if text:
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        # if self._config["auto_save"] or self.actions.saveAuto.isChecked():
        #     label_file = osp.splitext(self.imagePath)[0] + ".json"
        #     if self.output_dir:
        #         label_file_without_path = osp.basename(label_file)
        #         label_file = osp.join(self.output_dir, label_file_without_path)
        #     self.saveLabels(label_file)
        #     return
        # self.dirty = True
        self.actions.save.setEnabled(True)
        # title = __appname__
        # if self.filename is not None:
        #     title = "{} - {}*".format(title, self.filename)
        # self.setWindowTitle(title)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)

    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        # self.actions.undoLastPoint.setEnabled(drawing)
        # self.actions.undo.setEnabled(not drawing)
        # self.actions.delete.setEnabled(not drawing)
    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.current_img] = value

    def toolbar(self, title, actions=None):
        toolbar = self.addToolBar("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        if actions:
            utils.addActions(toolbar, actions)
        return toolbar

    def setEditMode(self):
        self.toggleDrawMode(True)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.actions.createMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)

        else:
            if createMode == "polygon":
                self.actions.createPointMode.setEnabled(True)
                self.actions.createMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)

            elif createMode == "point":
                self.actions.createMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
            elif createMode == "rectangle":
                self.actions.createMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
            else:
                raise ValueError("Unsupported createMode: %s" % createMode)
        self.actions.editMode.setEnabled(not edit)

    def validateLabel(self, label):
        return True

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def iou(self, target_mask, mask_list):
        target_mask = target_mask.reshape(1,-1)
        mask_list = mask_list.reshape(mask_list.shape[0], -1)
        i = (target_mask * mask_list)
        u = target_mask + mask_list - i
        return i.sum(1)/u.sum(1)


    def polygon2mask(self,polygon, size):
        mask = np.zeros((size)) # h,w
        contours = np.array(polygon)
        mask = cv2.fillPoly(mask, [contours.astype(np.int32)],1)
        return mask.astype(np.uint8)

    def mask2polygon(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours[0])
        return contours

    def editLabel(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        xx = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
        )
        if len(xx) == 4:
            text, flags, group_id,_ = xx
        else:
            text, flags, group_id = xx
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id

        self._update_shape_color(shape)
        if shape.group_id is None:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                )
            )
        else:
            item.setText("({}) {}".format(shape.group_id, shape.label))
        self.setDirty()
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            # rgb = self._get_rgb_by_label(shape.label)
            rgb = self._get_rgb_by_label(shape.group_id)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "({}) {}".format(shape.group_id, shape.label)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            # rgb = self._get_rgb_by_label(shape.label)
            rgb = self._get_rgb_by_label(shape.group_id)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _get_rgb_by_label(self, label):
        label = str(label)
        item = self.uniqLabelList.findItemByLabel(label)
        if item is None:
            item = self.uniqLabelList.createItemFromLabel(label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(label)
            self.uniqLabelList.setItemLabel(item, label, rgb)
        label_id = self.uniqLabelList.indexFromItem(item).row() + 1
        label_id += 0
        return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]

    def togglePolygons(self, value):
        for item in self.labelList:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def _update_shape_color(self, shape):
        # r, g, b = self._get_rgb_by_label(shape.label)
        r, g, b = self._get_rgb_by_label(shape.group_id)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def deleteSelectedShape(self):
        """選択されたシェイプを削除し、関連するprimaryを元に戻す"""
        # 選択されたシェイプが存在しない場合は何もしない
        if not self.canvas.selectedShapes:
            return
        
        # 削除前のデータ保存
        shapes_to_delete = self.canvas.selectedShapes.copy()
        
        # まず、secondaryシェイプが含まれているか確認
        for shape in shapes_to_delete:
            if shape.label == "secondary":
                try:
                    # 関連するprimaryを元に戻す処理
                    primary_ids = getattr(shape, 'primary_segment_ids', [])
                    secondary_group_id = shape.group_id
                    
                    # primaryを元に戻してから削除
                    self._restorePrimarySegments(primary_ids, secondary_group_id)
                except Exception as e:
                    print(f"Error restoring primaries: {e}")
        
        # 次に、すべてのシェイプを削除
        for shape in shapes_to_delete:
            # キャンバスからシェイプを削除
            self.canvas.deleteShape(shape)
            
            # labelListからシェイプに関連するアイテムを見つけて削除
            # 既存のremLabelsメソッドを使用
            self.remLabels([shape])
        
        # 変更を記録して更新
        self.setDirty()
        self.canvas.update()

    def _restorePrimarySegments(self, primary_ids, secondary_group_id):
        """
        primaryセグメントを元のマスク表示に戻す
        
        Args:
            primary_ids: 復元するprimaryセグメントのID配列
            secondary_group_id: 関連するsecondaryグループID
        """
        # デバッグ情報
        # print(f"Restoring primaries: {primary_ids} from secondary: {secondary_group_id}")
        
        # 復元されたシェイプを追跡
        restored = False
        
        for shape in self.canvas.shapes:
            # すべての可能なID属性をチェック
            shape_id = None
            for id_attr in ['id', 'group_id', 'particle_id']:
                if hasattr(shape, id_attr):
                    shape_id = getattr(shape, id_attr)
                    if shape_id and str(shape_id) in primary_ids:
                        break
            
            # 対象のIDを持つプライマリーセグメントを探す
            if shape_id and str(shape_id) in primary_ids:
                # print(f"Found primary to restore: {shape_id}")
                # セカンダリーグループIDが一致するか確認
                if hasattr(shape, 'secondary_group_id') and shape.secondary_group_id == secondary_group_id:
                    # print(f"Restoring shape with ID: {shape_id}")
                    
                    # 元のポイントと形状を復元
                    if hasattr(shape, 'original_points'):
                        # ポイントを復元
                        shape.points = shape.original_points.copy()
                        shape.shape_type = getattr(shape, 'original_shape_type', "polygon")
                        setattr(shape, 'is_mask_visible', True)  # マスク表示をON
                        
                        # secondary_group_id属性をクリア
                        if hasattr(shape, 'secondary_group_id'):
                            delattr(shape, 'secondary_group_id')
                        
                        # 見た目を更新
                        self._update_shape_color(shape)
                        restored = True
        
        # キャンバス全体を更新（labelListの操作は行わない）
        if restored:
            self.canvas.update()

    def duplicateSelectedShape(self):
        added_shapes = self.canvas.duplicateSelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def reducePoint(self):
        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data
        shapes = self.current_img
        shapes = [format_shape(item.shape()) for item in self.labelList.selectedItems()]
        rm_shapes = [item.shape() for item in self.labelList.selectedItems()]
        self.remLabels(rm_shapes)
        for shape in shapes:
            points = shape['points']
            min_dis = self.get_min_dis(points)
            points_new = [points[0]]
            for i in range(1,len(points)):
                d = math.sqrt((points[i][0] - points_new[-1][0]) ** 2 + (points[i][1] - points_new[-1][1]) ** 2)
                if d > (min_dis * 1.5):
                    points_new.append(points[i])
            shape['points'] = points_new
        #self.labelList.clear()
        for tmp_shape in shapes:
            shape = Shape(
                label=tmp_shape['label'],
                shape_type=tmp_shape['shape_type'],
                group_id=tmp_shape['group_id'],
            )
            for point_index in range(len(tmp_shape['points'])):
                shape.addPoint(QtCore.QPointF(tmp_shape['points'][point_index][0], tmp_shape['points'][point_index][1]))
            shape.close()
            self.addLabel(shape)
            tmp_item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(tmp_item)
            self.labelList.scrollToItem(tmp_item)
        self.canvas.loadShapes([item.shape() for item in self.labelList])
        self.actions.save.setEnabled(True)

    def get_min_dis(self, points):
        min_dis = 10000
        if len(points) >= 2:
            points_new = [points[0]]
            for i in range(1,len(points)):
                d = math.sqrt((points[i][0] - points_new[-1][0]) ** 2 + (points[i][1] - points_new[-1][1]) ** 2)
                min_dis = min(min_dis, d)
                points_new.append(points[i])
        return min_dis



    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)


    def noShapes(self):
        return not len(self.labelList)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def setZoom(self, value):
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.current_img] = (self.zoomMode, value)

    def paintCanvas(self):
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def _processSelectedSegments(self, selected_segments):
        """
        選択されたセグメントを処理し、グループ化と境界ボックスの計算を行う
        
        Args:
            selected_segments: 処理対象のセグメントリスト
            current_group_id: 新しいグループID
        
        Returns:
            box: バウンディングボックス座標 [min_x, min_y, max_x, max_y]
            all_points: すべてのセグメントの点のリスト
            primary_ids: プライマリー粒子IDのリスト
            segment_ids: セグメントIDのリスト
        """
        primary_ids = []
        segment_ids = []
        all_points = []
        
        # バウンディングボックスの最小値と最大値を初期化
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for segment in selected_segments:
            # 一次粒子IDを記録 - IDを変更せずに保持
            particle_id = getattr(segment, 'particle_id', None)
            if particle_id is not None:
                primary_ids.append(particle_id)
            
            # セグメントのIDを記録
            seg_id = getattr(segment, 'id', None) or getattr(segment, 'group_id', None)
            if seg_id is not None:
                segment_ids.append(str(seg_id))
            
            # グループ分けされたセグメントを保持
            if hasattr(self, 'grouped_segments'):
                self.grouped_segments.append(segment)
            
            # セグメントのポイントを集め、境界ボックスを更新
            points = segment.points
            for point in points:
                px, py = point.x(), point.y()
                all_points.append((px, py))
                min_x = min(min_x, px)
                min_y = min(min_y, py)
                max_x = max(max_x, px)
                max_y = max(max_y, py)
        
        # 境界ボックスと関連情報を返す
        box = np.array([min_x, min_y, max_x, max_y])
        return box, all_points, primary_ids, segment_ids

    def clickGroupSeg(self):
        # 選択されたセグメントの処理
        selected_segments = self.canvas.selectedShapes
        if not selected_segments:
            QMessageBox.warning(self, "警告", "グループ化するセグメントを選択してください")
            return
        
        # セグメントをグループ化して境界ボックスを計算
        bbox, all_points, primary_ids, segment_ids = self._processSelectedSegments(
            selected_segments
        )
        # primaryセグメントをOBBに変換して表示を更新
        primaries = self._convertPrimaryToOBB(selected_segments)
        
        # secondaryセグメント（グループのOBB）を生成
        secondary_shape = self._createOBBShape(all_points, bbox, "secondary")
        
        # SAMでsecondaryマスクを取得
        secondary_best_mask = None
        if self.predictor is not None and self.current_img:
            img = cv2.imread(self.current_img)[:,:,::-1].copy()
            rh, rw = img.shape[:2]
            input_box = bbox
            img, input_box, _ = self.transform_input(img, box=input_box)
            
            if not self.image_encoded_flag:
                self.predictor.set_image(img)
                self.image_encoded_flag = True
                
            masks, iou_prediction, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True,
            )
            
            masks = self.transform_output(masks.astype(np.uint8), (rh,rw))
            target_idx = np.argmax(iou_prediction)
            
            # 最適なマスクを選択
            secondary_best_mask = masks[target_idx]
        
        # 関連付けられたプライマリーセグメントIDを記録
        setattr(secondary_shape, 'primary_segment_ids', segment_ids)
        
        # セカンダリーグループを追加
        self.addLabel(secondary_shape)

        # UIの更新
        item = self.labelList.findItemByShape(secondary_shape)
        if item:
            # 追加したアイテムが見つかったら、それを選択して表示
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
            # ラベルテキストを更新して選択されたセグメントIDを表示
            ids_text = ", ".join(segment_ids) if segment_ids else "none"
            item.setText(f"secondary (contains: {ids_text})")

        # グループIDをインクリメント
        self.group_id += 1
        self.grouping_complete = True  # グループ分け完了

        # キャンバスの状態をリセット
        self.canvas.currentBox = None
        self.canvas.currentPos = None
        self.canvas.currentNeg = None
        self.sam_mask_proposal = []  # プロポーザルをクリア

        # UI を更新
        self.show_proposals()
        self.canvas.loadShapes(
            [item.shape() for item in self.labelList]
        )

        # 選択されたセグメントをCSVにエクスポート
        self.exportSelectedSegmentsToCSV(
            primaries, all_points, secondary_shape, secondary_best_mask
        )

        # ボタン状態を更新
        self.actions.save.setEnabled(True)  # 保存ボタンを有効化
        self.actions.editMode.setEnabled(True)  # 編集モードを有効化

        # セカンダリーマスクとL値の可視化
        self.visualizeSecondaryMaskAndL(secondary_shape, secondary_best_mask)

    def _convertPrimaryToOBB(self, segments):
        """
        primaryセグメントをOBBに変換する
        
        Args:
            segments: 変換対象のセグメントリスト
            group_id: グループID
        """
        primaries = []
        for segment in segments:
            # オリジナルのセグメント情報を保存
            setattr(segment, 'original_points', segment.points.copy())
            setattr(segment, 'original_shape_type', segment.shape_type)
            setattr(segment, 'is_mask_visible', False)  # マスク表示をOFF
            
            # セグメントの点を抽出
            points = [(p.x(), p.y()) for p in segment.points]
            # OBB用の空のバウンディングボックス初期化
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            # 点からバウンディングボックスを計算
            for point in points:
                min_x = min(min_x, point[0])
                min_y = min(min_y, point[1])
                max_x = max(max_x, point[0])
                max_y = max(max_y, point[1])
            
            # OBBに変換
            segment.points.clear()  # 既存の点をクリア
            
            if len(points) >= 4:
                # 最小面積の回転した長方形を計算
                points_array = np.array(points, dtype=np.float32)
                rect = cv2.minAreaRect(points_array)
                box_points = cv2.boxPoints(rect)
                
                # OBBの頂点を追加
                for point in box_points:
                    segment.addPoint(QtCore.QPointF(point[0], point[1]))
                
                segment.shape_type = "polygon"  # ポリゴンとして扱う
            else:
                # 点が少ない場合は通常の矩形を使用
                segment.addPoint(QtCore.QPointF(min_x, min_y))
                segment.addPoint(QtCore.QPointF(max_x, max_y))
                segment.shape_type = "rectangle"
            
            segment.close()
            
            # セカンダリーグループIDを関連付け
            setattr(segment, 'secondary_group_id', self.group_id)
            
            primaries.append(segment)

            # 見た目を更新
            self._update_shape_color(segment)

        return primaries

    def _createOBBShape(self, all_points, bbox, label_prefix):
        """
        点群からOBBを生成する
        
        Args:
            all_points: 点の配列
            bbox: バウンディングボックス（min_x, min_y, max_x, max_y）
            group_id: グループID
            label_prefix: ラベルプレフィックス（"primary"または"secondary"）
        
        Returns:
            生成されたShapeオブジェクト
        """
        min_x, min_y, max_x, max_y = bbox
        
        if len(all_points) >= 4:  # 少なくとも4点が必要
            points_array = np.array(all_points, dtype=np.float32)
            # 最小面積の回転した長方形を計算
            rect = cv2.minAreaRect(points_array)
            # 長方形の4つの頂点を取得
            box_points = cv2.boxPoints(rect)
            
            # OBBのシェイプを作成
            shape = Shape(
                label=f"{label_prefix}",
                shape_type="polygon",  # OBBはポリゴンとして扱う
                group_id=self.group_id,
            )
            
            # OBBの頂点を追加
            for point in box_points:
                shape.addPoint(QtCore.QPointF(point[0], point[1]))
            
        else:
            # 点が少ない場合は通常の矩形を使用
            shape = Shape(
                label=f"{label_prefix}",
                shape_type="rectangle",
                group_id=self.group_id,
            )
            shape.addPoint(QtCore.QPointF(min_x, min_y))
            shape.addPoint(QtCore.QPointF(max_x, max_y))
        
        shape.close()
        
        # バウンディングボックスの見た目を設定
        r, g, b = self._get_rgb_by_label(self.group_id)
        shape.line_color = QtGui.QColor(r, g, b, 200)
        shape.vertex_fill_color = QtGui.QColor(r, g, b, 120)
        shape.fill_color = QtGui.QColor(r, g, b, 30)
        
        return shape

    def exportSelectedSegmentsToCSV(
        self,
        primary_obbs,
        primary_masks,
        secondary_obb,
        secondary_mask,
    ):
        """
        選択されたセグメント（一次粒子）をグループ化した二次粒子としてCSVにエクスポートする
        一次粒子のラベルはそのまま保持し、グループ化情報のみを二次粒子として出力する
        
        Args:
            primary_obbs: 一次粒子のOBBのリスト
            primary_masks: 一次粒子のマスクのリスト
            secondary_obb: 二次粒子のOBB頂点リスト
            secondary_mask: 二次粒子のマスク配列(numpy.ndarray)
        """
        
        if not primary_obbs or not secondary_obb or not self.current_img:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No segments selected or no image loaded"))
            return
        
        # 画像ファイル名
        image_filename = self.current_img
        
        # スケーリング係数を取得
        scale = float(self.image_scaler_edit.text() or "1.0")
        
        # 一次粒子情報を収集
        primary_particles = []
        primary_ids = []
        
        # 通し番号用のカウンタ 
        # 既存の通し番号を継続するため、現在の最大IDを取得
        max_id = 0
        for shape in self.canvas.shapes:
            if hasattr(shape, 'particle_id') and shape.particle_id is not None:
                max_id = max(max_id, shape.particle_id)
        
        # 各一次粒子の処理
        for idx, (segment, mask) in enumerate(zip(primary_obbs, primary_masks)):
            # 各セグメントに一意なIDを割り当て
            if not hasattr(segment, 'particle_id') or segment.particle_id is None:
                segment.particle_id = max_id + idx + 1
            
            particle_id = segment.particle_id
            primary_ids.append(particle_id)
            
            # OBBの点を取得して長さを計算
            points = np.array([[p.x(), p.y()] for p in segment.points])
            
            # OpenCVの関数で形状解析
            if len(points) >= 4:  # OBBは4点必要
                # OBBから長さを計算
                rect = cv2.minAreaRect(points.astype(np.float32))
                (cx, cy), (width, height), angle = rect
                
                lmajor = max(width, height) * scale
                lminor = min(width, height) * scale
                l = (lmajor + lminor) / 2
                
                # マスクから面積を計算
                area = 0
                if mask is not None:
                    # マスクが画像サイズならnon-zeroピクセル数をカウント
                    if isinstance(mask, np.ndarray):
                        area = np.count_nonzero(mask) * scale**2
                    # マスクがポリゴンなら輪郭点から面積計算
                    elif hasattr(mask, 'points'):
                        mask_points = np.array([[p.x(), p.y()] for p in mask.points])
                        area = cv2.contourArea(mask_points.astype(np.int32)) * scale**2
                else:
                    # マスクがない場合はOBBから面積を概算
                    area = width * height * scale**2
                
                # 一次粒子情報を保存
                primary_particles.append({
                    "image_filename": image_filename,
                    "particle_id": particle_id,
                    "secondary_id": getattr(segment, 'secondary_group_id', ""),
                    "particle_type": segment.label,  # 元のラベルを保持
                    "Lmajor [um]": round(lmajor, 3),
                    "Lminor [um]": round(lminor, 3),
                    "L[um]": round(l, 1),
                    "Lmean[um]": "",
                    "n": "",
                    "Agg.": "",
                    "Area[um^2]": round(area, 2),
                })
        
        # 二次粒子情報を計算
        if secondary_obb:
            # 二次粒子のOBB頂点から長さを計算
            secondary_points = np.array([[p.x(), p.y()] for p in secondary_obb.points])
            
            # OBBから長さを計算
            rect = cv2.minAreaRect(secondary_points.astype(np.float32))
            (cx, cy), (width, height), angle = rect
            
            lmajor_secondary = max(width, height) * scale
            lminor_secondary = min(width, height) * scale
            l_secondary = (lmajor_secondary + lminor_secondary) / 2
            
            # 一次粒子の平均サイズを計算
            l_values = [p["L[um]"] for p in primary_particles]
            lmean = sum(l_values) / len(l_values) if l_values else 0
            
            # 凝集度（Aggregation）の計算
            n_particles = len(primary_particles)
            aggregation = l_secondary / lmean if lmean > 0 else 0
            
            # マスクから面積を計算
            total_area = 0
            if secondary_mask is not None:
                # NumPy配列の場合は非ゼロピクセル数をカウント
                if isinstance(secondary_mask, np.ndarray):
                    total_area = np.count_nonzero(secondary_mask) * scale**2
            else:
                # マスクがない場合はOBBから面積を概算
                total_area = width * height * scale**2
            
            # 二次粒子情報を追加
            secondary_particle = {
                "image_filename": image_filename,
                "particle_id": ",".join(map(str, primary_ids)),  # コンマ区切りの一次粒子ID
                "secondary_id": getattr(primary_obbs[0], 'secondary_group_id', "") if primary_obbs else "",
                "particle_type": "secondary",  # 二次粒子としてマーク
                "Lmajor [um]": round(lmajor_secondary, 3),
                "Lminor [um]": round(lminor_secondary, 3),
                "L[um]": round(l_secondary, 1),
                "Lmean[um]": round(lmean, 1),
                "n": n_particles,
                "Agg.": round(aggregation, 2),
                "Area[um^2]": round(total_area, 2),
            }
            
            # 全ての粒子情報を結合
            all_particles = primary_particles + [secondary_particle]
            
            # 実験パラメータの取得
            experiment_params = {
                "date": self.date_edit.text(),
                "experimenter": self.experimenter_edit.text(),
                "impurity_type": self.impurity_type_edit.text(),
                "impurity_concentration": self.impurity_conc_edit.text(),
                "seed_size": self.seed_size_edit.text(),
                "crystallization_time": self.crystal_time_edit.text(),
                "suspension_density": self.suspension_density_edit.text(),
                "image_scaler": self.image_scaler_edit.text(),
            }
            
            # CSVにエクスポート
            csv_path = csv_exporter.export_csv(all_particles, experiment_params, self.current_output_dir)
            work_dir = self.current_output_dir
            crystallization = CrystallizationAnalysis(
                csv_path, work_dir
            )
            crystallization(rank_range_min=-50, rank_range_max=750, rank_range_width=50)

        return None

    def saveMaskAndBBoxImages(self, filename_base):
        """
        アノテーションデータに基づいてマスク画像とバウンディングボックス画像を保存する
        
        Args:
            filename_base: 保存するファイル名のベース部分（拡張子なし）
        """
        if not self.canvas.shapes:
            return  # 形状がない場合は何もしない
        
        # 出力ディレクトリの作成
        mask_dir = os.path.join(self.current_output_dir, "masks")
        bbox_dir = os.path.join(self.current_output_dir, "bbox")
        
        if self.save_mask:
            os.makedirs(mask_dir, exist_ok=True)
        if self.save_bbox:
            os.makedirs(bbox_dir, exist_ok=True)
        
        # 入力画像の読み込み
        if hasattr(self, "image_np") and self.image_np is not None:
            img = self.image_np.copy()
        else:
            img = cv2.imread(self.current_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # マスク画像の作成と保存
        if self.save_mask:
            # canvas上の状態をそのままQPixmapにレンダリング
            width = self.canvas.pixmap.width()
            height = self.canvas.pixmap.height()
            output_pixmap = self.canvas.renderToPixmap(width=width, height=height)
            
            # QPixmapをQImageに変換
            qimage = output_pixmap.toImage()
            
            # QImageをNumPy配列に変換
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # 4は、RGBAのバイト数
            
            # 元の画像を読み込んでリサイズ
            original_img = cv2.imread(self.current_img)
            original_img = cv2.resize(original_img, (width, height))
            
            # マスク画像（透明部分を除く）と元の画像を合成
            alpha = arr[:, :, 3] / 255.0
            alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)  # 3チャンネル分に複製
            
            # RGB順をBGR順に変換（cv2はBGR形式）
            arr_bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            
            # 合成（マスクの不透明部分と元画像を混合）
            composite_img = (arr_bgr * alpha + original_img * (1 - alpha)).astype(np.uint8)
            
            # # 純粋なマスク画像（塗りつぶし領域）も保存
            # color_mask_filename = os.path.join(mask_dir, f"{os.path.basename(filename_base)}_mask_color.png")
            # output_pixmap.save(color_mask_filename, "PNG")
            
            # 合成画像の保存
            composite_filename = os.path.join(mask_dir, f"{os.path.basename(filename_base)}_composite.jpg")
            cv2.imwrite(composite_filename, composite_img)
        
        # バウンディングボックス画像の作成
        if self.save_bbox:
            # 以下は既存のコードと同じ
            bbox_img = img.copy()
            
            # 各形状のバウンディングボックスを描画
            for shape in self.canvas.shapes:
                points = np.array([[p.x(), p.y()] for p in shape.points], dtype=np.int32)
                
                # バウンディングボックスを取得
                x, y, w, h = cv2.boundingRect(points)
                
                # 形状のラベルとグループID
                label = shape.label
                group_id = shape.group_id if shape.group_id is not None else "N/A"
                
                # 色の決定（グループIDに基づく）
                color_idx = int(group_id) if isinstance(group_id, int) else 0
                color = LABEL_COLORMAP[color_idx % len(LABEL_COLORMAP)]
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                # バウンディングボックスを描画
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), color, 2)
                
                # ラベルとグループIDを描画
                text = f"{label} (ID:{group_id})"
                cv2.putText(bbox_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # バウンディングボックス画像の保存
            bbox_filename = os.path.join(bbox_dir, f"{os.path.basename(filename_base)}_bbox.jpg")
            cv2.imwrite(bbox_filename, cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def visualizeSecondaryMaskAndL(self, secondary_obb, secondary_mask):
        """
        secondaryマスクとL値（長軸・短軸）を可視化する
        
        Args:
            secondary_obb: 二次粒子のOBB
            secondary_mask: 二次粒子のマスク画像
        """
        if secondary_obb is None or secondary_mask is None or not self.current_img:
            QMessageBox.warning(self, "警告", "可視化するデータがありません")
            return
        
        # 元画像の読み込み
        original_img = cv2.imread(self.current_img)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]
        
        # 可視化用の画像を作成（元画像のコピー）
        visualization_img = original_img.copy()
        
        # スケーリング係数を取得
        scale = float(self.image_scaler_edit.text() or "1.0")
        
        # 1. マスクの可視化（半透明のオーバーレイ）
        if isinstance(secondary_mask, np.ndarray) and secondary_mask.shape[:2] == (h, w):
            # マスク領域を赤色で半透明にオーバーレイ
            mask_overlay = visualization_img.copy()
            mask_overlay[secondary_mask > 0] = np.array([255, 0, 0])  # 赤色
            
            # 半透明合成
            alpha = 0.4  # 透明度
            visualization_img = cv2.addWeighted(mask_overlay, alpha, visualization_img, 1 - alpha, 0)
        
        # 2. OBBの可視化と長軸・短軸の描画
        if secondary_obb:
            # OBBの頂点を取得（すでに存在する点を使用）
            points = np.array([[p.x(), p.y()] for p in secondary_obb.points], dtype=np.int32)
            
            # OBBを描画
            cv2.drawContours(visualization_img, [points], 0, (0, 255, 0), 2)
            
            # OBBの中心点を計算
            cx = np.mean(points[:, 0])
            cy = np.mean(points[:, 1])
            
            # 対角点のペアを計算（0-2と1-3が対角）
            diag1 = np.linalg.norm(points[0] - points[2])
            diag2 = np.linalg.norm(points[1] - points[3])
            
            # 辺の長さを計算
            edge1 = np.linalg.norm(points[0] - points[1])
            edge2 = np.linalg.norm(points[1] - points[2])
            
            # 長軸と短軸の長さを特定
            lmajor = max(edge1, edge2) * scale
            lminor = min(edge1, edge2) * scale
            
            # 長軸と短軸のベクトルを計算
            if edge1 > edge2:
                # edge1が長軸の場合
                major_vec = points[1] - points[0]
                minor_vec = points[2] - points[1]
            else:
                # edge2が長軸の場合
                major_vec = points[2] - points[1]
                minor_vec = points[1] - points[0]
            
            # ベクトルの正規化
            major_vec = major_vec / np.linalg.norm(major_vec) * max(edge1, edge2) / 2
            minor_vec = minor_vec / np.linalg.norm(minor_vec) * min(edge1, edge2) / 2
            
            # 長軸の描画（赤い矢印）
            start_major = (int(cx), int(cy))
            end_major = (int(cx + major_vec[0]), int(cy + major_vec[1]))
            cv2.arrowedLine(visualization_img, start_major, end_major, (255, 0, 0), 2)
            
            # 反対側の矢印も描画
            end_major_opposite = (int(cx - major_vec[0]), int(cy - major_vec[1]))
            cv2.arrowedLine(visualization_img, start_major, end_major_opposite, (255, 0, 0), 2)
            
            # 短軸の描画（青い矢印）
            start_minor = (int(cx), int(cy))
            end_minor = (int(cx + minor_vec[0]), int(cy + minor_vec[1]))
            cv2.arrowedLine(visualization_img, start_minor, end_minor, (0, 0, 255), 2)
            
            # 反対側の矢印も描画
            end_minor_opposite = (int(cx - minor_vec[0]), int(cy - minor_vec[1]))
            cv2.arrowedLine(visualization_img, start_minor, end_minor_opposite, (0, 0, 255), 2)
            
            # L値のテキスト表示
            cv2.putText(
                visualization_img, 
                f"Lmajor: {lmajor:.1f} um", 
                (int(cx) + 20, int(cy) - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
            cv2.putText(
                visualization_img, 
                f"Lminor: {lminor:.1f} um", 
                (int(cx) + 20, int(cy) + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        
        # GUIウィンドウは使用せず、直接ファイルに保存
        output_dir = os.path.join(self.current_output_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        image_basename = os.path.basename(self.current_img)
        filename = os.path.join(output_dir, f"{os.path.splitext(image_basename)[0]}_secondary_viz.jpg")
        cv2.imwrite(filename, cv2.cvtColor(visualization_img, cv2.COLOR_RGB2BGR))

    def update_area_threshold(self, value):
        self.area_threshold = value
        # Optionally, if a mask proposal is already shown, you might want to re-filter and update it.
        # This depends on the desired UX. For now, new proposals will use the new threshold.

    def filter_and_reconstruct_masks(self, masks_to_filter, area_threshold):
        if masks_to_filter is None or masks_to_filter.ndim != 3:
            return np.array([]) # Return empty if input is not as expected

        num_masks, h, w = masks_to_filter.shape
        # Ensure the output array is correctly initialized for boolean or uint8 masks
        filtered_reconstructed_masks = np.zeros_like(masks_to_filter, dtype=np.uint8)

        for i in range(num_masks):
            current_mask_slice = masks_to_filter[i]
            # Ensure current_mask_slice is C-contiguous and uint8 for findContours
            current_mask_contiguous = np.ascontiguousarray(current_mask_slice, dtype=np.uint8)
            
            contours, _ = cv2.findContours(current_mask_contiguous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours: # No contours found for this mask slice
                continue

            valid_contours = []
            for contour in contours: # Iterate through all found external contours for this mask slice
                area = cv2.contourArea(contour)
                if area >= area_threshold: # Apply threshold to ALL contours
                    valid_contours.append(contour)
            
            if valid_contours: # If any contours passed the threshold
                # Draw all valid contours onto the new mask for this index
                cv2.drawContours(filtered_reconstructed_masks[i], valid_contours, -1, (255), thickness=cv2.FILLED)
            # If no valid_contours, filtered_reconstructed_masks[i] remains all zeros, which is correct (empty mask)
                
        return filtered_reconstructed_masks

def get_parser():
    parser = argparse.ArgumentParser(description="pixel annotator by GroundedSAM")
    parser.add_argument(
        "--app_resolution",
        default='1000,1600',
        help="Application window resolution in format 'height,width'"
    )
    parser.add_argument(
        "--model_type",
        default='vit_b',
        help="Model type for SAM"
    )
    parser.add_argument(
        "--keep_input_size",
        type=bool,
        default=True,
        help="Keep original input size"
    )   
    parser.add_argument(
        "--max_size",
        default=720,
        help="Maximum size for image scaling"
    )
    parser.add_argument(
        "--category_file",
        default=None,
        help="Path to category file"
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=True,
        help="Save segmentation mask images"
    )
    parser.add_argument(
        "--save_bbox",
        action="store_true",
        default=True,
        help="Save bounding box visualization images"
    )
    parser.add_argument(
        "--save_labels",
        action="store_false",
        default=True,
        help="Save annotation labels as JSON files"
    )
    parser.add_argument(
        "--image_directory",
        default=None,
        help="Directory containing images to annotate"
    )   
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    global_h, global_w = [int(i) for i in args.app_resolution.split(',')]
    model_type = args.model_type
    keep_input_size = args.keep_input_size
    max_size = args.max_size
    category_file = args.category_file
    save_mask = args.save_mask
    save_bbox = args.save_bbox
    save_labels = args.save_labels
    image_directory = args.image_directory
    
    app = QApplication(sys.argv)
    main = MainWindow(
        global_h=global_h, 
        global_w=global_w, 
        model_type=model_type, 
        keep_input_size=keep_input_size, 
        max_size=max_size, 
        category_file=category_file,
        save_mask=save_mask,
        save_bbox=save_bbox,
        save_labels=save_labels,
        image_directory=image_directory
    )
    main.show()
    sys.exit(app.exec_())
