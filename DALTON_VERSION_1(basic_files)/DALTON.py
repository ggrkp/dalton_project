import datetime
# from datetime import datetime, time
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import ast
import copy
import csv
import os
import sqlite3
import sys
from sklearn.metrics import *
import pickle
from io import StringIO
from sklearn.model_selection import RandomizedSearchCV
import autosklearn
import numpy as np
import pandas as pd
import sklearn
from micromlgen import port
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tsfresh import (extract_features, extract_relevant_features,
                     select_features)
from tsfresh.feature_extraction import (ComprehensiveFCParameters,
                                        EfficientFCParameters,
                                        MinimalFCParameters)
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

from functions import Func
from GUI import Ui_MainWindow
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
import traceback
import sys
# from datetime import datetime


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Ui_MainWindow()

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()  # Done


class MainWindowUIClass(Ui_MainWindow):
    def __init__(self):

        global t_left
        super().__init__()
        self.functions = Func()
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())

    def setupUi(self, MW):

        super().setupUi(MW)

    def modelSlot(self):
        global completed_flag
        # Pass the function to execute
        # Any other args, kwargs are passed to the run function
        completed_flag = False
        # self.frame_progress.show()
        worker = Worker(self.modelCreate)
        # worker1 = Worker(self.setProgress)
        # worker.signals.result.connect(self.print_output)
        # worker.signals.finished.connect(self.thread_complete)
        # worker.signals.progress.connect(self.progress_fn)
        # Execute
        # self.threadpool.start(worker1)
        self.threadpool.start(worker)

    # def setProgress(self):
    #     self.progress.emit(100)
        # global t_left,completed_flag
        # try:
        #     # i=0
        #     # while (completed_flag==False) and i<t_left:
        #     for i in range(1,t_left,1):
        #         print(i)
        #         time.sleep(1)
        #         self.progress.emit(round((i/t_left)*100)    )
        #         # i=i+1

        #     self.progress.emit(100)
        # except:
        #     self.hidebar.emit()

    # ! 0. WELCOME SCREEN

    def get_started(self):
        global f_step, min_shift, split_size, completed_flag
        split_size = 0.3

        f_step = 0
        min_shift = 0
        completed_flag = False
        # !arxikopoihseis
        # Arxikopoihsh learning Type gia na mh faei error
        # self.stackedWidget.setCurrentIndex(0)  # Na ksekinaei apo 1h othoni
        # Otan ginei to import me valid file energopoieitai to next button
        self.nextButton.setEnabled(False)
        # Otan ginei to import me valid file energopoieitai to next button
        self.load_DB_btn.setEnabled(False)
        self.nextButton1.setEnabled(False)
        self.tiny_btn.setEnabled(False)
        self.frame_export.setEnabled(False)
        self.label_37.hide()
        self.textEdit.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.showen_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.text_settings.setReadOnly(True)
        self.result_text.setReadOnly(True)
        self.extract_frame.setEnabled(False)
        self.next_btn_7.setEnabled(False)
        self.roll_btn.setEnabled(False)
        self.submit_btn.setEnabled(False)
        self.frame_4.hide()
        self.radio_btn_c.setChecked(True)
        self.cl_radio_btn.setChecked(True)
        self.stackedWidget.setCurrentIndex(1)
        self.show_more_btn.setEnabled(False)
        self.tiny_btn.setEnabled(False)
        self.comboBox.clear()
        self.estimated.emit("")
        self.metricCombo.clear()
        self.ressampleCombo.clear()
        self.sort_box.clear()
        self.updateText.emit("")

    def home_slot(self):
        self.stackedWidget.setCurrentIndex(0)  # Na ksekinaei apo 1h othoni

    def history_tabs(self):
        self.show_more_btn.setEnabled(False)
        self.stackedWidget.setCurrentIndex(5)
        db_name = 'models.db'

# CREATE TABLE IF NOT EXISTS models(
#     name TEXT,
#     data BLOB,
#     timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
#     learning_type TEXT
# );
        class_query = "select name, timestamp from models where learning_type='Classification'"
        reg_query = "select name, timestamp from models where learning_type='Regression'"
        class_query_cnt = "select count(*) from models where learning_type='Classification'"
        reg_query_cnt = "select count(*) from models where learning_type='Regression'"
        table_c = self.classification_table
        table_r = self.regression_table
        table_c.setSelectionBehavior(QTableWidget.SelectRows)
        table_r.setSelectionBehavior(QTableWidget.SelectRows)

        self.functions.fill_tables(
            db_name, class_query, class_query_cnt, table_c)
        self.functions.fill_tables(db_name, reg_query, reg_query_cnt, table_r)

        header_c = table_c.horizontalHeader()
        header_c.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)
        header_c.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents)
        table_c.setHorizontalHeaderLabels(["Model Name", "TimeStamp"])

        header_r = table_r.horizontalHeader()
        header_r.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)
        header_r.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents)
        table_r.setHorizontalHeaderLabels(["Model Name", "TimeStamp"])

    # ! 1. IMPORT YOUR DATA SCREEN

# REFRESH

    def refreshAll(self):
        self.pathLine.setText(self.functions.getFileName())

# BROWSE BUTTON
    def browseSlot(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "CSV Files (*.csv);;Data Files(*.data)",
            options=options)
        if fileName:
            self.functions.setFileName(fileName)
            self.refreshAll()

# IMPORT BUTTON
    def importSlot(self):  # Slot gia to import button
        global fileName
        fileName = self.pathLine.text()
        if self.functions.isValid(fileName):
            # todo: na kanw akoma enan elegxo gia to an einai empty to file!
            # todo: na kanw elegxo an to file exei header h oxi kai na pernaw dika mou headers diaforetika

            global data
            data = self.functions.readFile(fileName)
            self.functions.setFileName(self.pathLine.text())
            self.refreshAll()
            self.functions.popup_window(
                "Your file is successfully imported!", " Success ", "Information")
            self.refreshAll()
            # Otan ginei to import me valid file energopoieitai to next button
            self.nextButton.setEnabled(True)
        else:
            self.functions.popup_window(
                "Please, choose a valid file to import!", " Error ", "Warning")

            self.refreshAll()

# RADIO BUTTONS - LEARNING TYPE
    def radio_c(self):
        global learning_type, series_flag

        if self.radio_btn_c.isChecked() or self.cl_radio_btn.isChecked():
            learning_type = "Classification"
            series_flag = False

    def radio_r(self):
        global learning_type, series_flag
        if self.radio_btn_r.isChecked() or self.reg_radio_btn.isChecked():
            learning_type = "Regression"
            series_flag = False

    def radio_ts(self):
        global learning_type, series_flag
        if self.radio_btn_ts.isChecked():
            learning_type = "Timeseries"
            series_flag = True

    def save_pred(self):
        pass
# CLEAR BUTTON

    def cancelSlot(self):
        # DONE: Na allaksw to clear button k na kanei clear - twra bazei to iris gia eukolia.
        self.pathLine.setText(
            "")

# NEXT BUTTON POU KANEI TO PREVIEW
    def nextSlot(self):  # Slot gia to next button
        self.frame_4.hide()

        global preview_num

        # NO WRAP GIA NA GINETAI SCROLL KAI NA MHN XALANE TA COLS
        self.describe_text_edit.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.describe_text_edit.setText(f"{data.describe()}")  # describe
        lines = len(data)
        print(lines)
        if lines > 10:
            preview_num = 20
        else:
            preview_num = lines
        self.stackedWidget.setCurrentIndex(2)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)

        # Molis pataw next tha kanei load to combo Box kai tha periexei ta features.
        # iterating the columns
        for col in data.columns:
            self.comboBox.addItem(col)
        self.comboBox.adjustSize()
        # dimiourgia table me ta dedomena tou dataset gia preview
        self.tableWidget.setRowCount(preview_num)  # set row Count
        self.tableWidget.setColumnCount(
            self.functions.colCount(data))  # set column count
        for i in range(preview_num):
            for j in range(self.functions.colCount(data)):
                self.tableWidget.setItem(
                    i, j, QTableWidgetItem(f"{ data.iloc[i][j] }"))
        header_labels = list(data.columns.values)
        self.tableWidget.setHorizontalHeaderLabels(header_labels)

        self.comboBox.adjustSize()
        if learning_type == 'Timeseries':
            self.label_37.show()
            self.load_DB_btn.hide()
            self.frame_4.show()
        else:
            self.load_DB_btn.show()    

# *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # *><><><><><<><><><><><><><><><><><<><><><><><><><><><><><<><><><><><><
    # ! 2. SELECT YOUR TARGET SCREEN

# TARGET FEATURE DROPDOWN KAI PREVIEW
    def featureSlot(self):  # Slot gia to drop down box
        global learning_type, X, y, y_df, item_index, f_step

        item_index = self.comboBox.currentIndex()
        print(f"Ok. Column {item_index} is your Target Feature! ")
        y = self.functions.pickTarget(item_index, data).to_numpy()
        y_df = self.functions.pickTarget(item_index, data)
        print("TYPE YDF:", type(y_df))
        # y.to_numpy()
        print("TYPE Y :", type(y))

        if learning_type == "Timeseries":
            X = copy.deepcopy(data)
            X.insert(0, 'id', 0)
            print(X)
            self.max_timeshift_box.setMinimum(1)
            self.max_timeshift_box.setMaximum(X.shape[0])
        else:
            X = self.functions.pickPredictors(item_index, data).to_numpy()
        # Otan ginei to import me valid file energopoieitai to next button

        self.load_DB_btn.setEnabled(True)
        self.nextButton1.setEnabled(True)
        if learning_type == "Classification":
            self.tiny_btn.setEnabled(True)

        for i in range(preview_num):
            for j in range(self.functions.colCount(data)):
                self.tableWidget.item(i, j).setBackground(
                    QtGui.QColor('lightgrey'))
            self.tableWidget.item(i, item_index).setBackground(
                QtGui.QColor('lightblue'))

# NEXT BUTTON - BACK BUTTON > DHLWNONTAI DEFAULTS GIA TO EPOMENO SCREEN ( MDOELING SCREEN - PARAMETERS )
    def backSlot(self):  # Slot gia to back button
        self.stackedWidget.setCurrentIndex(1)  # Pame ena screen pisw
        self.comboBox.clear()  # Katharizoyme to Combo box gia na mpoun nea features sto drop down
        # Otan ginei to import me valid file energopoieitai to next button
        self.nextButton.setEnabled(False)

    def forecasting_slot(self):
        global f_step
        f_step = self.forecast_box.value()

    def nextSlot_1(self):  # Next pou pigainei stis parametrous tou modeling
        # Pame ena screen mprosta sto next screen me preprocessing / modeling k parameter tuning
        global metric_dict, X, data, y_df, y, t_left, inc_est, disable_prepro, resample, resample_args, metric_var, ens_size, meta_disable, test_sz

        metric_dict = {
            "accuracy": autosklearn.metrics.accuracy,
            "balanced_accuracy": autosklearn.metrics.balanced_accuracy,
            "roc_auc": autosklearn.metrics.roc_auc,
            "average_precision": autosklearn.metrics.average_precision,
            "log_loss": autosklearn.metrics.log_loss,
            "precision": autosklearn.metrics.precision,
            "precision_macro": autosklearn.metrics.precision_macro,
            "precision_micro": autosklearn.metrics.precision_micro,
            "precision_samples": autosklearn.metrics.precision_samples,
            "precision_weighted": autosklearn.metrics.precision_weighted,
            "recall": autosklearn.metrics.recall,
            "recall_macro": autosklearn.metrics.recall_macro,
            "recall_micro": autosklearn.metrics.recall_micro,
            "recall_samples": autosklearn.metrics.recall_samples,
            "recall_weighted": autosklearn.metrics.recall_weighted,
            "f1": autosklearn.metrics.f1,
            "f1_macro": autosklearn.metrics.f1_macro,
            "f1_micro": autosklearn.metrics.f1_micro,
            "f1_samples": autosklearn.metrics.f1_samples,
            "f1_weighted": autosklearn.metrics.f1_weighted,
            "mean_absolute_error": autosklearn.metrics.mean_absolute_error,
            "mean_squared_error": autosklearn.metrics.mean_squared_error,
            "root_mean_squared_error": autosklearn.metrics.root_mean_squared_error,
            "mean_squared_log_error": autosklearn.metrics.mean_squared_log_error,
            "median_absolute_error": autosklearn.metrics.median_absolute_error,
            "r2": autosklearn.metrics.r2
        }
        if learning_type == "Classification":
            self.stackedWidget.setCurrentIndex(4)
            #! ARXIKOPOIHSEIS ANALOGA ME REGRESSION CLASSIFICATION
            self.test_sz_box.setValue(0.2)

            ens_size = 50
            meta_disable = 25

            # metric_list = ["accuracy", "balanced_accuracy", "roc_auc", "average_precision", "log_loss",
            #                "precision", "precision_macro", "precision_micro", "precision_samples", "precision_weighted",
            #                "recall", "recall_macro", "recall_micro", "recall_samples", "recall_weighted",
            #                "f1", "f1_macro", "f1_micro", "f1_samples", "f1_weighted"]
            if (sklearn.utils.multiclass.type_of_target(y) == "binary"):
                metric_list = ["accuracy", "balanced_accuracy", "roc_auc",
                               "average_precision", "log_loss", "precision", "recall", "f1"]
            else:
                metric_list = ["accuracy", "balanced_accuracy", "average_precision",
                               "precision_macro", "precision_micro", "precision_weighted",
                               "recall_macro", "recall_micro", "recall_weighted",
                               "f1_macro", "f1_micro", "f1_weighted"]

            self.metricCombo.addItems(metric_list)
            metric_var = None
            self.ardBox.hide()
            self.linearsvr_Box.hide()
            self.libsvrBox.hide()
            self.gausianPro_Box.hide()

            self.bernoulliBox.show()
            self.liblinearBox.show()
            self.libsvmBox.show()
            self.qdaBox.show()
            self.pasagrBox.show()
            self.multinbBox.show()
            self.gaussianBox.show()
            self.ldaBox.show()
            self.bernoulliBox.show()
            self.bernoulliBox.show()

            inc_est = ["adaboost",
                       "bernoulli_nb",
                       "decision_tree",
                       "extra_trees",
                       "gaussian_nb",
                       "gradient_boosting",
                       "k_nearest_neighbors",
                       "lda",
                       "liblinear_svc",
                       "libsvm_svc",
                       "multinomial_nb",
                       "passive_aggressive",
                       "random_forest",
                       "sgd",
                       "qda"]
            # MODELING DEFAULTS
            self.groupBox.setCheckable(True)
            self.groupBox.setChecked(False)

            self.timeLeft_box.setMinimum(30)
            self.timeLeft_box.setMaximum(30000)

            meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1])) for i in open(
                '/proc/meminfo').readlines())  # analoga me ta specs tou pc
            mem_Mb = int(meminfo['MemTotal']/1024)
            self.memory_box.setMinimum(1024)
            self.memory_box.setMaximum(mem_Mb)

            # DONE: Na dialeksw poies metrikes tha xrisimopoiw kai na emfanizontai sto teliko model.

            resample_list = ["Default", "Cross Validation", "Holdout"]
            self.ressampleCombo.addItems(resample_list)

            resample_args = {'train_size': 0.67}
            resample = "holdout"

            disable_prepro = None

            self.holdout_box.setEnabled(False)
            self.cvfoldsBox.setEnabled(False)
            self.test_sz_box.setMinimum(0.1)
            self.test_sz_box.setMaximum(0.9)
            self.split_box.setValue(0.2)

        elif learning_type == "Regression":

            self.stackedWidget.setCurrentIndex(4)

            #! ARXIKOPOIHSEIS ANALOGA ME REGRESSION CLASSIFICATION
            self.test_sz_box.setValue(0.2)

            ens_size = 50
            meta_disable = 25
            self.ardBox.show()
            self.linearsvr_Box.show()
            self.libsvrBox.show()
            self.gausianPro_Box.show()

            self.bernoulliBox.hide()
            self.liblinearBox.hide()
            self.libsvmBox.hide()
            self.qdaBox.hide()
            self.pasagrBox.hide()
            self.multinbBox.hide()
            self.gaussianBox.hide()
            self.ldaBox.hide()
            self.bernoulliBox.hide()
            self.bernoulliBox.hide()

            # metric_list = ["mean_absolute_error",
            #                "mean_squared_error",
            #                "root_mean_squared_error",
            #                "mean_squared_log_error",
            #                "median_absolute_error",
            #                "r2"]
            metric_list = ["mean_absolute_error",
                           "mean_squared_error",
                           "mean_squared_log_error",
                           "median_absolute_error",
                           "r2"]

            self.metricCombo.addItems(metric_list)
            metric_var = None
            inc_est = ["adaboost",
                       "ard_regression",
                       "decision_tree",
                       "extra_trees",
                       "gaussian_process",
                       "gradient_boosting",
                       "k_nearest_neighbors",
                       "liblinear_svr",
                       "libsvm_svr",
                       "random_forest",
                       "sgd"]
            # MODELING DEFAULTS
            self.groupBox.setCheckable(True)
            self.groupBox.setChecked(False)

            self.timeLeft_box.setMinimum(30)
            self.timeLeft_box.setMaximum(30000)

            meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1])) for i in open(
                '/proc/meminfo').readlines())  # analoga me ta specs tou pc
            mem_Mb = int(meminfo['MemTotal']/1024)
            self.memory_box.setMinimum(1024)
            self.memory_box.setMaximum(mem_Mb)

            # DONE: Na dialeksw poies metrikes tha xrisimopoiw kai na emfanizontai sto teliko model.

            resample_list = ["Default", "Cross Validation", "Holdout"]
            self.ressampleCombo.addItems(resample_list)

            resample_args = {'train_size': 0.67}
            resample = "holdout"

            disable_prepro = None

            self.holdout_box.setEnabled(False)
            self.cvfoldsBox.setEnabled(False)
            self.test_sz_box.setMinimum(0.1)
            self.test_sz_box.setMaximum(0.9)
            self.split_box.setMinimum(0.1)
            self.split_box.setMaximum(0.9)

        else:  # If learning_type == "Timeseries"
            self.stackedWidget.setCurrentIndex(8)
            self.roll_frame.setEnabled(True)
            if f_step != 0:
                X = X[:-f_step]
                y_df = y_df.iloc[f_step:]
            print(X)
            for col in X.columns:
                self.sort_box.addItem(col)


# *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # *><><><><><<><><><><><><><><><><><<><><><><><><><><><><><<><><><><><><
    # ! 3. PARAMETER TUNING SCREEN
# TEST SPLIT SPINBOX


    def test_sz_Slot(self):
        global test_sz
        self.test_sz_box.setSingleStep(0.01)
        test_sz = self.test_sz_box.value()

# TIME LEFT FOR THIS TASK SPINBOX
    def timeleft_Slot(self):
        global t_left
        t_left = self.timeLeft_box.value()

# ENSEMBLE MEMORY LIMIT SPINBOX
    def ensmemory_Slot(self):
        global mem_limit
        mem_limit = self.memory_box.value()

# METRIC SELECTION DROPDOWN
    def metricBox(self):
        global metric_var, metric_dict, metric_text
        combo_idx_metric = self.metricCombo.currentIndex()
        metric_text = self.metricCombo.itemText(combo_idx_metric)
        metric_var = metric_dict[f"{metric_text}"]
        # metric_var = autosklearn.metrics.log_loss
        # print("metric_var is : ", metric_var)

# RESAMPLING STRATEGY DROPDOWN
    def resampleBox(self):
        global resample, resample_args
        combo_idx_resample = self.ressampleCombo.currentIndex()
        resample_str = self.ressampleCombo.itemText(combo_idx_resample)

        if resample_str == "Cross Validation":
            self.cvfoldsBox.setEnabled(True)
            self.holdout_box.setEnabled(False)
            self.cvfoldsBox.setRange(2, 10)
            resample_args = {'folds': 2}
            resample = "cv"

        elif resample_str == "Default":
            self.holdout_box.setEnabled(False)
            self.cvfoldsBox.setEnabled(False)
            resample_args = {'train_size': 0.67}
            resample = "holdout"

        else:
            self.holdout_box.setEnabled(True)
            self.cvfoldsBox.setEnabled(False)
            self.holdout_box.setRange(0.1, 1.0)
            self.holdout_box.setSingleStep(0.01)
            self.holdout_box.setDecimals(2)
            self.holdout_box.setValue(0.67)
            resample_args = {'train_size': 0.67}
            resample = "holdout"

# CV FOLDS - HOLDOUT TRAIN SIZE SPINBOXES
    def cv_Folds(self):
        global resample_args
        folds = self.cvfoldsBox.value()
        resample_args = {'folds': folds}

    def holdout_Size(self):
        global resample_args
        h_size = self.holdout_box.value()
        resample_args = {'train_size': h_size}

# NEXT BUTTON (PRINTS) - BACK BUTTON
    # DONE: To next button na se pigainei sto epomeno screen kai na pernaei oti arguments kai data xreiazontai gi auto :

    # def nextSlot_2(self):
    #     print(f"Included:   {inc_est}")
    #     print("Metric:", metric_var)
    #     print("Resampling_Technique:", resample)
    #     print("Args:", resample_args)
    #     print(t_left)
    #     print("split = ", test_sz)
    #     print("ensemble size = ", ens_size)
    #     print("meta-learning = ", meta_disable)
    #     print("Yoleleison")

    def backSlot_1(self):
        if series_flag == False:
            self.stackedWidget.setCurrentIndex(2)
        else:  # Pame ena screen pisw
            self.stackedWidget.setCurrentIndex(8)  # Pame ena screen pisw

        self.checkBox_16.setChecked(False)
        self.metricCombo.clear()
        self.ressampleCombo.clear()

# BUTTON GIA MANUALL SELCECTION ESTIMATORS

    def select_all_Estimators(self):
        global inc_est
        if self.groupBox.isChecked():
            inc_est = []
        else:
            if learning_type == "Classification":

                self.extratreeBox.setChecked(False)
                self.adaBox.setChecked(False)
                self.gaussianBox.setChecked(False)
                self.bernoulliBox.setChecked(False)
                self.rforoestBox.setChecked(False)
                self.qdaBox.setChecked(False)
                self.multinbBox.setChecked(False)
                self.ldaBox.setChecked(False)
                self.libsvmBox.setChecked(False)
                self.liblinearBox.setChecked(False)
                self.dtreeBox.setChecked(False)
                self.sgdBox.setChecked(False)
                self.gradientBox.setChecked(False)
                self.knnBox.setChecked(False)
                self.pasagrBox.setChecked(False)
                inc_est = ["adaboost",
                           "bernoulli_nb",
                           "decision_tree",
                           "extra_trees",
                           "gaussian_nb",
                           "gradient_boosting",
                           "k_nearest_neighbors",
                           "lda",
                           "liblinear_svc",
                           "libsvm_svc",
                           "multinomial_nb",
                           "passive_aggressive",
                           "random_forest",
                           "sgd",
                           "qda"]
            else:
                self.extratreeBox.setChecked(False)
                self.adaBox.setChecked(False)
                self.ardBox.setChecked(False)
                self.gausianPro_Box.setChecked(False)
                self.rforoestBox.setChecked(False)
                self.multinbBox.setChecked(False)
                self.libsvrBox.setChecked(False)
                self.linearsvr_Box.setChecked(False)
                self.dtreeBox.setChecked(False)
                self.sgdBox.setChecked(False)
                self.gradientBox.setChecked(False)
                self.knnBox.setChecked(False)

                inc_est = ["adaboost",
                           "ard_regression",
                           "decision_tree",
                           "extra_trees",
                           "gaussian_process",
                           "gradient_boosting",
                           "k_nearest_neighbors",
                           "liblinear_svr",
                           "libsvm_svr",
                           "random_forest",
                           "sgd"]

# ESTIMATORS CHECK BOXES (MPAINOUN SE LISTA) :
    def ard_Checked(self):
        global inc_est
        box = self.ardBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def gausianPro_Checked(self):
        global inc_est
        box = self.gausianPro_Box
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def liblinearsvr_Checked(self):
        global inc_est
        box = self.linearsvr_Box
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def libsvr_Checked(self):
        global inc_est
        box = self.libsvrBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def adaChecked(self):
        global inc_est
        box = self.adaBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def bernoulliChecked(self):
        global inc_est
        box = self.bernoulliBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def dectreeChecked(self):
        global inc_est
        box = self.dtreeBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def extraTreeChecked(self):
        global inc_est
        box = self.extratreeBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def gausianChecked(self):
        global inc_est
        box = self.gaussianBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def gradient_checked(self):
        global inc_est
        box = self.gradientBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def knnChecked(self):
        global inc_est
        box = self.knnBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def ldaChecked(self):
        global inc_est
        box = self.ldaBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def liblinearChecked(self):
        global inc_est
        box = self.liblinearBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def libsvmChecked(self):
        global inc_est
        box = self.libsvmBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def multnbChecked(self):
        global inc_est
        box = self.multinbBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def pasagrChecked(self):
        global inc_est
        box = self.pasagrBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def rforestChecked(self):
        global inc_est
        box = self.rforoestBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def sgdChecked(self):
        global inc_est
        box = self.sgdBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

    def qdaChecked(self):
        global inc_est
        box = self.qdaBox
        box_state = box.isChecked()
        est_name = box.text()
        inc_est = self.functions.app_Estimator(inc_est, box_state, est_name)

# DISABLE PREPROCESSING CHECKBOX & DISABLE ENSEMBLING CHECKBOX
    def prepro_Checked(self):  # Disable Feature Preprocessing
        global disable_prepro
        if self.checkBox_16.isChecked():
            disable_prepro = []
            disable_prepro.append("no_preprocessing")
        else:
            disable_prepro = None

    def ensembling_checked(self):  # Disable Feature Preprocessing
        global ens_size
        if self.ensembling_checkbox.isChecked():
            ens_size = 1
        else:
            ens_size = 50

    def metalearning_checked(self):
        global meta_disable
        if self.metalearning_checkbox.isChecked():
            meta_disable = 0
        else:
            meta_disable = 25

    def model_pop(self):
        global model, inc_est, resample, completed_flag
        minutes = t_left/60
        seconds = t_left
        self.stackedWidget.setEnabled(False)

        popup = QtWidgets.QMessageBox()
        popup.setWindowTitle(" Running ")
        popup.setText(
            "Please, wait. An ensemble is being created...  ")
        if minutes < 1:
            popup.setInformativeText(
                f"This process will take about {seconds} seconds.   Press OK to continue...")
        elif minutes == 1:
            popup.setInformativeText(
                f"This process will take about 1 minute"
            )
        else:
            popup.setInformativeText(
                f"This process will take about {minutes} minutes. Press OK to continue...")

        popup.setStandardButtons(QtWidgets.QMessageBox.Ok)
        popup.setIcon(QtWidgets.QMessageBox.Information)
        popup.exec_()

    def pop_success(self):
        self.functions.popup_window(
            "A new ensemble is created successfully!", " Done ", "Information")

    def aboutslot(self):
        self.functions.popup_window("Machine learning processes require expertise, time and resources in order to successfully produce efficient models. Automated machine learning(AutoML) is an innovative field of computer science which removes the human factor from time consuming machine learning processes making it easier to create computational models. In this diploma thesis, an automated machine learning tool for IoT applications was designed and developed. The developed tool is an easy-to-use, user-friendly computer application that aims to further facilitate experienced users in implementing machine learning processes, but mainly to make Machine Learning accessible to non-experienced people. The application was implemented in Python and it is based on the Python library 'auto-sklearn', through which it implements the process of creating machine learning models, which is its main function. Secondary functions of the application are the conversion of timeseries problems into classical machine learning problems, the extraction of features from timeseries data sets, the export of models supported by microcontrollers, the storage of the generated models for retrieval and reuse. Additionally, via the PyQt library, the graphical user interface of the application was implemented for an easy navigation by unexperienced users. We consider that the research, the achievement of all the objectives which were set during the initial design of the application and the final tool are an important contribution in the field of Automated Machine Learning, making it accessible to more people.\n\nGeorgakopoulos Georgios\n2021","About DALTON 1.0", "Information")


# RUN BUTTON => START CREATING ENSEMBLES

    def modelCreate(self):
        global model, inc_est, resample, completed_flag, model_name, metric_var, ens_size, meta_disable, resample_args, disable_prepro, t_left, metric_text
        try:
            #! ELEGXOS AN EXOUN EPILEXTHEI ESTIMATORS:
            if not inc_est:
                self.functions.popup_window(
                    "Please select at least one Estimator from the list.", " Error ", "Warning")

            else:
                minutes = t_left/60
                seconds = t_left
                self.modelpop.emit()
                # a = datetime.datetime.now()
                # b = a + datetime.timedelta(0,t_left) # days, seconds, then other fields.

                # self.estimated.emit(f"EST completion on: {b}")
                #! Data Splitting:
                X_train, X_test, y_train, y_test = self.functions.splitData(
                    X, y, test_sz)
                base = os.path.basename(fileName)
                dataset_name = os.path.splitext(base)[0]

                #! Check learning problem Type:
                if learning_type == "Classification":  # classifier call
                    model = self.functions.callClassifier(
                        t_left, inc_est, disable_prepro, resample, resample_args, metric_var, ens_size, meta_disable)
                elif learning_type == "Regression":  # regressor call
                    model = self.functions.callRegressor(
                        t_left, inc_est, disable_prepro, resample, resample_args, metric_var, ens_size, meta_disable)

                #! Model Fit:

                model = self.functions.fitModel(
                    X_train, y_train, model, dataset_name)

                completed_flag = True
                # self.hidebar.emit()
                pred = model.predict(X_test)
                # DONE: Na ftiaksw to popup me signals
                print("Done")
                print(metric_var)
                self.model_success.emit()

                # self.functions.popup_window(
                #     "A new ensemble is created successfully!", " Done ", "Information")
                # DONE na ftiaksw ta signals gia na grafei edw.
                #! Metric results:
                if learning_type == 'Regression':
                    self.updateText.emit(
                        f"Max error: {sklearn.metrics.max_error(y_test, pred)}")
                    if metric_var == metric_dict["mean_absolute_error"] or metric_var == None:
                        self.updateText.emit(
                            f"Mean AE: {sklearn.metrics.mean_absolute_error(y_test, pred)}")
                    elif metric_var == metric_dict["mean_squared_error"]:
                        self.updateText.emit(
                            f"MSE: {sklearn.metrics.mean_squared_error(y_test, pred)}")
                    elif metric_var == metric_dict["mean_squared_log_error"]:
                        self.updateText.emit(
                            f"MSLE: {sklearn.metrics.mean_squared_log_error(y_test, pred)}")
                    elif metric_var == metric_dict["median_absolute_error"]:
                        self.updateText.emit(
                            f"Median AE: {sklearn.metrics.median_absolute_error(y_test, pred)}")
                    elif metric_var == metric_dict["r2"]:
                        self.updateText.emit(
                            f"R2: {sklearn.metrics.r2_score(y_test, pred)}")

                #     print("Max error", sklearn.metrics.max_error(y_test, pred))
                #     self.result_text.setText(
                #         f"Max error: {sklearn.metrics.max_error(y_test, pred)}")  # describe

                elif learning_type == "Classification":
                    if metric_var == metric_dict["accuracy"] or metric_var == None:
                        self.updateText.emit(
                            f"Accuracy: {sklearn.metrics.accuracy_score(y_test, pred)}")
                    elif metric_var == metric_dict["balanced_accuracy"]:
                        self.updateText.emit(
                            f"Balanced Acc: {sklearn.metrics.balanced_accuracy_score(y_test, pred)}")
                    elif metric_var == metric_dict["roc_auc"]:
                        self.updateText.emit(
                            f"Roc Auc: {sklearn.metrics.roc_auc_score(y_test, pred)}")
                    elif metric_var == metric_dict["average_precision"]:
                        self.updateText.emit(
                            f"Avg Precision: {sklearn.metrics.average_precision_score(y_test, pred)}")
                    elif metric_var == metric_dict["precision"]:
                        self.updateText.emit(
                            f"Precision: {sklearn.metrics.precision_score(y_test, pred)}")
                    elif metric_var == metric_dict["log_loss"]:
                        self.updateText.emit(
                            f"Log Loss: {sklearn.metrics.log_loss(y_test, pred)}")
                    elif metric_var == metric_dict["recall"]:
                        self.updateText.emit(
                            f"Recall: {sklearn.metrics.recall_score(y_test, pred)}")
                    elif metric_var == metric_dict["f1"]:
                        self.updateText.emit(
                            f"F1: {sklearn.metrics.f1_score(y_test, pred)}")

                    elif metric_var == metric_dict["precision_micro"]:
                        self.updateText.emit(
                            f"Precision Micro: {sklearn.metrics.precision_score(y_test, pred, average='micro')}")

                    elif metric_var == metric_dict["precision_macro"]:
                        self.updateText.emit(
                            f"Precision Macro: {sklearn.metrics.precision_score(y_test, pred, average='macro')}")

                    elif metric_var == metric_dict["precision_weighted"]:
                        self.updateText.emit(
                            f"Precision Weighted: {sklearn.metrics.precision_score(y_test, pred, average='weighted')}")

                    elif metric_var == metric_dict["recall_micro"]:
                        self.updateText.emit(
                            f"Recall Micro: {sklearn.metrics.recall_score(y_test, pred, average='micro')}")

                    elif metric_var == metric_dict["recall_macro"]:
                        self.updateText.emit(
                            f"Recall Macro: {sklearn.metrics.recall_score(y_test, pred, average='macro')}")

                    elif metric_var == metric_dict["recall_weighted"]:
                        self.updateText.emit(
                            f"Recall Weighted: {sklearn.metrics.recall_score(y_test, pred, average='weighted')}")

                    elif metric_var == metric_dict["f1_macro"]:
                        self.updateText.emit(
                            f"F1 Macro: {sklearn.metrics.f1_score(y_test, pred, average='macro')}")

                    elif metric_var == metric_dict["f1_weighted"]:
                        self.updateText.emit(
                            f"F1 Weighted: {sklearn.metrics.f1_score(y_test, pred, average='weighted')}")

                    elif metric_var == metric_dict["f1_micro"]:
                        self.updateText.emit(
                            f"F1 Micro: {sklearn.metrics.f1_score(y_test, pred, average='micro')}")

                #     print("Accuracy score",
                #           sklearn.metrics.accuracy_score(y_test, pred))
                #     self.result_text.setText(
                #         f"Accuracy: {sklearn.metrics.accuracy_score(y_test, pred)}")  # describe

                # if self.savemodel_Box.isChecked():
                #     model_name = self.model_text.toPlainText()
                #     if model_name == "":
                #         model_name = "unnamed_model"
                #     self.functions.store_model(
                #         model, model_name, learning_type)
                #     #  # !Save model as file:
                #     options = QtWidgets.QFileDialog.Options()
                #     options |= QtWidgets.QFileDialog.DontUseNativeDialog
                #     filename, _ = QFileDialog.getSaveFileName(None, "Save File As", model_name,
                #                                             "Pickle Files (*);", options=options)
                #     if filename:
                #         with open(filename, 'wb') as f:
                #             pickle.dump(model, f)
                # self.stackedWidget.setEnabled(True)
                self.modelsave.emit()
        except:  # lathos learning type h lathos target variable
            self.m_error.emit()

    def model_save(self):
        global model, inc_est, resample, completed_flag, model_name, filename
        if self.savemodel_Box.isChecked():
            # model_name = self.model_text.toPlainText()
            # if model_name == "":
            model_name = "unnamed_model"

            #  # !Save model as file:
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QFileDialog.getSaveFileName(None, "Save File As", model_name,
                                                      "Pickle Files (*);", options=options)
            if filename:
                # print(f"To filename einai to eksis: {filename}")
                base = os.path.basename(filename)
                model_name = os.path.splitext(base)[0]
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
            self.functions.store_model(filename,
                                       model, model_name, learning_type)
        self.stackedWidget.setEnabled(True)

    def model_error(self):
        print("An error has occured!")
        self.comboBox.clear()
        self.metricCombo.clear()
        self.ressampleCombo.clear()
        self.nextButton.setEnabled(False)
        self.functions.popup_window(
            "An error has occured. Restart the Process. Make sure you defined the correct learning type and selected the correct target feature!", " Error ", "Warning")
        self.stackedWidget.setEnabled(True)
        if (series_flag == True):
            self.stackedWidget.setCurrentIndex(9)
        else:
            self.stackedWidget.setCurrentIndex(1)
# *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # *><><><><><<><><><><><><><><><><><<><><><><><><><><><><><<><><><><><><
    # ! 4. MODELS SCREEN

    def models_back(self):
        self.stackedWidget.setCurrentIndex(2)

    def load_DB(self):  # tha kanei connect meta tha kanei load kai tha petaei mesa ta records!
        self.stackedWidget.setCurrentIndex(3)
        db_name = 'models.db'
        query1 = "select name, timestamp from models"
        query1_cnt = "select count(*) from models"
        table4 = self.dbTable

        self.functions.fill_tables(db_name, query1, query1_cnt, table4)
        # table4.setHorizontalHeaderLabels(["Model Name", "Timestamp"])

        header_4 = table4.horizontalHeader()
        header_4.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)
        header_4.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents)
        table4.setHorizontalHeaderLabels(["Model Name", "TimeStamp"])

    # * LOAD MODEL BUTTON!!

    def fetch_model_2(self):
        global ft_model
        self.predict_btn.setEnabled(True)

        if self.tabWidget.currentIndex() == 0:
            table = self.classification_table
            text_edit = self.textEdit_2
        else:
            table = self.regression_table
            text_edit = self.textEdit_2
        if self.stackedWidget.currentIndex() == 3:
            table = self.dbTable
            text_edit = self.textEdit
        currow = table.currentRow()
        tstamp = table.item(currow, 1)
        self.show_more_btn.setEnabled(True)

        if tstamp == None:
            self.functions.popup_window(
                "No Models Were Selected! Please select a model from the list.", " Error ", "Warning")

        else:
            try:
                self.pushButton_7.setEnabled(True)
                tstamp = table.item(currow, 1).text()
                ft_model = self.functions.load_model(tstamp)
                # show loaded model statistics
                text_edit.setText(ft_model.sprint_statistics())
            except:
                self.show_more_btn.setEnabled(False)
                self.predict_btn.setEnabled(False)
                text_edit.setText("Model does not exist")

    def show_more_slot(self):
        global ft_model
        try:

            # self.info_text.setText(ft_model.get_models_with_weights())
            model_list = ft_model.get_models_with_weights()
            numrows = len(model_list)  # 6 rows in your example
            numcols = len(model_list[0])  # 3 columns in your example
            self.weights_table.setColumnCount(numcols)
            self.weights_table.setRowCount(numrows)
            print(numrows)
            for row in range(numrows):
                self.weights_table.setRowHeight(row, 510)
                for column in range(numcols):
                    self.weights_table.setItem(
                        row, column, QTableWidgetItem((str(model_list[row][column]))))
            header = self.weights_table.horizontalHeader()
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
            self.weights_table.setHorizontalHeaderLabels(
                ["Weight", "Information"])
            self.stackedWidget.setCurrentIndex(6)
        except:
            self.functions.popup_window(
                "Selected model Does not exist! ", " Error ", "Warning")

    def ok_slot(self):
        self.stackedWidget.setCurrentIndex(5)

    def show_ensembles(self):
        global ft_model
        # print(ft_model.show_models())
        print(ft_model.show_models())

    def predict_y(self):
        try:
            self.textEdit.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

            global X, y, ft_model, learning_type
            # score = ft_model.score(X, y)
            # print("Test Score with pickle model: {0:.2f} %". format(
            #     100 * score))
            y_predict = ft_model.predict(X)

            # print(y_predict)

            # !new!
            self.textEdit.append(
                "---------------------------------------------------------------------------------------------------")
            self.textEdit.append("\t\tPREDICTION RESULTS")
            self.textEdit.append(
                "---------------------------------------------------------------------------------------------------")
            redColor = QColor(178, 34, 34)
            greenColor = QColor(0, 100, 0)
            blackColor = QColor(0, 0, 0)
            if(learning_type == "Classification"):

                # self.textEdit.append(
                #     f"Target_type: {sklearn.utils.multiclass.type_of_target(y)}")
                self.textEdit.append(
                    f"Accuracy: {sklearn.metrics.accuracy_score(y, y_predict)}")
                self.textEdit.append(
                    f"Balanced Acc: {sklearn.metrics.balanced_accuracy_score(y, y_predict)}")

                if (sklearn.utils.multiclass.type_of_target(y) == "binary"):
                    self.textEdit.append(
                        f"Roc Auc: {sklearn.metrics.roc_auc_score(y, y_predict)}")
                    self.textEdit.append(
                        f"Avg Precision: {sklearn.metrics.average_precision_score(y, y_predict)}")
                    self.textEdit.append(
                        f"Precision: {sklearn.metrics.precision_score(y, y_predict)}")
                    self.textEdit.append(
                        f"Log Loss: {sklearn.metrics.log_loss(y, y_predict)}")
                    self.textEdit.append(
                        f"Recall: {sklearn.metrics.recall_score(y, y_predict)}")
                    self.textEdit.append(
                        f"F1: {sklearn.metrics.f1_score(y, y_predict)}")
                else:
                    self.textEdit.append(
                        f"Precision Micro: {sklearn.metrics.precision_score(y, y_predict, average='micro')}")
                    self.textEdit.append(
                        f"Precision Macro: {sklearn.metrics.precision_score(y, y_predict, average='macro')}")
                    self.textEdit.append(
                        f"Precision Weighted: {sklearn.metrics.precision_score(y, y_predict, average='weighted')}")
                    self.textEdit.append(
                        f"Recall Micro: {sklearn.metrics.recall_score(y, y_predict, average='micro')}")
                    self.textEdit.append(
                        f"Recall Macro: {sklearn.metrics.recall_score(y, y_predict, average='macro')}")
                    self.textEdit.append(
                        f"Recall Weighted: {sklearn.metrics.recall_score(y, y_predict, average='weighted')}")
                    self.textEdit.append(
                        f"F1 Micro: {sklearn.metrics.f1_score(y, y_predict, average='micro')}")
                    self.textEdit.append(
                        f"F1 Macro: {sklearn.metrics.f1_score(y, y_predict, average='macro')}")
                    self.textEdit.append(
                        f"F1 Weighted: {sklearn.metrics.f1_score(y, y_predict, average='weighted')}")
                    self.textEdit.append(
                        "---------------------------------------------------------------------------------------------------")

                for i in range(len(X)):
                    if y[i] != y_predict[i]:
                        self.textEdit.setTextColor(redColor)
                    else:
                        self.textEdit.setTextColor(greenColor)

                    self.textEdit.append("X=%s\nTrue=%s\nPredicted=%s\t\n---------------------------------------------------------------------------------------------------\n" %
                                        (X[i], y[i], y_predict[i]))
                    # self.textEdit.append(f"{ft_model.score(X, y)}")

                    # print("X=%s,\t True=%s,\t Predicted=%s" %
                    #       (X[i], y[i], y_predict[i]))
                # print(ft_model.score(X, y))
            else:
                self.textEdit.append(
                    f"Max error: {sklearn.metrics.max_error(y, y_predict)}")

                self.textEdit.append(
                    f"Mean AE: {sklearn.metrics.mean_absolute_error(y, y_predict)}")

                self.textEdit.append(
                    f"MSE: {sklearn.metrics.mean_squared_error(y, y_predict)}")

                self.textEdit.append(
                    f"MSLE: {sklearn.metrics.mean_squared_log_error(y, y_predict)}")

                self.textEdit.append(
                    f"Median AE: {sklearn.metrics.median_absolute_error(y, y_predict)}")

                self.textEdit.append(
                    f"R2: {sklearn.metrics.r2_score(y, y_predict)}")

                self.textEdit.append(
                    "---------------------------------------------------------------------------------------------------")
                for i in range(len(X)):

                    self.textEdit.append("X=%s\tTrue=%s\tPredicted=%s\t\n" %
                                        (X[i], y[i], y_predict[i]))
                    # self.textEdit.append(f"{r2_score(y, y_predict)}")
                    # print("X=%s, True=%s, Predicted=%s" %
                    #       (X[i], y[i], y_predict[i]))
                # print(r2_score(y, y_predict))
            self.textEdit.setTextColor(blackColor)
        except ValueError:
            self.functions.popup_window(
                "The selected model is trained on a different Data Set.", " Error ", "Warning")

    def save_Checked(self):
        pass

# *^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ! 5. MODEL HISTORY SCREEN

    # ! 6. TIME SERIES MDOE SCREEN
    def min_shift_slot(self):
        global min_shift
        min_shift = self.min_timeshift_box.value()

    def max_shift_slot(self):
        global max_shift
        max_shift = self.max_timeshift_box.value()

    def max_ts_checked(self):
        global max_shift, X
        if self.checkBox.isChecked():
            self.max_timeshift_box.setEnabled(False)
            max_shift = X.shape[0]
        else:
            self.max_timeshift_box.setEnabled(True)

    def roll_slot(self):
        global X, X_rld, sort_by, y_df, y, item_index, min_shift, max_shift
        self.comprehensive_radio.setChecked(True)
        try:
            X_rld = roll_time_series(
                X, column_id="id", column_sort=f'{sort_by}', min_timeshift=int(min_shift), max_timeshift=int(max_shift), rolling_direction=1)
            print("\n_______________________________________~-Rolled Dataframe-~_______________________________________________")
            print(X_rld)
            y_df = y_df.iloc[min_shift:].to_numpy()
            y = pd.Series(
                y_df, name=f"{self.functions.getLabel(item_index, data)}")
            self.extract_frame.setEnabled(True)
            self.roll_frame.setEnabled(False)
            self.back_btn_7.hide()
        except KeyError:
            self.functions.popup_window(
                "Minimum timeshift cannot be greater than Maximum timeshift", " Error ", "Warning")

    def extract_slot(self):
        global X, X_rld, y, sort_by, fc_settings, extracted_df
        try:
            X = extract_features(
                X_rld, column_id='id', column_sort=f'{sort_by}', default_fc_parameters=fc_settings)
            print(X)
            X.reset_index(drop=True, inplace=True)
            impute(X)
            X = select_features(X, y, show_warnings=False)
            if X.empty:
                self.functions.popup_window(
                    "Zero features were extracted. Change your preferences.", " Error ", "Warning")
            else:
                self.next_btn_7.setEnabled(True)
            print("x"*50)
            print(X)
            print("x"*50)
            print(type(y))
            y_frame = y.to_frame()
            y_frame.columns = ['Target']
            print(type(y_frame))
            print("y"*50)
            print(y_frame)
            frames = [X, y_frame]
            extracted_df = pd.concat(frames, axis=1)

            print("y"*50)

        except NameError:
            self.functions.popup_window(
                "Please Choose Valid Parameters", " Error ", "Warning")

    def save_pred(self):
        global extracted_df
        # frames = [X, y_frame]
        # bigdata = pd.concat(frames,axis=1)
        # print(bigdata)
        # !Export me file browser gia save as:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None, "Save File As", "",
                                                  "CSV Files (*.csv);", options=options)

        # print(type(y_frame))
        if filename:
            extracted_df.to_csv(filename, index=False)
            # original_stdout = sys.stdout
            # with open(filename+".csv", 'w') as f:
            #     # https://www.learnpyqt.com/tutorials/example-browser-file-operations/
            #     sys.stdout = f
            #     print(X)
            #     sys.stdout = original_stdout
        # self.functions.popup_window(
        #     "Dataset exported successfully.", " Success ", "Information")

    def next_slot_7(self):
        global y, preview_num, X, learning_type
        global fc_settings
        header_labels_X = list(X.columns.values)
        header_labels_y = list([f"{y.name}"])
        fc_settings = None
        learning_type = "Classification"
        self.stackedWidget.setCurrentIndex(9)
        self.target_preview.setRowCount(preview_num)
        self.target_preview.setColumnCount(1)

        for i in range(preview_num):
            self.target_preview.setItem(
                i, 0, QTableWidgetItem(f"{ y.iloc[i]}"))

        # dimiourgia table me ta dedomena tou dataset gia preview
        self.features_preview.setRowCount(preview_num)  # set row Count
        self.features_preview.setColumnCount(
            self.functions.colCount(X))  # set column count
        for i in range(preview_num):
            for j in range(self.functions.colCount(X)):
                self.features_preview.setItem(
                    i, j, QTableWidgetItem(f"{ X.iloc[i][j] }"))
        self.features_preview.setHorizontalHeaderLabels(header_labels_X)
        self.target_preview.setHorizontalHeaderLabels(header_labels_y)

        print("next-slot-7")

    def back_slot_7(self):
        self.stackedWidget.setCurrentIndex(2)
        self.sort_box.clear()
        print("back-slot-7")

    def next_slot_8(self):
        global learning_type
        self.stackedWidget.setCurrentIndex(4)
        print("next-slot-8")

    def back_slot_8(self):
        self.stackedWidget.setCurrentIndex(8)
        print("back-slot-8")

    def sort_col_slot(self):
        global sort_idx, sort_by
        sort_idx = self.sort_box.currentIndex()
        sort_by = self.sort_box.itemText(sort_idx)
        self.roll_btn.setEnabled(True)

    def submit_slot(self):
        global fc_settings
        try:
            fc_settings = ast.literal_eval(self.text_settings.toPlainText())
            print(fc_settings)
        except ValueError:
            fc_settings = None
            self.functions.popup_window(
                "Please, Insert a dictionary format. - > fc_settings are set to None.", " Error ", "Warning")

    def comprehensive_slot(self):
        global fc_settings
        self.submit_btn.setEnabled(False)

        if self.comprehensive_radio.isChecked():
            fc_settings = ComprehensiveFCParameters()
            print(fc_settings)

    def minimal_slot(self):
        global fc_settings
        self.submit_btn.setEnabled(False)

        if self.minimal_radio.isChecked():
            fc_settings = MinimalFCParameters()
            print(fc_settings)

    def efficient_slot(self):
        global fc_settings
        self.submit_btn.setEnabled(False)

        if self.minimal_radio.isChecked():
            fc_settings = EfficientFCParameters()
            print(fc_settings)

    def custom_slot(self):
        self.submit_btn.setEnabled(True)
        if self.custom_radio.isChecked():
            self.text_settings.setReadOnly(False)
        else:
            self.text_settings.setReadOnly(True)


# ! TINY MACHINE LEARNING FUNCTIONS !


    def tiny_slot(self):

        self.stackedWidget.setCurrentIndex(10)

    def set_split(self):
        global split_size
        self.split_box.setSingleStep(0.01)
        split_size = self.split_box.value()

    def set_folds(self):
        pass

    def micro_model(self):

        global X, y, model_h, split_size

        X_train, X_test, y_train, y_test = train_test_split(
            X, np.ravel(y),
            test_size=split_size, random_state=101)

        # * model 2: RFCLASSIFIER TRAIN HPTUNE FIT

        model_1 = RandomForestClassifier()
        model_1.fit(X_train, y_train)
        # y_pred = model_1.predict(X_test)
        # Number of trees in random forest
        n_estimators = [int(x)
                        for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        model_1 = RandomizedSearchCV(model_1, param_distributions=random_grid,
                                     n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

        model_1.fit(X_train, y_train)
        y_pred = model_1.best_estimator_.predict(X_test)
        acc1 = metrics.accuracy_score(y_test, y_pred)
        # print(rf_random.best_estimator_)
        # print('ACC : ', metrics.accuracy_score(y_test, y_pred))

        # * model 2: SVM TRAIN HPTUNE FIT
        model_2 = SVC()
        model_2.fit(X_train, y_train)
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}
        model_2 = RandomizedSearchCV(
            model_2, param_grid, refit=True, verbose=3)
        model_2.fit(X_train, y_train)

        predictions = model_2.best_estimator_.predict(X_test)
        acc2 = metrics.accuracy_score(y_test, predictions)
        print(f"{acc1} vs {acc2}")
# print prediction results
        if acc1 > acc2:
            model_h = model_1
        else:
            model_h = model_2
        print(model_h)
        # self.header_text.setText('model')
        self.frame_export.setEnabled(True)
        self.frame_pref.setEnabled(False)

    def home_micro(self):
        self.stackedWidget.setCurrentIndex(0)  # Na ksekinaei apo 1h othoni

    def export_h(self):
        global model_h
        # !Export me file browser gia save as:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None, "Save File As", "",
                                                  "C Header Files (*.h);", options=options)
        if filename:
            original_stdout = sys.stdout
            with open(filename+".h", 'w') as f:
                # https://www.learnpyqt.com/tutorials/example-browser-file-operations/
                sys.stdout = f
                print(port(model_h.best_estimator_))
                sys.stdout = original_stdout

    # model_h_name = self.header_text.toPlainText()
    # original_stdout = sys.stdout  # Save a reference to the original standard output
    # with open('/home/larry/Documents/projectv1/'+model_h_name+'.h', 'w') as f:
    #     # Change the standard output to the file we created.
    #     sys.stdout = f
    #     print(port(model_h.best_estimator_))
    #     sys.stdout = original_stdout  # Reset the standard output to its original value
        self.functions.popup_window(
            "Model exported successfully.", " Success ", "Information")


def main():
    CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(open('theme.qss').read())
    font_path = os.path.join(
        CURRENT_DIRECTORY, "Font_fold", "Quicksand-Regular.ttf")
    _id = QtGui.QFontDatabase.addApplicationFont(font_path)

    # font_path = os.path.join(CURRENT_DIRECTORY, "Roboto", "Quicksand-Bold.ttf")
    # _id = QtGui.QFontDatabase.addApplicationFont(font_path)
    font = QtGui.QFont("Quicksand", 10)

    app.setFont(font)

    # app.setStyle('Fusion')
    # app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    app.setWindowIcon(QtGui.QIcon('./img/icon.png'))

    ex = MainWindowUIClass()
    MainWindow = QtWidgets.QMainWindow()
    ex.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
