# Edw tha uparxei to file pou tha ginei load kai ta periexomena tou
# ola tha emfanizontai sto GUI (logika)
# yolo1

import csv
import sys
import sqlite3
import pandas as pd
import sklearn
import autosklearn
import autosklearn.metrics as mtc
import numpy as np

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from GUI import Ui_MainWindow

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


class Func:
    def __init__(self):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.fileName = None

    def isValid(self, fileName):  # checks if file exists
        try:
            file = open(fileName, 'r')
            file.close()
            return True
        except:
            return False

    def setFileName(self, fileName):
        if self.isValid(fileName):
            self.fileName = fileName
        else:
            self.fileName = ""

    def getFileName(self):
        return self.fileName

    def readFile(self, fileName):
        df = pd.read_csv(fileName, sep="[,;]",
                         engine='python')  # Read DataFrame
        return df

    def pickTarget(self, tar_idx, df):
        target = df.iloc[:, (tar_idx)]
        return target

    def getLabel(self, tar_idx, df):
        label = df.columns[tar_idx]
        return label

    def pickPredictors(self, tar_idx, df):
        # Predictor Variables (All except the Target one.)
        df = df.iloc[:, df.columns != df.columns[tar_idx]]
        columns = df.columns.values
        for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]

            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x+=1

                df[column] = list(map(convert_to_int, df[column]))
        return df

    def rowCount(self, df):
        return df.shape[0]

    def colCount(self, df):
        return df.shape[1]
# todo : na ftiaksw to function

    def hasHeader(self, df):
        pass

    # nees sunartiseis

    def splitData(self, pred, target, test_sz):
        return train_test_split(pred, target, test_size=test_sz, random_state=1)

    def callClassifier(self, t_left, inc_est, disable_prepro, resample, resample_args, metric_var, ens_size, meta_dis):
        automl = AutoSklearnClassifier(
            initial_configurations_via_metalearning=meta_dis,

            delete_output_folder_after_terminate=False,

            delete_tmp_folder_after_terminate=False,
            # TIME RESTRICTION
            time_left_for_this_task=t_left,

            # MEMORY RESTRICTION
            ensemble_size=ens_size,
            # ALGORITHM RESTRICTION
            include_estimators=inc_est,

            # APENERGOPOIHSH PREPROSSESORS
            include_preprocessors=disable_prepro,

            # RESAMPLING (CROSS VALIDATION) - ISWS ME NEA SUNARTHSH GIA NA PAIRNEI KI AUTO PARAMETROUS
            resampling_strategy=resample,
            resampling_strategy_arguments=resample_args,

            # EPILOGI METRIKWN
            metric=metric_var
        )
        return automl
#! call regressor

    def callRegressor(self, t_left, inc_est, disable_prepro, resample, resample_args, metric_var, ens_size, meta_dis):
        automl = AutoSklearnRegressor(

            initial_configurations_via_metalearning=meta_dis,
            delete_output_folder_after_terminate=False,

            delete_tmp_folder_after_terminate=False,

            # TIME RESTRICTION
            time_left_for_this_task=t_left,

            # MEMORY RESTRICTION
            ensemble_size=ens_size,
            # ALGORITHM RESTRICTION
            include_estimators=inc_est,

            # APENERGOPOIHSH PREPROSSESORS
            include_preprocessors=disable_prepro,

            # RESAMPLING (CROSS VALIDATION) - ISWS ME NEA SUNARTHSH GIA NA PAIRNEI KI AUTO PARAMETROUS
            resampling_strategy=resample,
            resampling_strategy_arguments=resample_args,

            # EPILOGI METRIKWN
            metric=metric_var


        )
        return automl

    # isws sinartisi na epilegei classifier i regressor meta
    def fitModel(self, pred_train, target_train, automl, d_name):
        automl = automl.fit(pred_train, target_train, dataset_name=d_name)
        return automl

    def app_Estimator(self, inc_est, box_state, est_name):
        if box_state:
            if not est_name in inc_est:
                inc_est.append(est_name)
        else:
            if est_name in inc_est:
                inc_est.remove(est_name)
        return(inc_est)

    def store_model(self, filename, model, model_name, l_type):
        pickled_model = pickle.dumps(model)
        #! INSERT
        learning_type = l_type
        insertion_time = datetime.now()
        conn = sqlite3.connect('models.db')
        query = 'insert into models values (?, ?, ?, ?)'
        # todo: elegxos gia to an den dwthei name
        conn.execute(query, [model_name, filename,
                             insertion_time, learning_type])
        conn.commit()

    def load_model(self, id):
        #! connect to database
        conn = sqlite3.connect('models.db')
        #!SELECT model
        cursor = conn.execute(f'select * from models where timestamp = "{id}"')
        name, pickled_model, timestamp, learning_type = cursor.fetchone()
        # fetched_model = pickle.loads(pickled_model)
        try:
            fetched_model = pickle.load(open(pickled_model,'rb'))
            return fetched_model
        except:
            cursor = conn.execute(f'delete from models where timestamp = "{id}"')
            conn.commit()
            print("Model Not Found... Row Deleted.")


    def fill_tables(self, db_name, query, query_cnt, table):
        conn = sqlite3.connect(db_name)

        # QUERIES
        cursor = conn.execute(f"{query}")
        curs1 = conn.execute(f"{query_cnt}")
        tbl_rowcount = curs1.fetchone()
        table.setRowCount(tbl_rowcount[0])
        table.setColumnCount(2)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        # DATABASE VIEW
        row = 0
        while True:
            form = cursor.fetchone()
            if form == None:
                break
            for column, item in enumerate(form):
                table.setItem(
                    row, column, QTableWidgetItem(f'{item}'))
            row += 1

    def popup_window(self,text,title,type):
        if type == "Warning":
            popup = QtWidgets.QMessageBox()
            popup.setWindowTitle(f"{title}")
            popup.setText(
                f"{text}")
            popup.setStandardButtons(QtWidgets.QMessageBox.Retry)
            popup.setIcon(QtWidgets.QMessageBox.Warning)
            popup.setDefaultButton(QtWidgets.QMessageBox.Retry)
            popup.exec()
        elif type == "Information":
            popup = QtWidgets.QMessageBox()
            popup.setText(f"{text}")
            popup.setWindowTitle(f"{title}")
            popup.setIcon(QtWidgets.QMessageBox.Information)
            popup.setStandardButtons(QtWidgets.QMessageBox.Close)
            popup.setDefaultButton(QtWidgets.QMessageBox.Close)
            popup.exec_()
            
# def getScore(self, automl, p_test, t_test):
#     pred = automl.predict(p_test)
#     return sklearn.metrics.accuracy_score(t_test, pred)
