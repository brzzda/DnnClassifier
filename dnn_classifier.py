import os
import sys
import json
import pandas as pd
import csv
import tensorflow as tf

from tensorflow import feature_column
from sklearn.model_selection import train_test_split


def df_to_dataset(features, labels=None, training=True, repeat=False, batch_size=256):
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        if repeat:
            dataset = dataset.shuffle(buffer_size=1000).repeat()
        else:
            dataset = dataset.shuffle(buffer_size=len(features))
    return dataset.batch(batch_size)


def load_json(file_path, raise_exception=True, verbose=False):
    if not os.path.exists(file_path):
        if raise_exception:
            raise FileNotFoundError("HelperFunctions.load_json: file '" + file_path + " does not exist.")
        print("HelperFunctions.load_json: file '" + file_path + " not found.")
        return False
    with open(file_path) as file:
        res = json.load(file)
        if verbose:
            print("HelperFunctions.load_json: file '" + file_path + "' LOADED.")
        return res


def load_dnn_classifier(feature_columns, model_path_dir):
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[50, 30],
        # dropout=0.1,
        model_dir=model_path_dir,
        n_classes=2
    )
    return dnn_classifier


class DnnClassifier:
    MODEL_DIR_PATH = r".\model_dir"
    CLASSES_RECORD = {'ma': 0, 'hma': 0, 'he': 0, 'od': 0, 'fov': 0, 'cw': 0}  # retrieved from data
    # CLASSES_RECORD = {'ma': 0, 'hma': 0, 'he': 0}

    def __init__(self, data_folder_path: str):
        self.data_folder_path = os.path.join(data_folder_path, "data_json")
        self.training_data_file_path = os.path.join(data_folder_path, "train.csv")
        self.test_data_file_path = os.path.join(data_folder_path, "test.csv")
        self.feature_columns = []
        for fc in self.CLASSES_RECORD.keys():
            self.feature_columns.append(feature_column.numeric_column(fc))
        if not os.path.exists(self.MODEL_DIR_PATH):
            os.mkdir(self.MODEL_DIR_PATH)

    def load_data(self, only_testing=False):
        self.only_testing = only_testing
        self.testing_json, self.testing_ground_truth = self.load_data_from_csv(self.test_data_file_path)
        if only_testing:
            return
        self.training_json, self.training_ground_truth = self.load_data_from_csv(self.training_data_file_path)

    def load_data_from_csv(self, csv_data_path):
        res_data = []
        res_ground_truth = []
        with open(csv_data_path) as csv_f:
            csv_reader = csv.reader(csv_f, delimiter=",")
            csv_f.seek(0)
            next(csv_reader)
            for i, row in enumerate(csv_reader):
                ground_truth = row[1]
                json_file_name = os.path.join(self.data_folder_path, row[2].split('.')[0] + ".json")
                train_data = load_json(json_file_name)
                res_data.append(train_data)
                res_ground_truth.append(ground_truth)
        print("Data '{}' Loaded".format(csv_data_path))
        return res_data, res_ground_truth

    def prepare_data(self):
        if not self.only_testing:
            self.training_data = self.prepare_data_from_json(self.training_json, self.training_ground_truth)
            self.training_data, self.validation_data = train_test_split(self.training_data, test_size=0.2)
            training_data = self.training_data.copy()
            validation_data = self.validation_data.copy()
            training_labels = training_data.pop('result')
            validation_labels = validation_data.pop('result')
            self.train_ds = df_to_dataset(training_data, training_labels)
            self.val_ds = df_to_dataset(validation_data, validation_labels, training=False)

        self.testing_data = self.prepare_data_from_json(self.testing_json, self.testing_ground_truth)
        testing_data = self.testing_data.copy()
        testing_labels = testing_data.pop('result')
        self.test_ds = df_to_dataset(testing_data, testing_labels, training=False)
        # self.feature_columns = []
        # for fc in self.training_data.columns:
        #     if fc == 'result':
        #         continue
        #     self.feature_columns.append(feature_column.numeric_column(fc))
        print('Data preprocessed.')

    def prepare_data_from_json(self, json_data, ground_truth):
        full_record_values = []
        for i, record in enumerate(json_data):
            gt = 0 if ground_truth[i] == "No DR" else 1
            classes_record = self.from_json_2_data_list(record, ground_truth=gt)
            full_record_values.append(classes_record)
        result = pd.DataFrame(full_record_values)
        return result

    def from_json_2_data_list(self, json_file, ground_truth=None):
        result = self.CLASSES_RECORD.copy()
        # counts = self.classes_record.copy()
        if ground_truth is not None:
            result['result'] = ground_truth
        for polygon_data in json_file['polygon_objects']:
            cn = polygon_data['class_name']
            # if cn in result and polygon_data['class_score'] > 0.7:
            if cn in result:
                # counts[cn] += 1
                # result[cn] += polygon_data['class_score']
                result[cn] += 1
        # for key in counts.keys():
        #     if counts[key] != 0:
        #         result[key] /= counts[key]
        return result

    def create_dnn_classifier(self):
        model_data_files = os.listdir(self.MODEL_DIR_PATH)
        for model_data_file in model_data_files:
            os.remove(os.path.join(self.MODEL_DIR_PATH, model_data_file))
        self.dnn_classifier = load_dnn_classifier(self.feature_columns, self.MODEL_DIR_PATH)
        print("DNN classifier created.")

    def load_dnn_classifier(self):
        self.dnn_classifier = load_dnn_classifier(self.feature_columns, self.MODEL_DIR_PATH)
        print("DNN classifier loaded")

    def train_dnn_classifier(self, *args, **kwargs):
        self.checkpoint_counter = 0
        train_data = self.training_data.copy()
        labels = train_data.pop('result')
        print("Training classifier...")
        self.dnn_classifier.train(
            input_fn=lambda: df_to_dataset(train_data, labels, repeat=True),
            steps=5000,
        )
        print("dnn classifier trained.")

    def evaluate_dnn(self, *args, **kwargs):
        test_data = self.testing_data.copy()
        labels = test_data.pop('result')
        self.evaluation = self.dnn_classifier.evaluate(input_fn=lambda: df_to_dataset(test_data, labels, training=False, repeat=True))
        print(self.evaluation)

    def predict_dnn(self):
        test_data = self.testing_data.copy()
        labels = test_data.pop('result')
        predictions = self.dnn_classifier.predict(input_fn=lambda: df_to_dataset(test_data, training=False, repeat=True))
        res = [0, 0]
        for i, pred_dict in enumerate(predictions):
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            class_name = 'DR' if class_id == 1 else 'No DR'
            if class_name == self.testing_ground_truth[i]:
                res[0] += 1
            else:
                res[1] += 1
        self.dnn_accuracy = res[0]/sum(res)
        print("dnn accuracy: " + str(self.dnn_accuracy))

    def predict_dnn_from_json(self, json_file):
        if isinstance(json_file, str):
            if os.path.exists(json_file):
                json_file = load_json(json_file)
            elif os.path.exists(os.path.join(self.data_folder_path, json_file)):
                json_file = load_json(os.path.join(self.data_folder_path, json_file))
            else:
                print("file '{}' does not exist.".format(json_file))
                return
        result = []
        def get_class_name(class_id):
            return "DR" if class_id == 1 else "No Dr"

        data = self.from_json_2_data_list(json_file)
        data_frame = pd.DataFrame([data])
        prediction = self.dnn_classifier.predict(input_fn=lambda: df_to_dataset(data_frame, training=False, repeat=True))
        for pred_dict in prediction:
            class_id_1 = pred_dict['class_ids'][0]
            class_id_2 = 0 if class_id_1 == 1 else 1
            probability_1 = pred_dict['probabilities'][class_id_1]
            probability_2 = pred_dict['probabilities'][class_id_2]
            result.append(((get_class_name(class_id_1), probability_1), (get_class_name(class_id_2), probability_2)))
        return result


if __name__ == '__main__':
    main = DnnClassifier(sys.argv[1])
    if sys.argv[2] == '-l':
        main.load_data()
        main.prepare_data()
        main.create_dnn_classifier()
        main.train_dnn_classifier()
    elif sys.argv[2] == "-e":
        main.load_data(only_testing=True)
        main.prepare_data()
        main.load_dnn_classifier()
        main.predict_dnn()
    elif sys.argv[2] == '-s':
        main.load_data(only_testing=True)
        main.prepare_data()
        main.load_dnn_classifier()
        print(main.predict_dnn_from_json(sys.argv[3]))
