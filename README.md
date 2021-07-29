# DnnClassifier
An example of a tensorflow DNN classifier

# Introduction
This is a simple DNN classifier examle uploaded due to existing assignment for the Aireen company. 
The script is capable of training and storing the DNN clasifier, then loading and evaluating the existing classifier and last loading an existing classifier with specified json file used as input to which the classifier prints out an output of a classifier. 

# Usage
First the model must be trained. The trained model is automatically saved and used during the prediction or evaluation of the model. 

There are 2, optionally 3 arguments expected: 
1. Path to the data folder as specified in the assignment
2. Flag specifying whether we want to 
 - train the classifier:  "-l"
 - evaluate the classifier: "-e"
 - use the classifier for prediction with specified input "-s"
3. If the "-s" flag is used the last argument is either full path to the json file that is used as input or only the json file name if it is located in the "data_json" folder located in the original data folder (which is passed as the first argument).

# Examples of usage
python dnn_classifier.py "[path_to_the_data_folder]" -l 

python dnn_classifier.py "[path_to_the_data_folder]" -s "[json_file_name]"
