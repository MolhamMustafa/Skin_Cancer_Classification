# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 12:25:33 2021

@author: molha
"""
# =========================================================================
# Import Main Libraries
# =========================================================================

##
# Import Required Libraries
#

import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob # import images
from PIL import Image # load images

import seaborn as sns
np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential # To create the model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats # For statistics purpose
from sklearn.preprocessing import LabelEncoder # Convert lables into numbers
import datetime
import statistics
        


# =========================================================================
# Importing the data
# =========================================================================

##
# Data upload class combines the csv file along with the image.jpg files in one dataframe
#
class Data_Upload:
    
    ##
    # Define the required paths and the start time for importing the data
    #
    def __init__(self):
        self.starttime = datetime.datetime.now()
        self.main_path = os.path.join('D:/University of Huddersfield','9) Individual Project','Potential Data','Skin Cancer','Data')
        self.csv_path = os.path.join('D:/University of Huddersfield','9) Individual Project','Potential Data','Skin Cancer','Data','HAM10000_metadata.csv')
        self.image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(self.main_path, '*', '*.jpg'))}

    ##
    # Read the csv file as dataframe (pandas)
    #
    def Read_CSV(self):
        self.stepstarttime = datetime.datetime.now()
        self.csv_df = pd.read_csv(self.csv_path)
        print()
        print('csv data imported successfully')
        print('**Below is the data head**')
        print(self.csv_df.head())
        print('time taken for this step is ', datetime.datetime.now() - self.stepstarttime)
        print()
        print()
        
        
    ##
    # Import the path of each image to the dataframe (csv_df)
    #
    def Get_Image_Paths(self):
        self.stepstarttime = datetime.datetime.now()
        self.csv_df['ImagePaths'] = self.csv_df['image_id'].map(self.image_path.get)
        print()
        print('The path for each image is appended to the dataframe in column "ImagePaths"')
        print('**Below is the data head**')
        print(self.csv_df.head())
        print('time taken for this step is ', datetime.datetime.now() - self.stepstarttime)
        print()
        print()
        
    
    ##
    # Append image to new colomn in the dataframe (image dimentions = 32 * 32)
    #
    def Import_Images_to_DataFrame(self):
        self.stepstarttime = datetime.datetime.now()
        self.csv_df['image'] = self.csv_df['ImagePaths'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))
        print()
        print('Images imported to "image" column with dimentions 32*32')
        print('**Below is the data head**')
        print(self.csv_df.head())
        print('time taken for this step is ', datetime.datetime.now() - self.stepstarttime)
        print()
        print()
        
        
    ##
    # Save the created DataFrame in new csv file in the same folder
    #
    def Save_Data(self):
        self.stepstarttime = datetime.datetime.now()
        self.savepath = os.path.join('D:/University of Huddersfield','9) Individual Project','Coding','Full_Data.csv')
        self.csv_df.to_csv(self.savepath, index = False)
        print()
        print('Dataframe is saved in ', self.savepath)
        print('time taken for this step is ', datetime.datetime.now() - self.stepstarttime)
        print()
        print()
    
    ##
    # Recalling the data
    # to get the whole colums make the column variable = 0
    # to get all data except lesion_id and image_path make columns = 1
    #   
    def recall_dataframe(self, columns):
        if columns == 0:
            return self.csv_df
        elif columns == 1:
            return self.csv_df[self.csv_df.columns.values[[1,2,3,4,5,6,8]]]
        else:
            return self.csv_df[[columns]]
    
    ##
    # Print sample images for each lesion
    #
    def Sample_Images(self):
        self.n_samples = 5 # set gte number of samples per lesion type
        # Plot the sample
        fig, self.m_axs = plt.subplots(7, self.n_samples, figsize = (4* self.n_samples, 3*7))
        for self.n_axs, (type_name, type_rows) in zip(self.m_axs, 
                                         self.csv_df.sort_values(['dx']).groupby('dx')):
            self.n_axs[0].set_title(type_name)
            for c_ax, (_, c_row) in zip(self.n_axs, type_rows.sample(self.n_samples, random_state=1234).iterrows()):
                c_ax.imshow(c_row['image'])
                c_ax.axis('off')
    
                
DataUpload = Data_Upload() # the main class name
DataUpload.Read_CSV() # read the csv file
DataUpload.Get_Image_Paths() # append the path of each image to the csv dataframe
DataUpload.Import_Images_to_DataFrame() # append the matrix for each image (32*32) to the main dataframe
#DataUpload.Save_Data() # save the full data in one csv file
#csv_df = DataUpload.recall_dataframe(0) # Store the dataframe in a new object
DataUpload.Sample_Images() #Print 5 sample images for each lesion type



# =========================================================================
# Data Analysis & Deep Dive
# =========================================================================

Figures_path = r'D:\University of Huddersfield\9) Individual Project\Coding\Figures'

##
# Data Exploration and Descriptive Analysis
#
class Descriptive_Analysis:
    
    ##
    # provides data exploration (Nulls, duplicates, data type, initial statistics)
    # in case of numerical column it will return main statistics
    # in case of categorical it will return frequency table
    # in case of all columns selected (0 or1) it will return data summary
    #
    def Data_Exploration(self,column):
        if column == 1 or column == 0:
            print()
            print('Data Summary')
            return DataUpload.recall_dataframe(column).info()
        else:
            if str(DataUpload.recall_dataframe(column).dtypes[0]) != 'object':
                print()
                print('Main Statistics for', column)
                print(DataUpload.recall_dataframe(column).describe())
                print()
                print('Number of true duplications vs. false duplications')
                return DataUpload.recall_dataframe(column).duplicated().value_counts()
            else:
                print()
                print('Frequency Table for', column)
                print(DataUpload.recall_dataframe(column).value_counts())
                print()
                print('Number of true duplications vs. false duplications')
                return DataUpload.recall_dataframe(column).duplicated().value_counts()
            
    ##
    # Descriptive analysis graph
    # deal with numeric data only
    # to get the overall column plots without filters, set the 'filters' = 0
    # to get the automatic graph title, set "graph_title" = 0
    # to avoid saving the image in local drive, set "save_folder_path" = 0
    # NOTE: enter the folder path like this --> r'path'
    #
    def Des_Graphs_Num (self, column, filters, graph_title, save_folder_path):
        if column == 0 or column == 1:
            print('This fuction acts on a single feature only')
        else:
            if str(DataUpload.recall_dataframe(column).dtypes[0]) != 'object':
                if filters == 0:
                    print()
                    print(column, 'Main Statistics')
                    print(DataUpload.recall_dataframe(column).describe())
                    print()
                    print(column, 'Distance Plot')
                    
                    # Set the graph title
                    if graph_title == 0:
                        self.title = column + ' Distance Plot'
                    else:
                        self.title = graph_title
                    
                    sns.distplot(DataUpload.recall_dataframe(column), color= 'green',
                    hist_kws=dict(edgecolor="black", linewidth=1))
                    plt.title(self.title)
                    
                    # Set the save condition
                    if save_folder_path != 0:
                        plt.savefig(save_folder_path + '\\{v} distplot.png'.format(v = column), dpi=300)
                        print('Graph saved undel the title:', '\\{v} distplot.png'.format(v = column))

                        
                else:
                    self.filter_items = DataUpload.recall_dataframe(0)[filters].unique()
                    for f in self.filter_items:
                        print()
                        print(f ,column, 'Main Statistics')
                        print(DataUpload.recall_dataframe(0)[DataUpload.recall_dataframe(0)[filters] == f][column].describe())
                        print()
                        print(column, 'Distance Plot')
                        
                        # Set the graph title
                        if graph_title == 0:
                            self.title = f + ' ' + column + ' Distance Plot'
                        else:
                            self.title = f + graph_title
                        
                        sns.distplot(DataUpload.recall_dataframe(0)[DataUpload.recall_dataframe(0)[filters] == f][column], color= 'green',
                        hist_kws=dict(edgecolor="black", linewidth=1))
                        plt.title(self.title)
                        
                        # Set the save condition
                        if save_folder_path != 0:
                            plt.savefig(save_folder_path + '\\{fil} {v} distplot.png'.format(fil = f, v = column), dpi=300)
                        plt.clf()
            else:
                print('for categorical features, please use "Des_Graphs_Cat"')
                
                
    ##
    # Descriptive analysis for categorical data
    # Same features as the previous one, but deals with categorical data
    #
    def Des_Graphs_Cat(self, column, filters, graph_title, save_folder_path, Title_Font_Size):
        
        # define the graph lables
        self.labels = DataUpload.recall_dataframe(1)[column].value_counts().index.tolist()
        
                # Set the graph title
        if graph_title == 0:
            self.title = column + ' Frequency'
        else:
            self.title = graph_title
        
        ##
        # Bar chart
        #
        
        # auto set x labels fontsize
        if len(self.labels) > 7:
            mpl.rcParams['font.size'] = 7.0
            f, ax = plt.subplots(figsize=(13,5))
        plt.bar(self.labels,DataUpload.recall_dataframe(1)[column].value_counts(), color = 'green', ec = 'black')
        for i in range(len(self.labels)):
            plt.text(i,DataUpload.recall_dataframe(1)[column].value_counts()[i], DataUpload.recall_dataframe(1)[column].value_counts() [i],
             ha = 'center', va = 'bottom')
        #plt.ylim([0,7500])
        plt.title(self.title, fontsize = Title_Font_Size)
        plt.savefig(save_folder_path + '\\{v} frequency.png'.format(v = column), dpi=300)
        print('Graph saved undel the title:', '\\{v} frequency.png'.format(v = column))
        plt.clf()

        ##
        # Pie chart
        #
        
        # auto set the lable sizes
        if len(self.labels) > 7:
            mpl.rcParams['font.size'] = 6.0 # set the test size
        plt.pie(DataUpload.recall_dataframe(1)[column].value_counts(), labels = self.labels, autopct='%1.1f%%')
        plt.title(self.title + ' Pie Chart', fontsize = Title_Font_Size)
        #plt.title('Proportion of each Localization')
        plt.savefig(save_folder_path + '\\{v} pie chart.png'.format(v = column), dpi=300)
        print('Graph saved undel the title:', '\\{v} pie chart.png'.format(v = column))


    ##
    # Doing correlation matrix for all the data features
    #
    
    def Overall_Correlations(self, save_folder_path):
        
        # create the correlation dataset (subset from the original data)
        self.corr_df = DataUpload.recall_dataframe(1)[['dx', 'dx_type', 'age', 'sex','localization']]
        
        self.header_list = self.corr_df.columns.values.tolist()
        self.header_list.remove('age')
        
        # convert categorical variables into numeric
        le = LabelEncoder()
        for i in self.header_list:
            le.fit(self.corr_df[i])
            LabelEncoder()
            self.corr_df[i + '_label'] = le.transform(self.corr_df[i])
            print(i, 'encoded successfully')
        
        # create the correlation matrix
        self.corr_df = self.corr_df.drop(self.header_list, axis = 1)
        self.new_heads = self.corr_df.columns.values.tolist()
        self.new_heads = self.corr_df.columns[1:].str.split('_label')
        self.new_headers = ['age']
        for i in range(len(self.new_heads)):
            self.new_headers.append(self.new_heads[i][0])
            
        self.corr_df.columns = self.new_headers # adjsut the table column names
        
        # Drow the graph
        self.corr_mat = sns.heatmap(self.corr_df.corr(), annot = True
                    ,cmap = "YlGnBu")
        self.corr_mat.set_yticklabels(self.corr_mat.get_yticklabels(), rotation=45)
        if save_folder_path != 0:
            plt.savefig(save_folder_path + '\\Overall Corr. Matrix.png', dpi=300)
        plt.clf()
            
    ##
    # Correlation between each lesion type and the localization
    #
    
    def dx_localization_Correlations (self, save_folder_path):
        # Create the dummy variables for localization and dx
        self.localization_dx = pd.get_dummies(DataUpload.recall_dataframe(1)[['dx','localization']])
        
        # create the adjusted dx headers
        self.names = self.localization_dx.columns.str.split('dx_')
        self.names = self.names.tolist()
        self.new_names = []
        
        for i in range(len(self.names)):
            if len(self.names[i]) == 2:
                self.new_names.append(self.names[i][1])
                print(self.names[i],'adjusted and added successfully')
        
        # Create the adjusted localization headers
        self.names = self.localization_dx.columns.str.split('localization_')
        self.names =self. names.tolist()
        
        for i in range(len(self.names)):
            if len(self.names[i]) == 2:
                self.new_names.append(self.names[i][1])
                print(self.names[i],'adjusted successfully')
        
        # Replace the localization_dx column names
        self.localization_dx.columns = self.new_names
        
        
        # create the confusion matrix
        # Full matrix
        self.corrMatrix = self.localization_dx.corr()
        sns.heatmap(self.corrMatrix, annot=True, annot_kws={"size": 3})
        if save_folder_path != 0:
            plt.savefig(save_folder_path + '\\Loc dx Corr matrix.png', dpi=300)
        plt.clf()
        
        # Shrinked matrix
        self.new_corrMatrix = self.corrMatrix[DataUpload.recall_dataframe(1).dx.unique()]
        self.new_corrMatrix = self.new_corrMatrix.loc[DataUpload.recall_dataframe(1).localization.unique()]
        self.loc_dx_corr = sns.heatmap(self.new_corrMatrix, annot=True, 
                                  annot_kws={"size": 6}, cmap="YlGnBu")
        
        self.loc_dx_corr.set_yticklabels(self.loc_dx_corr.get_yticklabels(), rotation=45)
        if save_folder_path != 0:
            plt.savefig(save_folder_path + '\\Shrink loc dx Corr matrix.png', dpi=300)
        
        # Print the max and min correlations for each lesion type
        for i in self.new_corrMatrix:
            print('max % in', i, 'is', max(self.new_corrMatrix[i]),
                  'and min % is', min(self.new_corrMatrix[i]))
        


DesAnalysis = Descriptive_Analysis()
DesAnalysis.Data_Exploration('localization')
DesAnalysis.Des_Graphs_Num('age','sex' ,0, Figures_path) # Descriptive analysis graph for numerical features
DesAnalysis.Des_Graphs_Cat('dx_type',0 ,0, Figures_path, 8) # # Descriptive analysis graph for categorical features
DesAnalysis.Overall_Correlations(Figures_path) # Overall correlation matrix
DesAnalysis.dx_localization_Correlations(Figures_path) # Correlation matrix focused on dx and localization only



##
# Getting the main statistics for age after removing the 0s and null values
#
DataUpload.recall_dataframe(0)[(DataUpload.recall_dataframe(0).age != 0) &
                               (DataUpload.recall_dataframe(0).age.notnull())]['age'].describe()



 


# =========================================================================
# Data Preprocessing
# =========================================================================

# Convert the lesion type in column ['dx'] to numeric values
# ==========================================================
le = LabelEncoder()
le.fit(csv_df.dx)
LabelEncoder()
print(list(le.classes_)) # print unique valuse before encoding

csv_df['dxLabel'] = le.transform(csv_df.dx) # encode dx values
Sample_Encoded_dx = csv_df[['dx', 'dxLabel']].sample(20) # print sample
csv_df.dxLabel.value_counts()



# Balancing the data
# ===================

# Create a variable for each dx class to deal with each one separatly
dx0 = csv_df[csv_df.dxLabel == 0]
dx1 = csv_df[csv_df.dxLabel == 1]
dx2 = csv_df[csv_df.dxLabel == 2]
dx3 = csv_df[csv_df.dxLabel == 3]
dx4 = csv_df[csv_df.dxLabel == 4]
dx5 = csv_df[csv_df.dxLabel == 5]
dx6 = csv_df[csv_df.dxLabel == 6]


# Balance the data uisng sklearn resample fuction
# The number of samples for each dx class will be set as 500
# so that for dx types that have more then 500 records, the fuction will capture 500 only reandomly
# while for those that have records less than 500, the function will select the whole records the keep duplicating to reach the 500 records for each

from sklearn.utils import resample

NumberOfSamples = 500 # Set the number of required samples for each dx
dx0Balanced = resample(dx0, replace=True, n_samples = NumberOfSamples, random_state = 42)
dx1Balanced = resample(dx1, replace=True, n_samples = NumberOfSamples, random_state = 42)
dx2Balanced = resample(dx2, replace=True, n_samples = NumberOfSamples, random_state = 42)
dx3Balanced = resample(dx3, replace=True, n_samples = NumberOfSamples, random_state = 42)
dx4Balanced = resample(dx4, replace=True, n_samples = NumberOfSamples, random_state = 42)
dx5Balanced = resample(dx5, replace=True, n_samples = NumberOfSamples, random_state = 42)
dx6Balanced = resample(dx6, replace=True, n_samples = NumberOfSamples, random_state = 42)

# Concatenate the balanced dx types in a new dataset
BalancedSkinData = pd.concat([dx0Balanced, dx1Balanced, dx2Balanced, 
                              dx3Balanced, dx4Balanced, dx5Balanced,
                              dx6Balanced])

BalancedSkinData.dx.value_counts()



# =========================================================================
# Create the Initial Training and Testing Splits
# =========================================================================

#Convert dataframe column of images into numpy array
X = np.asarray(BalancedSkinData['image'].tolist())
X = X/255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y=BalancedSkinData['dxLabel'] #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem (somthing like the todummy in pandas)
#Split to training and testing. Get a very small dataset for training as we will be 
# fitting it to many potential models. 
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)


# =========================================================================
# Apply the Keras Model Using 5 Folds
# =========================================================================
modelstarttime = datetime.datetime.now()
batch_size = 16
epochs = 50
num_folds = 5
acc_per_fold = []
loss_per_fold = []

from sklearn.model_selection import KFold
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)


all_loss = [] # to fill with the loss per epoc for each fold
all_val_loss = [] # to fill with the validation loss per epoch and fold
all_acc = [] # to fill with the accuracy per epoch and fold
all_val_acc = [] # to fill with the validation accuracy per epoch and fold
y_pred = [] # to fill with the predicition values
y_pred_classes = []
y_true = []

fold_no = 0
for train, test in kfold.split(inputs, targets):
    fold_no += 1

    
    x_train = X[train]
    y_train = Y_cat[train]
    x_test = X[test]
    y_test = Y_cat[test]
    
    # Define the model architecture
    SIZE = 32 # pixels size

    num_classes = 7

    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation="relu", 
                     input_shape=(SIZE, SIZE, 3)))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))
    
    
    model.add(Conv2D(128, (3, 3),activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, (3, 3),activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))
    model.add(Flatten())
    
    model.add(Dense(32))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='Adam', metrics=['acc'])
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    foldstarttime = datetime.datetime.now()
    
    # Fit data to model
    history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test),
        epochs=epochs,batch_size = batch_size,verbose=2)
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
   
    all_loss.append(loss)
    all_val_loss.append(val_loss)
    all_acc.append(acc)
    all_val_acc.append(val_acc)
    
    y_pred.append(model.predict(x_test))
    y_true.append(np.argmax(y_test, axis = 1))
    
    
    
    print('Fold', fold_no, 'time taken was', datetime.datetime.now() - foldstarttime)

print('Whole module time taken is', datetime.datetime.now() - modelstarttime)


# Calculate the average loss per each epoch and fold
average_folds_loss = []  
for i in range(len(all_loss[0])):
    average_folds_loss.append(statistics.mean([
        all_loss[0][i], all_loss[1][i], all_loss[2][i],
        all_loss[3][i], all_loss[4][i]]))
    
# Calculate the average validation loss per each epoch and fold
average_folds_val_loss = []  
for i in range(len(all_val_loss[0])):
    average_folds_val_loss.append(statistics.mean([
        all_val_loss[0][i], all_val_loss[1][i], all_val_loss[2][i],
        all_val_loss[3][i], all_val_loss[4][i]]))
  
# Calculate the average accuract per each epoch and fold
average_folds_all_acc = []  
for i in range(len(all_acc[0])):
    average_folds_all_acc.append(statistics.mean([
        all_acc[0][i], all_acc[1][i], all_acc[2][i],
        all_acc[3][i], all_acc[4][i]]))

# Calculate the average validation accuracy per each epoch and fold
average_folds_all_val_acc = []  
for i in range(len(all_acc[0])):
    average_folds_all_val_acc.append(statistics.mean([
        all_val_acc[0][i], all_val_acc[1][i], all_val_acc[2][i],
        all_val_acc[3][i], all_val_acc[4][i]]))
    


# plot the training and validation loss at each epoch
epochs = range(1, len(average_folds_loss) + 1)
plt.plot(epochs, average_folds_loss, 'y', label='Training loss')
plt.plot(epochs, average_folds_val_loss, 'r', label='Validation loss')
plt.title('Average Training and validation loss all fold')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.savefig(charts_path + '\\Avge Training & Validation Loss per Epoch.png',
            dpi=300)

# plot the training and validation accuracy at each epoch
plt.plot(epochs, average_folds_all_acc, 'y', label='Training acc')
plt.plot(epochs, average_folds_all_val_acc, 'r', label='Validation acc')
plt.title('Average Training and validation accuracy all fold')
plt.xlabel('Epochs')
plt.ylabel('Average Accuracy')
plt.legend()
plt.savefig(charts_path + '\\Avge Training & Validation Accuracy per Epoch.png', dpi=300)


# Plot the Accuracy for each fold then average accuracy
accuracy_numbers = acc_per_fold.copy()
accuracy_numbers = accuracy_numbers + [statistics.mean(acc_per_fold)]

labels = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Average']
mpl.rcParams['font.size'] = 12.0 # set the test size
f, ax = plt.subplots(figsize=(13,5))
plt.bar(labels,accuracy_numbers, color = 'green', 
        ec = 'black')
for i in range(len(labels)):
    plt.text(i,round(accuracy_numbers[i]),
             round(accuracy_numbers[i]),
             ha = 'center', va = 'bottom')
plt.title('Accuracy %', fontsize = 14)
plt.savefig(charts_path + '\\Accuracy per Fold.png', dpi=300)



# Results Correlation
#--------------------
# Combining all y_pred_classes
y_pred_classes = []
for i in range(len(y_pred)):
    y_pred_classes.append(np.argmax(y_pred[i], axis = 1))
    
# Create correlatin matrix for accuracy per each fold
cm1 = []
cm2 = []
cm3 = []
cm4 = []
cm5 = []

cm = [cm1, cm2, cm3, cm4, cm5]
for c in range(len(cm)):
    cm[c].append(confusion_matrix(y_true[c], y_pred_classes[c]))

# Create and plot average correlation matrix for prediction accuracy
cm_avg = sum(cm1 + cm2 + cm3 + cm4 + cm5) / len(cm)

# Returen the encoded values to their original names
dx_values = BalancedSkinData[['dx','dxLabel']].drop_duplicates()
cm_avg_d = pd.DataFrame(cm_avg)
cm_avg_d.columns = dx_values.dx.tolist()
cm_avg_d.index = dx_values.dx.tolist()

cm_avg_d1 = cm_avg_d.copy()
for i in cm_avg_d.columns.values:
    for a in range(len(cm_avg_d)):
        cm_avg_d1[i][a] = round((cm_avg_d[i][a] / sum(cm_avg_d[i]))*100,1)

sns.heatmap(cm_avg_d, annot=True, annot_kws={"size": 10}, cmap="YlGnBu")
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Accuracy % per Lesion Type Confusion Matrix')
plt.savefig(charts_path + '\\ Accuracy Corr matrix.png', dpi=300)


#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm_avg) / np.sum(cm_avg, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')

# Evaluate (Apply) the model on the whole dataset
csv_df_testing = csv_df.copy()
for i in range(len(dx_values)):
    csv_df_testing[csv_df_testing.dx == dx_values.dx.unique()[i]]['dxLabel'] = dx_values[dx_values.dx == dx_values.dx.unique()[i]]['dxLabel']


X_whole = np.asarray(csv_df_testing['image'].tolist())
Y_Whole = csv_df_testing['dxLabel']
Y_cat_Whole = to_categorical(Y_Whole, num_classes=7)

scores_whole = model.evaluate(X_whole, Y_cat_Whole, verbose=1)
acc_whole = scores_whole[1] * 100
print('Accuracy after applying on the whole datadet is ', 
      round(acc_whole,1),'%', sep ='')
loss_whole = scores_whole[0]

#===========================================================================
#===========================================================================
#===========================================================================
#===========================================================================
#===========================================================================
#END OF MODEL 1
#===========================================================================
#===========================================================================
#===========================================================================
#===========================================================================
#===========================================================================


#===========================================================================
# MODEL 2
#===========================================================================


# =========================================================================
# Create the Initial Training and Testing Splits
# =========================================================================

#Convert dataframe column of images into numpy array
X = np.asarray(BalancedSkinData['image'].tolist())
X = X/255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y=BalancedSkinData['dxLabel'] #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem (somthing like the todummy in pandas)
#Split to training and testing. Get a very small dataset for training as we will be 
# fitting it to many potential models. 
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)


# =========================================================================
# Apply the Keras Model Using 5 Folds
# =========================================================================
modelstarttime = datetime.datetime.now()
batch_size = 32
epochs = 50
num_folds = 5
acc_per_fold = []
loss_per_fold = []

from sklearn.model_selection import KFold
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)


all_loss = [] # to fill with the loss per epoc for each fold
all_val_loss = [] # to fill with the validation loss per epoch and fold
all_acc = [] # to fill with the accuracy per epoch and fold
all_val_acc = [] # to fill with the validation accuracy per epoch and fold
y_pred = [] # to fill with the predicition values
y_pred_classes = []
y_true = []

fold_no = 0
for train, test in kfold.split(inputs, targets):
    fold_no += 1

    
    x_train = X[train]
    y_train = Y_cat[train]
    x_test = X[test]
    y_test = Y_cat[test]
    
    # Define the model architecture
    SIZE = 32 # pixels size

    num_classes = 7

    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation="relu", 
                     input_shape=(SIZE, SIZE, 3)))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))
    
    
    model.add(Conv2D(128, (3, 3),activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, (3, 3),activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))  
    model.add(Dropout(0.3))
    model.add(Flatten())
    
    model.add(Dense(32))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='Adam', metrics=['acc'])
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    foldstarttime = datetime.datetime.now()
    
    # Fit data to model
    history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test),
        epochs=epochs,batch_size = batch_size,verbose=2)
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
   
    all_loss.append(loss)
    all_val_loss.append(val_loss)
    all_acc.append(acc)
    all_val_acc.append(val_acc)
    
    y_pred.append(model.predict(x_test))
    y_true.append(np.argmax(y_test, axis = 1))
    
    
    
    print('Fold', fold_no, 'time taken was', datetime.datetime.now() - foldstarttime)

print('Whole module time taken is', datetime.datetime.now() - modelstarttime)


# Calculate the average loss per each epoch and fold
average_folds_loss = []  
for i in range(len(all_loss[0])):
    average_folds_loss.append(statistics.mean([
        all_loss[0][i], all_loss[1][i], all_loss[2][i],
        all_loss[3][i], all_loss[4][i]]))
    
# Calculate the average validation loss per each epoch and fold
average_folds_val_loss = []  
for i in range(len(all_val_loss[0])):
    average_folds_val_loss.append(statistics.mean([
        all_val_loss[0][i], all_val_loss[1][i], all_val_loss[2][i],
        all_val_loss[3][i], all_val_loss[4][i]]))
  
# Calculate the average accuract per each epoch and fold
average_folds_all_acc = []  
for i in range(len(all_acc[0])):
    average_folds_all_acc.append(statistics.mean([
        all_acc[0][i], all_acc[1][i], all_acc[2][i],
        all_acc[3][i], all_acc[4][i]]))

# Calculate the average validation accuracy per each epoch and fold
average_folds_all_val_acc = []  
for i in range(len(all_acc[0])):
    average_folds_all_val_acc.append(statistics.mean([
        all_val_acc[0][i], all_val_acc[1][i], all_val_acc[2][i],
        all_val_acc[3][i], all_val_acc[4][i]]))
    


# plot the training and validation loss at each epoch
epochs = range(1, len(average_folds_loss) + 1)
plt.plot(epochs, average_folds_loss, 'y', label='Training loss')
plt.plot(epochs, average_folds_val_loss, 'r', label='Validation loss')
plt.title('Average Training and validation loss all fold (2nd model)')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.savefig(charts_path + '\\2Avge Training & Validation Loss per Epoch.png',
            dpi=300)

# plot the training and validation accuracy at each epoch
plt.plot(epochs, average_folds_all_acc, 'y', label='Training acc')
plt.plot(epochs, average_folds_all_val_acc, 'r', label='Validation acc')
plt.title('Average Training and validation accuracy all fold   (2nd model)')
plt.xlabel('Epochs')
plt.ylabel('Average Accuracy')
plt.legend()
plt.savefig(charts_path + '\\2Avge Training & Validation Accuracy per Epoch.png', dpi=300)


# Plot the Accuracy for each fold then average accuracy
accuracy_numbers = acc_per_fold.copy()
accuracy_numbers = accuracy_numbers + [statistics.mean(acc_per_fold)]

labels = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Average']
mpl.rcParams['font.size'] = 12.0 # set the test size
f, ax = plt.subplots(figsize=(13,5))
plt.bar(labels,accuracy_numbers, color = 'green', 
        ec = 'black')
for i in range(len(labels)):
    plt.text(i,round(accuracy_numbers[i]),
             round(accuracy_numbers[i]),
             ha = 'center', va = 'bottom')
plt.title('Accuracy % (2nd model)', fontsize = 14)
plt.savefig(charts_path + '\\2Accuracy per Fold.png', dpi=300)



# Results Correlation
#--------------------
# Combining all y_pred_classes
y_pred_classes = []
for i in range(len(y_pred)):
    y_pred_classes.append(np.argmax(y_pred[i], axis = 1))
    
# Create correlatin matrix for accuracy per each fold
cm1 = []
cm2 = []
cm3 = []
cm4 = []
cm5 = []

cm = [cm1, cm2, cm3, cm4, cm5]
for c in range(len(cm)):
    cm[c].append(confusion_matrix(y_true[c], y_pred_classes[c]))

# Create and plot average correlation matrix for prediction accuracy
cm_avg = sum(cm1 + cm2 + cm3 + cm4 + cm5) / len(cm)

# Returen the encoded values to their original names
dx_values = BalancedSkinData[['dx','dxLabel']].drop_duplicates()
cm_avg_d = pd.DataFrame(cm_avg)
cm_avg_d.columns = dx_values.dx.tolist()
cm_avg_d.index = dx_values.dx.tolist()

cm_avg_d1 = cm_avg_d.copy()
for i in cm_avg_d.columns.values:
    for a in range(len(cm_avg_d)):
        cm_avg_d1[i][a] = round((cm_avg_d[i][a] / sum(cm_avg_d[i]))*100,1)

sns.heatmap(cm_avg_d, annot=True, annot_kws={"size": 10}, cmap="YlGnBu")
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Accuracy % per Lesion Type Confusion Matrix  (2nd model)')
plt.savefig(charts_path + '\\2 Accuracy Corr matrix.png', dpi=300)


#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm_avg) / np.sum(cm_avg, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')


# Evaluate (Apply) the model on the whole dataset
csv_df_testing = csv_df.copy()
for i in range(len(dx_values)):
    csv_df_testing[csv_df_testing.dx == dx_values.dx.unique()[i]]['dxLabel'] = dx_values[dx_values.dx == dx_values.dx.unique()[i]]['dxLabel']


X_whole = np.asarray(csv_df_testing['image'].tolist())
Y_Whole = csv_df_testing['dxLabel']
Y_cat_Whole = to_categorical(Y_Whole, num_classes=7)

scores_whole = model.evaluate(X_whole, Y_cat_Whole, verbose=1)
acc_whole = scores_whole[1] * 100
print('Accuracy after applying on the whole datadet is  (2nd model)', 
      round(acc_whole,1),'%', sep ='')
loss_whole = scores_whole[0]

# END