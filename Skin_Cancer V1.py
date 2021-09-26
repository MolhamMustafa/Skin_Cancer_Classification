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

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential # To create the model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats # For statistics purpose
from sklearn.preprocessing import LabelEncoder # Convert lables into numbers
import datetime
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import itertools
        


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
        plt.clf()
    
                
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
    def Data_Exploration(self,column, filters):
        if column == 1 or column == 0:
            print()
            print('Data Summary')
            return DataUpload.recall_dataframe(column).info()
        elif filters == 0:
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
        else:
            print()
            print('Data Summary filtered on ', filters)
            print()
            for i in DataUpload.recall_dataframe(1)[filters].unique():
                print('Summary for data filtered on', i)
                print(DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1)[filters] ==  i) & (DataUpload.recall_dataframe(1)[column] !=  0)].describe())
                print()
            
    ##
    # Descriptive analysis graph
    # deal with numeric data only
    # to get the overall column plots without filters, set the 'filters' = 0
    # to get the automatic graph title, set "graph_title" = 0
    # to avoid saving the image in local drive, set "save_folder_path" = 0
    # NOTE: enter the folder path like this --> r'path'
    #
    def Des_Graphs_Num (self, column, filters, graph_title, save_folder_path, Title_Font_Size, Bar_Colour, XY_Font_Size):
        if column == 0 or column == 1:
            print('This fuction acts on a single feature only')
                            # Define the bars colours

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
                    
                    # Set the bar colours
                    if Bar_Colour == 0:
                        Bar_Colour = (0.2, 0.4, 0.6, 0.6)
                        
                    sns.distplot(DataUpload.recall_dataframe(column), color= Bar_Colour,
                    hist_kws=dict(edgecolor="black", linewidth=1))
                    plt.title(self.title, fontsize = Title_Font_Size)
                    plt.xticks(fontsize = XY_Font_Size)
                    plt.yticks(fontsize = XY_Font_Size)
                    
                    # Set the save condition
                    if save_folder_path != 0:
                        plt.savefig(save_folder_path + '\\{v} distplot.png'.format(v = column), dpi=300)
                        print('Graph saved undel the title:', '\\{v} distplot.png'.format(v = column))
                    plt.clf()

                        
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
                            
                        # Set the bar colours
                        if Bar_Colour == 0:
                            Bar_Colour = (0.2, 0.4, 0.6, 0.6)
                        
                        sns.distplot(DataUpload.recall_dataframe(0)[DataUpload.recall_dataframe(0)[filters] == f][column], color= Bar_Colour,
                        hist_kws=dict(edgecolor="black", linewidth=1))
                        plt.title(self.title, fontsize = Title_Font_Size)
                        plt.xticks(fontsize = XY_Font_Size)
                        plt.yticks(fontsize = XY_Font_Size)
                        
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
    def Des_Graphs_Cat(self, column, filters, graph_title, save_folder_path, Bar_Colour, Title_Font_Size, XY_Font_Size, Values_Font_Size):
        
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
        
        # Adjust the colour
        if Bar_Colour == 0:
            Bar_Colour = (0.2, 0.4, 0.6, 0.6)
        
        # auto set x labels fontsize
        if len(self.labels) > 7:
            mpl.rcParams['font.size'] = 10.0
            f, ax = plt.subplots(figsize=(13,5))
            
        # setting the rotaion for x axix
        x_rotation = 25
            
        plt.bar(self.labels,DataUpload.recall_dataframe(1)[column].value_counts(), color = Bar_Colour, ec = 'black')
        plt.xticks(rotation = x_rotation, fontsize = XY_Font_Size)
        plt.yticks(fontsize = XY_Font_Size)
        for i in range(len(self.labels)):
            plt.text(i,DataUpload.recall_dataframe(1)[column].value_counts()[i], DataUpload.recall_dataframe(1)[column].value_counts() [i],
             ha = 'center', va = 'bottom', fontsize = Values_Font_Size)
        #plt.ylim([0,7500])
        plt.title(self.title, fontsize = Title_Font_Size)
        plt.savefig(save_folder_path + '\\{v} frequency.png'.format(v = column), dpi=300)
        print('Graph saved undel the title:', '\\{v} frequency.png'.format(v = column))
        plt.clf()

        ##
        # Pie chart
        #
        
        # Set the graph title
        if graph_title == 0:
            self.title = column
        else:
            self.title = graph_title
        
        # auto set the lable sizes
        if len(self.labels) > 7:
            mpl.rcParams['font.size'] = 6.0 # set the test size
        
        plt.pie(DataUpload.recall_dataframe(1)[column].value_counts(), labels = self.labels, autopct='%1.1f%%', textprops={'fontsize': Values_Font_Size})
        plt.title(self.title + ' Pie Chart', fontsize = Title_Font_Size)
        #plt.title('Proportion of each Localization')
        plt.savefig(save_folder_path + '\\{v} pie chart.png'.format(v = column), dpi=300)
        print('Graph saved undel the title:', '\\{v} pie chart.png'.format(v = column))
        plt.clf()


    ##
    # Doing correlation matrix for all the data features
    #
    
    def Overall_Correlations(self, save_folder_path, Font_Size, colours):
        
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
                    ,cmap = colours)
        self.corr_mat.set_yticklabels(self.corr_mat.get_yticklabels(), rotation=45)
        
        self.corr_mat.set_xticklabels(self.corr_mat.get_xmajorticklabels(), fontsize = Font_Size) # adjust x axis font
        
        self.corr_mat.set_yticklabels(self.corr_mat.get_yticklabels(), rotation=45, fontsize = Font_Size) # adjust y axis format
        
        self.corr_mat.set_title('All features Correlation Matrix')
        
        if save_folder_path != 0:
            plt.savefig(save_folder_path + '\\Overall Corr. Matrix.png', dpi=300)
        plt.clf()
            
    ##
    # Correlation between each lesion type and the localization
    #
    
    def dx_localization_Correlations (self, save_folder_path, Colours, Font_Size, Values_Fontsize):
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
                                  annot_kws={"size": Values_Fontsize}, cmap=Colours, cbar =False) # annot_kws set the font size inside the table
        self.loc_dx_corr.set_xticklabels(self.loc_dx_corr.get_xmajorticklabels(), fontsize = Font_Size) # adjust x axis font
        
        self.loc_dx_corr.set_yticklabels(self.loc_dx_corr.get_yticklabels(), rotation=45, fontsize = Font_Size) # adjust y axis format
        
        self.loc_dx_corr.set_title('Lesion Types to Localizations Correlations')
        
        if save_folder_path != 0:
            plt.savefig(save_folder_path + '\\Lesion Types to Localization Correlations.png', dpi=300)
        plt.clf()
        
        # Print the max and min correlations for each lesion type
        for i in self.new_corrMatrix:
            print('max % in', i, 'is', max(self.new_corrMatrix[i]),
                  'and min % is', min(self.new_corrMatrix[i]))
        


DesAnalysis = Descriptive_Analysis()
#DesAnalysis.Data_Exploration('age','dx')
#DesAnalysis.Des_Graphs_Num('age','dx' ,0, Figures_path, 15, 0, 10) # Descriptive analysis graph for numerical features
#DesAnalysis.Des_Graphs_Cat('dx',0 ,0, Figures_path, 0, 11, 10, 8) # Descriptive analysis graph for categorical features
#DesAnalysis.Overall_Correlations(Figures_path, 10, "coolwarm") # Overall correlation matrix
#DesAnalysis.dx_localization_Correlations(Figures_path, "coolwarm", 7, 7) # Correlation matrix focused on dx and localization only

#YlGnBu

##
# Getting the main statistics for age after removing the 0s and null values
#
DataUpload.recall_dataframe(0)[(DataUpload.recall_dataframe(0).age != 0) &
                               (DataUpload.recall_dataframe(0).age.notnull())]['age'].describe()


##
# The relationship between gender and lesion type (dx)
#
Age_Lesion = pd.DataFrame([['bkl'], ['nv'], ['df'], ['mel'], ['vasc'], ['bcc'], ['akiec']], columns = ['dx'])
for s in ['male', 'female']:
    Age_Lesion[s] = DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1).dx == 'bkl') & (DataUpload.recall_dataframe(1).sex == s)].shape[0],\
        DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1).dx == 'nv') & (DataUpload.recall_dataframe(1).sex == s)].shape[0],\
            DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1).dx == 'df') & (DataUpload.recall_dataframe(1).sex == s)].shape[0],\
                DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1).dx == 'mel') & (DataUpload.recall_dataframe(1).sex == s)].shape[0],\
                    DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1).dx == 'vasc') & (DataUpload.recall_dataframe(1).sex == s)].shape[0],\
                        DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1).dx == 'bcc') & (DataUpload.recall_dataframe(1).sex == s)].shape[0],\
                            DataUpload.recall_dataframe(1)[(DataUpload.recall_dataframe(1).dx == 'akiec') & (DataUpload.recall_dataframe(1).sex == s)].shape[0]
                                                                                                                                               
            
Age_Lesion['male%'] = Age_Lesion[Age_Lesion.dx == 'bkl'].male.item() / (Age_Lesion[Age_Lesion.dx == 'bkl'].male.item() + Age_Lesion[Age_Lesion.dx == 'bkl'].female.item()),\
    Age_Lesion[Age_Lesion.dx == 'nv'].male.item() / (Age_Lesion[Age_Lesion.dx == 'nv'].male.item() + Age_Lesion[Age_Lesion.dx == 'nv'].female.item()),\
        Age_Lesion[Age_Lesion.dx == 'df'].male.item() / (Age_Lesion[Age_Lesion.dx == 'df'].male.item() + Age_Lesion[Age_Lesion.dx == 'df'].female.item()),\
            Age_Lesion[Age_Lesion.dx == 'mel'].male.item() / (Age_Lesion[Age_Lesion.dx == 'mel'].male.item() + Age_Lesion[Age_Lesion.dx == 'mel'].female.item()),\
                Age_Lesion[Age_Lesion.dx == 'vasc'].male.item() / (Age_Lesion[Age_Lesion.dx == 'vasc'].male.item() + Age_Lesion[Age_Lesion.dx == 'vasc'].female.item()),\
                    Age_Lesion[Age_Lesion.dx == 'bcc'].male.item() / (Age_Lesion[Age_Lesion.dx == 'bcc'].male.item() + Age_Lesion[Age_Lesion.dx == 'bcc'].female.item()),\
                        Age_Lesion[Age_Lesion.dx == 'akiec'].male.item() / (Age_Lesion[Age_Lesion.dx == 'akiec'].male.item() + Age_Lesion[Age_Lesion.dx == 'akiec'].female.item())                                                                                                           

Age_Lesion['female%'] = Age_Lesion[Age_Lesion.dx == 'bkl'].female.item() / (Age_Lesion[Age_Lesion.dx == 'bkl'].male.item() + Age_Lesion[Age_Lesion.dx == 'bkl'].female.item()),\
    Age_Lesion[Age_Lesion.dx == 'nv'].female.item() / (Age_Lesion[Age_Lesion.dx == 'nv'].male.item() + Age_Lesion[Age_Lesion.dx == 'nv'].female.item()),\
        Age_Lesion[Age_Lesion.dx == 'df'].female.item() / (Age_Lesion[Age_Lesion.dx == 'df'].male.item() + Age_Lesion[Age_Lesion.dx == 'df'].female.item()),\
            Age_Lesion[Age_Lesion.dx == 'mel'].female.item() / (Age_Lesion[Age_Lesion.dx == 'mel'].male.item() + Age_Lesion[Age_Lesion.dx == 'mel'].female.item()),\
                Age_Lesion[Age_Lesion.dx == 'vasc'].female.item() / (Age_Lesion[Age_Lesion.dx == 'vasc'].male.item() + Age_Lesion[Age_Lesion.dx == 'vasc'].female.item()),\
                    Age_Lesion[Age_Lesion.dx == 'bcc'].female.item() / (Age_Lesion[Age_Lesion.dx == 'bcc'].male.item() + Age_Lesion[Age_Lesion.dx == 'bcc'].female.item()),\
                        Age_Lesion[Age_Lesion.dx == 'akiec'].female.item() / (Age_Lesion[Age_Lesion.dx == 'akiec'].male.item() + Age_Lesion[Age_Lesion.dx == 'akiec'].female.item())                                                                                                           

           
Age_Lesion[['dx','male%','female%']].plot(
    x = 'dx',
    kind = 'barh',
    stacked = False,
    title = 'Gender Proportion per Lesion Type',
    mark_right = True)
plt.savefig(Figures_path + '\\Age_Lesion', dpi=300)
 
# Age_Lesion_total = Age_Lesion["male%"] + Age_Lesion["female%"]
# Age_Lesion_rel = Age_Lesion[Age_Lesion.columns[1:]].div(Age_Lesion_total, 0)*100
  
# for n in Age_Lesion_rel:
#     for i, (cs, ab, pc) in enumerate(zip(Age_Lesion.iloc[:, 1:].cumsum(1)[n], 
#                                          Age_Lesion[n], Age_Lesion_rel[n])):
#         plt.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%', 
#                  va = 'center', ha = 'center')

# plt.savefig(Figures_path + '\\Age_Lesion', dpi=300)





# =========================================================================
# Data Preprocessing
# =========================================================================

##
# Encode targer (transform to categorical data)
#

class Data_Preprocessing:
    # Encoding a single categorical feature
    def Encoding(self, column):
        self.le = LabelEncoder()
        self.le.fit(DataUpload.recall_dataframe(1)[column])
        LabelEncoder()
        print(list(self.le.classes_)) # print the unique values per encoding
        self.processed_data = DataUpload.recall_dataframe(1)
        self.processed_data['{C}Label'.format(C = column)] = self.le.transform(self.processed_data[column]) # Encoding the required column
        print()
        print(column, 'encoded successfully, and below is a sample of it')
        print(self.processed_data[[column, '{C}Label'.format(C = column)]].sample(20))

    # Splitting the data into training-validation and testing
    def Train_Test_Split(self, test_ratio):
        self.train_val_ratio = 1 - test_ratio
        
        # Create the trainig-validation dataset for each lesion type (dx)
        self.akiec_size = round(self.processed_data[self.processed_data.dx == 'akiec'].shape[0] * self.train_val_ratio)
        self.train_akiec = self.processed_data[self.processed_data.dx == 'akiec'].sample(n = self.akiec_size)
        
        self.bcc_size = round(self.processed_data[self.processed_data.dx == 'bcc'].shape[0] * self.train_val_ratio)
        self.train_bcc = self.processed_data[self.processed_data.dx == 'bcc'].sample(n = self.bcc_size)
        
        self.mel_size = round(self.processed_data[self.processed_data.dx == 'mel'].shape[0] * self.train_val_ratio)
        self.train_mel = self.processed_data[self.processed_data.dx == 'mel'].sample(n = self.mel_size)
        
        self.bkl_size = round(self.processed_data[self.processed_data.dx == 'bkl'].shape[0] * self.train_val_ratio)
        self.train_bkl = self.processed_data[self.processed_data.dx == 'bkl'].sample(n = self.bkl_size)
        
        self.df_size = round(self.processed_data[self.processed_data.dx == 'df'].shape[0] * self.train_val_ratio)
        self.train_df = self.processed_data[self.processed_data.dx == 'df'].sample(n = self.df_size)
        
        self.nv_size = round(self.processed_data[self.processed_data.dx == 'nv'].shape[0] * self.train_val_ratio)
        self.train_nv = self.processed_data[self.processed_data.dx == 'nv'].sample(n = self.nv_size)
        
        self.vasc_size = round(self.processed_data[self.processed_data.dx == 'vasc'].shape[0] * self.train_val_ratio)
        self.train_vasc = self.processed_data[self.processed_data.dx == 'vasc'].sample(n = self.vasc_size)
        
        # Concatinate the training-validation dataset
        self.train_val_data = pd.concat([self.train_akiec, self.train_bcc, self.train_mel,self.train_bkl, self.train_df, self.train_nv,self.train_vasc])
    
        # Create the test data
        
        self.test_data = self.processed_data[~self.processed_data.image_id.isin(self.train_val_data.image_id.values.tolist())]
    
        print('Data split successfully to train-validation and test data with proportions', self.train_val_ratio, 'and', test_ratio)
        print('Training-validation data size is', self.train_val_data.shape[0])
        print('Testing data size is', self.test_data.shape[0])


    # Balancing the data
    def Balancing_Data(self, Sample_Volume):
        
        # Split the training-validation data into 7 datasets (1 for each class)
        self.dx0 = DataPreprocessing.ProcessedData('train')[DataPreprocessing.ProcessedData('train').dxLabel == 0]
        self.dx1 = DataPreprocessing.ProcessedData('train')[DataPreprocessing.ProcessedData('train').dxLabel == 1]
        self.dx2 = DataPreprocessing.ProcessedData('train')[DataPreprocessing.ProcessedData('train').dxLabel == 2]
        self.dx3 = DataPreprocessing.ProcessedData('train')[DataPreprocessing.ProcessedData('train').dxLabel == 3]
        self.dx4 = DataPreprocessing.ProcessedData('train')[DataPreprocessing.ProcessedData('train').dxLabel == 4]
        self.dx5 = DataPreprocessing.ProcessedData('train')[DataPreprocessing.ProcessedData('train').dxLabel == 5]
        self.dx6 = DataPreprocessing.ProcessedData('train')[DataPreprocessing.ProcessedData('train').dxLabel == 6]
        
        # Balance the lesion types
        from sklearn.utils import resample
        
        self.dx0Balanced = resample(self.dx0, replace=True, n_samples = Sample_Volume, random_state = 42)
        self.dx1Balanced = resample(self.dx1, replace=True, n_samples = Sample_Volume, random_state = 42)
        self.dx2Balanced = resample(self.dx2, replace=True, n_samples = Sample_Volume, random_state = 42)
        self.dx3Balanced = resample(self.dx3, replace=True, n_samples = Sample_Volume, random_state = 42)
        self.dx4Balanced = resample(self.dx4, replace=True, n_samples = Sample_Volume, random_state = 42)
        self.dx5Balanced = resample(self.dx5, replace=True, n_samples = Sample_Volume, random_state = 42)
        self.dx6Balanced = resample(self.dx6, replace=True, n_samples = Sample_Volume, random_state = 42)
        
        # Concatenate the balanced dx types in a new dataset
        self.BalancedData = pd.concat([self.dx0Balanced, self.dx1Balanced, self.dx2Balanced, 
                                      self.dx3Balanced, self.dx4Balanced, self.dx5Balanced, self.dx6Balanced])
        
        print('Trainig-validation data balanced where wach lesion count is', Sample_Volume)
        print()
        print('Lesioin frequency before balancing the data:')
        print(DataPreprocessing.ProcessedData('train').dx.value_counts())
        print()
        print('Count per lesion:')
        print(self.BalancedData.dx.value_counts())
        

        
    # Recall_the_data (options are 'train','test','all','balanced data')
    def ProcessedData(self, RequiredData):
        if RequiredData == 'train':
            return self.train_val_data
        elif RequiredData == 'test':
            return self.test_data
        elif RequiredData == 'all':
            return self.processed_data
        elif RequiredData == 'balanced data':
            return self.BalancedData
        else:
            print('valid arguments are "train", "test", "all", or "balanced data"')
        

DataPreprocessing = Data_Preprocessing()
DataPreprocessing.Encoding('dx') # encoding the target (dx)
DataPreprocessing.Train_Test_Split(0.25) # splitting the data into training-validation (75%) and testing (25%)
# pp = DataPreprocessing.ProcessedData('all') # recalling processed data (dx encoded)
# dd = DataPreprocessing.ProcessedData('train') # recalling training-validation data
# tt = DataPreprocessing.ProcessedData('test') # recalling testing data
DataPreprocessing.Balancing_Data(500)

# =========================================================================
# DEVELOPING THE MODEL
# =========================================================================
 
##
# Build the model
#

class CNN_Model:
    
    # Set the training and validation dataset
    def train_val_split(self, selected_data, validation_ratio):
        self.number_of_classes = len(selected_data.dxLabel.unique())
        
        self.selected_data = selected_data
        self.X = np.asarray(selected_data['image'].tolist())
        self.X = self.X/255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
        self.Y=selected_data['dxLabel'] #Assign label values to Y
        self.Y_cat = self.Y.to_numpy() # convert the y into ndarray
        #self.Y = selected_data['dxLabel']
        #self.Y_cat = to_categorical(self.Y, num_classes=self.number_of_classes)
        
        
        #Split to training and testing. Get a very small dataset for training as we will be 
        # fitting it to many potential models. 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y_cat, test_size=validation_ratio, random_state=42)
        self.y_train, self.y_test = self.y_train.flatten(), self.y_test.flatten()##############################
        
        print()
        print('Split done with ratios', 1-validation_ratio, ' for training data and ', validation_ratio, 'for validation data')

    # recall the split data
    def split_data(self, RequiredData):
        if RequiredData == 'x_train':
            return self.x_train
        elif RequiredData == 'x_test':
            return self.x_test
        elif RequiredData == 'y_train':
            return self.y_train
        elif RequiredData == 'y_test':
            return self.y_test
        else:
            print('valid arguments for "split_data" are "x_train","x_test","y_train","y_test"')
    
    # Compare different models without CV
    def Fit_Model(self, Optimizer, Loss, n_epochs):
        self.model_names = len(models)
        for m,c in zip(models, range(self.model_names)):
            m.compile(optimizer=Optimizer, loss=Loss, metrics=['accuracy'])
            r = m.fit(CNNModel.split_data('x_train'), CNNModel.split_data('y_train'),
                  validation_data=(CNNModel.split_data('x_test'), CNNModel.split_data('y_test')), epochs=n_epochs)
            
            
            # Create Folder for each model
            try:
                os.mkdir(Figures_path + '\\Model {count}'.format(count = c))
                print('Model {count} folder created'.format(count = c))
            except:
                print('Model {count} folder already exists'.format(count = c))
            
            
            # Plot and save the loss for each model
            plt.plot(r.history['loss'], label='loss')
            plt.plot(r.history['val_loss'], label='val_loss')
            plt.legend()
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count} Loss'.format(count = c), dpi=300)
            plt.clf()
            
            # Plot and save accuracy per iteration
            plt.plot(r.history['accuracy'], label='acc')
            plt.plot(r.history['val_accuracy'], label='val_acc')
            plt.legend()
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count} Accuracy'.format(count = c), dpi=300)
            plt.clf()
     
    # Compare different models using Cross-Validation Teqnique (CV)
    def Fit_Model_CV(self, Optimizer, Loss, n_epochs, num_folds, data_augmentation, batch_size):
        
        # Define the CV folds
        
        # First concatinate the training and validation data
        self.inputs = np.concatenate((CNNModel.split_data('x_train'), CNNModel.split_data('x_test')), axis=0)
        self.targets = np.concatenate((CNNModel.split_data('y_train'), CNNModel.split_data('y_test')), axis=0)
        
        # Define the K-fold Cross Validator
        self.kfold = KFold(n_splits=num_folds, shuffle=True)
        #self.folds = StratifiedKFold(n_splits = num_folds)
        
        
        self.model_names = len(models)
        for m,c in zip(models, range(self.model_names)):
            
            self.start_time = datetime.datetime.now()
            
            self.acc_per_fold = []
            self.loss_per_fold = []
            
            self.all_loss = [] # to fill with the loss per epoc for each fold
            self.all_val_loss = [] # to fill with the validation loss per epoch and fold
            self.all_acc = [] # to fill with the accuracy per epoch and fold
            self.all_val_acc = [] # to fill with the validation accuracy per epoch and fold
            self.y_pred = [] # to fill with the predicition values
            self.y_pred_classes = []
            self.y_true = []
            
            self.fold_no = 0
            
            
            for train_index, test_index in self.kfold.split(self.inputs):
                self.fold_no += 1
                
                
                self.x_train = self.inputs[train_index]
                self.x_test = self.inputs[test_index]
                self.y_train = self.targets[train_index]
                self.y_test = self.targets[test_index]
                
                m.compile(optimizer=Optimizer, loss=Loss, metrics=['accuracy'])
                r = m.fit(self.x_train, self.y_train,
                      validation_data=(self.x_test, self.y_test), epochs=n_epochs)
                
                if data_augmentation == True:
                    self.batch_size = batch_size 
                    self.data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
                    self.train_generator = self.data_generator.flow(self.x_train, self.y_train, batch_size)
                    self.steps_per_epoch = self.x_train.shape[0] // batch_size ## // discards the remainder
                    r = m.fit(self.train_generator, validation_data=(self.x_test, self.y_test), steps_per_epoch = self.steps_per_epoch, epochs=n_epochs)

            
            
                # Create Folder for each model
                try:
                    os.mkdir(Figures_path + '\\Model {count}'.format(count = c))
                    print('Model {count} folder created'.format(count = c))
                except:
                    print('Model {count} folder already exists'.format(count = c))
                
                
                # Plot and save the loss for each model
                plt.plot(r.history['loss'], label='loss')
                plt.plot(r.history['val_loss'], label='val_loss')
                plt.legend()
                plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count}, fold {f} Loss'.format(count = c, f = self.fold_no), dpi=300)
                plt.clf()
                
                # Plot and save accuracy per iteration
                plt.plot(r.history['accuracy'], label='acc')
                plt.plot(r.history['val_accuracy'], label='val_acc')
                plt.legend()
                plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count}, fold {f} Accuracy'.format(count = c, f = self.fold_no), dpi=300)
                plt.clf()
            
            
                # Create and plot the folds' average accuracy and loss for each model 
                self.scores = m.evaluate(self.inputs[test_index], self.targets[test_index], verbose=0)
                #print(f'Score for fold {fold_no}: {m.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
                self.acc_per_fold.append(self.scores[1] * 100)
                self.loss_per_fold.append(self.scores[0])
                
                
                self.loss = r.history['loss']
                self.val_loss = r.history['val_loss']
                self.acc = r.history['accuracy']
                self.val_acc = r.history['val_accuracy']
               
                self.all_loss.append(self.loss)
                self.all_val_loss.append(self.val_loss)
                self.all_acc.append(self.acc)
                self.all_val_acc.append(self.val_acc)
                
                self.y_pred.append(m.predict(self.x_test))
                self.y_true.append(self.y_test)
                #self.y_true.append(np.argmax(self.y_test, axis = 1))
                print()
                print('Model {count}, fold {f} completed'.format(count = c, f = self.fold_no))
                print()
            
            # Create correlation Matrix with the results
            self.average_folds_loss = []  
            for i in range(len(self.all_loss[0])):
                self.average_folds_loss.append(statistics.mean([
                    self.all_loss[0][i], self.all_loss[1][i], self.all_loss[2][i],
                    self.all_loss[3][i], self.all_loss[4][i]]))
                
            # Calculate the average validation loss per each epoch and fold
            self.average_folds_val_loss = []  
            for i in range(len(self.all_val_loss[0])):
                self.average_folds_val_loss.append(statistics.mean([
                    self.all_val_loss[0][i], self.all_val_loss[1][i], self.all_val_loss[2][i],
                    self.all_val_loss[3][i], self.all_val_loss[4][i]]))
              
            # Calculate the average accuract per each epoch and fold
            self.average_folds_all_acc = []  
            for i in range(len(self.all_acc[0])):
                self.average_folds_all_acc.append(statistics.mean([
                    self.all_acc[0][i], self.all_acc[1][i], self.all_acc[2][i],
                    self.all_acc[3][i], self.all_acc[4][i]]))
            
            # Calculate the average validation accuracy per each epoch and fold
            self.average_folds_all_val_acc = []  
            for i in range(len(self.all_acc[0])):
                self.average_folds_all_val_acc.append(statistics.mean([
                    self.all_val_acc[0][i], self.all_val_acc[1][i], self.all_val_acc[2][i],
                    self.all_val_acc[3][i], self.all_val_acc[4][i]]))
                
            
            
            # plot the training and validation loss at each epoch
            self.epochs = range(1, len(self.average_folds_loss) + 1)
            plt.plot(self.epochs, self.average_folds_loss, 'y', label='Training loss')
            plt.plot(self.epochs, self.average_folds_val_loss, 'r', label='Validation loss')
            plt.title('Average Training and validation loss all fold')
            plt.xlabel('Epochs')
            plt.ylabel('Average Loss')
            plt.legend()
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count} average loss'.format(count = c), dpi=300)
            plt.clf()
            
            # plot the training and validation accuracy at each epoch
            plt.plot(self.epochs, self.average_folds_all_acc, 'y', label='Training acc')
            plt.plot(self.epochs, self.average_folds_all_val_acc, 'r', label='Validation acc')
            plt.title('Average Training and validation accuracy all fold')
            plt.xlabel('Epochs')
            plt.ylabel('Average Accuracy')
            plt.legend()
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count} average accuracy'.format(count = c), dpi=300)
            plt.clf()
            
            # Plot the Accuracy for each fold then average accuracy
            self.accuracy_numbers = self.acc_per_fold.copy()
            self.accuracy_numbers = self.accuracy_numbers + [statistics.mean(self.acc_per_fold)]
            
            self.labels = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Average']
            mpl.rcParams['font.size'] = 12.0 # set the test size
            f, ax = plt.subplots(figsize=(13,5))
            plt.bar(self.labels,self.accuracy_numbers, color = 'green', 
                    ec = 'black')
            for i in range(len(self.labels)):
                plt.text(i,round(self.accuracy_numbers[i]),
                         round(self.accuracy_numbers[i]),
                         ha = 'center', va = 'bottom')
            plt.title('Accuracy %', fontsize = 14)
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count} accuracy per fold'.format(count = c), dpi=300)
            plt.clf()
            
            
            # Results Correlation
            #--------------------
            # Combining all y_pred_classes
            self.y_pred_classes = []
            for i in range(len(self.y_pred)):
                self.y_pred_classes.append(np.argmax(self.y_pred[i], axis = 1))
                
            # Create correlatin matrix for accuracy per each fold
            self.cm1 = []
            self.cm2 = []
            self.cm3 = []
            self.cm4 = []
            self.cm5 = []
            
            self.cm = [self.cm1, self.cm2, self.cm3, self.cm4, self.cm5]
            for c0 in range(len(self.cm)):
                self.cm[c0].append(confusion_matrix(self.y_true[c0], self.y_pred_classes[c0]))
            
            # Create and plot average correlation matrix for prediction accuracy
            self.cm_avg = (sum(self.cm1 + self.cm2 + self.cm3 + self.cm4 + self.cm5) / len(self.cm)).round()
            
            # Returen the encoded values to their original names
            self.dx_values = self.selected_data[['dx','dxLabel']].drop_duplicates()
            self.cm_avg_d = pd.DataFrame(self.cm_avg)
            self.cm_avg_d.columns = self.dx_values.dx.tolist()
            self.cm_avg_d.index = self.dx_values.dx.tolist()
            self.cm_avg_d.to_excel(Figures_path + '\\Model {count}'.format(count = c) + '\\Confusion Matrix on Training-Val data.xlsx')
            
            
            sns.heatmap(self.cm_avg_d, annot=True, fmt=".1f", annot_kws={"size": 10}, cmap="YlGnBu")
            plt.xlabel('Predictions')
            plt.ylabel('Actuals')
            plt.title('Accuracy Confusion Matrix')
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count} accuracy correlation matrix'.format(count = c), dpi=300)
            plt.clf()
            
            self.cm_avg_d1 = self.cm_avg_d.copy()
            for i in self.cm_avg_d.columns.values:
                for a in range(len(self.cm_avg_d)):
                    self.cm_avg_d1[i][a] = round((self.cm_avg_d[i][a] / sum(self.cm_avg_d[i]))*100,1)
            
            sns.heatmap(self.cm_avg_d, annot=True, fmt=".1f", annot_kws={"size": 10}, cmap="YlGnBu")
            plt.xlabel('Predictions')
            plt.ylabel('Actuals')
            plt.title('Accuracy % per Lesion Type Confusion Matrix')
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model {count} accuracy % correlation matrix'.format(count = c), dpi=300)
            plt.clf()
            
            #PLot fractional incorrect misclassifications
            self.incorr_fraction = 1 - np.diag(self.cm_avg) / np.sum(self.cm_avg, axis=1)
            self.incorr_fraction = self.incorr_fraction.tolist()
            plt.bar(np.arange(7), self.incorr_fraction)
            plt.xlabel('True Label', fontsize = 8)
            plt.ylabel('Fraction of incorrect predictions', fontsize = 8)
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Model test Misclassification % per class', dpi=300)
            plt.clf()
            
            # Evaluate (Apply) the model on the whole dataset
            self.csv_df_testing = DataPreprocessing.ProcessedData('test')
            
            #csv_df_testing = csv_df.copy()
            for i in range(len(self.dx_values)):
                self.csv_df_testing[self.csv_df_testing.dx == self.dx_values.dx.unique()[i]]['dxLabel'] = self.dx_values[self.dx_values.dx == self.dx_values.dx.unique()[i]]['dxLabel']
            
            
            self.X_whole = np.asarray(self.csv_df_testing['image'].tolist())
            self.Y_Whole = np.asarray(self.csv_df_testing['dxLabel'].tolist())
            #Y_Whole = csv_df_testing['dxLabel']
            #Y_cat_Whole = to_categorical(Y_Whole, num_classes=7)
            
            self.scores_whole = model0.evaluate(self.X_whole, self.Y_Whole, verbose=1)
            #results = model0.evaluate(X_whole, Y_cat_Whole)
            #results = model0.evaluate(X_whole, Y_Whole)
            
            self.acc_whole = self.scores_whole[1] * 100
            print()
            print()
            print('Model {count} Accuracy after applying on the testing datadet is '.format(count = c), 
                  round(self.acc_whole,1),'%', sep ='')
            self.loss_whole = self.scores_whole[0]
            print('Model {count} loss after applying on the testing datadet is '.format(count = c), 
                  round(self.loss_whole,1),'%', sep ='')
            print()
            print()
            
            # Confusion matrix when applying the model on the testing dataaet
            self.y_Whole_perd = m.predict(self.X_whole)
            self.y_pred_classes = np.argmax(self.y_Whole_perd, axis = 1)
            
            self.Y_True = self.Y_Whole
            
            self.Con_matrix_tesing_dataset = []
            self.Con_matrix_tesing_dataset = confusion_matrix(self.Y_True, self.y_pred_classes)
            # self.Con_matrix_tesing_dataset.append(pd.DataFrame(confusion_matrix(self.Y_True, self.y_pred_classes)))
            self.Con_matrix_tesing_dataset = pd.DataFrame(self.Con_matrix_tesing_dataset)
            self.Con_matrix_tesing_dataset.columns = self.dx_values.dx.tolist()
            self.Con_matrix_tesing_dataset.index = self.dx_values.dx.tolist()
            self.Con_matrix_tesing_dataset.to_excel(Figures_path + '\\Model {count}'.format(count = c) + '\\Confusion Matrix on TEST DATASET.xlsx')
            
            sns.heatmap(self.Con_matrix_tesing_dataset, annot=True, fmt=".1f", annot_kws={"size": 10}, cmap="YlGnBu")
            plt.xlabel('Predictions')
            plt.ylabel('Actuals')
            plt.title('Accuracy Confusion Matrix')
            plt.savefig(Figures_path + '\\Model {count}'.format(count = c) + '\\Confusion Matrix on TESTING DATASET', dpi=300)
            plt.clf()
            print()
            print('Model {count} COMPLETED'.format(count = c))
            print('Model {count} took: '.format(count = c), datetime.datetime.now() - self.start_time)
            print()
            
CNNModel = CNN_Model()
CNNModel.train_val_split(DataPreprocessing.ProcessedData('balanced data') ,0.20)



##
# Building the models' architectures
#

K = len(set(CNNModel.split_data('y_train')))
print('Number of classes:', K)

## 
# First architectures
#
i = Input(shape=CNNModel.split_data('x_train')[0].shape)
x = Conv2D(32, (2, 2), strides=1, activation='relu')(i)
x = Conv2D(64, (2, 2), strides=1, activation='relu')(x)
x = Conv2D(128, (2, 2), strides=1, activation='relu')(x)
x = Conv2D(32, (2, 2), strides=1, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.6)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model0 = Model(i, x)

##
# 2nd architecture
#
i = Input(shape = CNNModel.split_data('x_train')[0].shape)
x = Conv2D(32,(2,2), strides = 1, activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(x)
x = Conv2D(32,(2,2), strides = 1, activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64,(2,2), strides = 1, activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(x)
x = Conv2D(64,(2,2), strides = 1, activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128,(2,2), strides = 1, activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(x)
x = Conv2D(128,(2,2), strides = 1, activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation = 'softmax')(x)

model1 = Model(i, x)


## 
# 3rd architecture
#
i = Input(shape=CNNModel.split_data('x_train')[0].shape)
x = (Conv2D(256, (3, 3), activation="relu"))(i)
#model.add(BatchNormalization())
x = (MaxPool2D(pool_size=(2, 2)))(x)
x = (Dropout(0.1))(x)


x = (Conv2D(128, (3, 3),activation='relu'))(x)
#model.add(BatchNormalization())
x = (MaxPool2D(pool_size=(2, 2)))(x)
x = (Dropout(0.1))(x)

x = (Conv2D(64, (3, 3),activation='relu'))(x)
#model.add(BatchNormalization())
x = (MaxPool2D(pool_size=(2, 2)))(x)
x = (Dropout(0.1))(x)
x = (Flatten())(x)

x = (Dense(32))(x)
x = (Dense(K, activation='softmax'))(x)

model2 = Model(i, x)


models = [model0, model1, model2]



#CNNModel.Fit_Model('adam', 'sparse_categorical_crossentropy', 50)
CNNModel.Fit_Model_CV('adam', 'sparse_categorical_crossentropy', 50, 5, False, 512)
 

    
##
# Hyperparameters
#

# Get the numner of classes
K = len(set(CNNModel.split_data('y_train')))
print('Number of classes:', K)


all_loss = []
all_val_loss = []
all_acc = []
all_val_acc = []
architecture = []

filters = [2,3,5]
strides = [1]
DropOut1 = [0.1,0.2,0.3]
#Dense = [512, 1024]
itr = 0

# Hyperparameters
for f in filters:
    for s in strides:
        for D in DropOut1:
            itr += 1
            stepstarttime = datetime.datetime.now()
            i = Input(shape=CNNModel.split_data('x_train')[0].shape)
            x = Conv2D(256, (f, f), strides = s, activation="relu", padding = 'same')(i)
            #model.add(BatchNormalization())
            x = MaxPool2D(pool_size=(2, 2))(x)
            x = Dropout(D)(x)
            
            
            x = Conv2D(128, (f, f), strides = s,activation='relu', padding = 'same')(x)
            #model.add(BatchNormalization())
            x = MaxPool2D(pool_size=(2, 2))(x)
            x = Dropout(D)(x)
            
            x = Conv2D(64, (f, f), strides = s,activation='relu', padding = 'same')(x)
            #model.add(BatchNormalization())
            x = MaxPool2D(pool_size=(2, 2))(x)
            x = Dropout(D)(x)
            
            x = Flatten()(x)
            
            x = Dense(32)(x)
            x = Dense(K, activation='softmax')(x)
                
            model1 = Model(i, x)
            
            models = [model1]
            
            # Compile and fit
            # Note: make sure you are using the GPU for this!
            model1.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            r = model1.fit(CNNModel.split_data('x_train'), CNNModel.split_data('y_train'),
                          validation_data=(CNNModel.split_data('x_test'), CNNModel.split_data('y_test')), epochs=15)
            all_loss.append(r.history['loss'][-1])
            all_val_loss.append(r.history['val_loss'][-1])
            all_acc.append(r.history['accuracy'][-1])
            all_val_acc.append(r.history['val_accuracy'][-1])
            architecture.append('Itteration ' + str(itr) + ' : filter Size: ' +str(f) + ', strides: ' + str(s) + ', Dropout 1: ' + str(D)+' ,time taken: ' +  str(datetime.datetime.now() - stepstarttime))
            print()
            print('Iteration', itr, 'finished')
            print('iteration architecture:', architecture[-1])
            print('time taken:', datetime.datetime.now() - stepstarttime)
            print()




# END