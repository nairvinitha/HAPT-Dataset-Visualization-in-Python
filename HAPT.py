#Import the required modules
import tensorflow as tf
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

#print the entire data from the HAPT Dataset
print("Loading data from the HAPT dataset") 
print(os.listdir(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set"))
#To view all the files in the dataset, process the for loop as below
for dirname, _, filename in os.walk(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set"):
    for name in filename:
        print(os.path.join(dirname, name))

# Loading the training dataset 'TrainX'
TrainX = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Train\X_train.txt", 
                     sep="\s+", header=None)
#Load all the features from the file "features.txt"
list_of_features = list()
with open(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\features.txt") as filename:
    list_of_features = [line.strip() for line in filename.readlines()]
print("Total number of features: {}".format(len(list_of_features)))

TrainX.columns = [list_of_features]

#For each row in the "subject_id_train.txt" file, the volunteer who performed the activity for each window sample is identified.
# The volunteer Ids range from 1 to 30.
TrainX["Volunteer"] = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Train\subject_id_train.txt",
                                  sep="\s+",header=None)

# Loading the Training labels 'TrainY'
TrainY = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Train\y_train.txt", 
                     sep="\s+", header=None, names=["ActivityLabel"], squeeze = True)
                                  
#Mapping the Training labels with the activities                                 
TrainYlabel_Mapping = TrainY.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING', 7:'STAND_TO_SIT', 8:'SIT_TO_STAND',9:'SIT_TO_LIE',
                       10:'LIE_TO_SIT', 11:'STAND_TO_LIE', 12:'LIE_TO_STAND'})
                                  
# Combining all columns in a single dataframe
training_set = TrainX 
training_set['ActivityLabel'] = TrainY
training_set['ActivityName'] = TrainYlabel_Mapping
training_set

# Loading the test dataset 'TestX'
TestX = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Test\X_test.txt", 
                    sep="\s+", header=None)
TestX.columns = [list_of_features]

#For each row in the "subject_id_test.txt" file, the volunteer who performed the activity for each window sample is identified.
# The volunteer Ids range from 1 to 30.
TestX["Volunteer"] = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Test\subject_id_test.txt",
                                 sep="\s+",header=None)

# Loading the Training labels 'TrainY'
TestY = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Test\y_test.txt", 
                     sep="\s+", header=None, names=["ActivityLabel"], squeeze = True)
#Mapping the Training labels with the activities                                 
TestYlabel_Mapping = TestY.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                       4:'SITTING', 5:'STANDING',6:'LAYING', 7:'STAND_TO_SIT', 8:'SIT_TO_STAND',9:'SIT_TO_LIE',
                       10:'LIE_TO_SIT', 11:'STAND_TO_LIE', 12:'LIE_TO_STAND'})


# put all columns in a single dataframe
testing_set = TestX
testing_set['ActivityLabel'] = TestY
#testing_set['ActivityLabel']
testing_set['ActivityName'] = TestYlabel_Mapping
testing_set

# We look for any duplicate/null values
print('No of duplicates in train: {}'.format(sum(TrainX.duplicated())))
print('No of duplicates in test : {}'.format(sum(TestX.duplicated())))
print('We have {} NaN/Null values in train'.format(TrainX.isnull().values.sum()))
print('We have {} NaN/Null values in test'.format(TestX.isnull().values.sum()))

#Saving these dataframes into another CSV file
TrainX.to_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Train.csv", index=False)
TestX.to_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Test.csv", index=False)

#Import the training and testing dataset
TrainData = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Train.csv")
TestData = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\Test.csv")
print(TrainData.shape, TestData.shape)

column_data = TrainData.columns

# Removing unnecessary special characters from column names
column_data = column_data.str.replace('[()]','')
column_data = column_data.str.replace('[-]', '')
column_data = column_data.str.replace('[,]','')


TrainData.column_data = column_data
TestData.column_data = column_data

TrainData.column_data
TestData.column_data

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Verdana"

plt.figure(figsize=(20,10))
plt.title("Activities performed by the volunteers", fontsize=22)
sns.countplot(x="Volunteer",hue="ActivityName", data=TrainData)
plt.show()

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Verdana"

plt.figure(figsize=(20,10))
plt.title("Data provided by each Volunteer", fontsize=22)
sns.countplot(x="Volunteer",hue="ActivityName", data=TestData)
plt.show()


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# Plotting data
label_counts = TrainData['ActivityName'].value_counts()

# Get colors
x = label_counts.shape[0]
colourmap = plt.get_cmap('Reds')
colours = [mlt.colors.to_hex(colourmap(col)) for col in np.arange(0, 1.01, 1/(x-1))]

# Create plot
ShowData = go.Bar(x = label_counts.index,
              y = label_counts,
              marker = dict(color = colours))

PlotLayout = go.Layout(title = 'Distribution of Activities',
                   xaxis = dict(title = 'Activity  Name'),
                   yaxis = dict(title = 'Count Values'))

fig = go.Figure(data=[ShowData], layout=PlotLayout)
fig.show()

# for plotting purposes taking datapoints of each activity to a different dataframe
df1 = TrainData[TrainData['ActivityLabel']==1]
df2 = TrainData[TrainData['ActivityLabel']==2]
df3 = TrainData[TrainData['ActivityLabel']==3]
df4 = TrainData[TrainData['ActivityLabel']==4]
df5 = TrainData[TrainData['ActivityLabel']==5]
df6 = TrainData[TrainData['ActivityLabel']==6]
df7 = TrainData[TrainData['ActivityLabel']==7]
df8 = TrainData[TrainData['ActivityLabel']==8]
df9 = TrainData[TrainData['ActivityLabel']==9]
df10 = TrainData[TrainData['ActivityLabel']==10]
df11 = TrainData[TrainData['ActivityLabel']==11]
df12 = TrainData[TrainData['ActivityLabel']==12]

plt.figure(figsize=(14,7))
plt.subplot(4,3,1)
plt.title('Stationary Activities')
sns.distplot(df4['tBodyAccMag-Mean-1'],color = 'orange',hist = False, label = 'Sitting')
sns.distplot(df5['tBodyAccMag-Mean-1'],color = 'purple',hist = False,label = 'Standing')
sns.distplot(df6['tBodyAccMag-Mean-1'],color = 'blue',hist = False, label = 'Laying')
plt.axis([-1.08, -0.2, 0.1, 20])
plt.legend(loc='center')

plt.subplot(4,3,2)
plt.title('Moving Activities')
sns.distplot(df1['tBodyAccMag-Mean-1'],color = 'orange',hist = False, label = 'Walking')
sns.distplot(df2['tBodyAccMag-Mean-1'],color = 'purple',hist = False,label = 'Walking Up')
sns.distplot(df3['tBodyAccMag-Mean-1'],color = 'blue',hist = False, label = 'Walking down')
plt.legend(loc='center right')

plt.subplot(4,3,3)
plt.title('Transitional Activities')
sns.distplot(df1['tBodyAccMag-Mean-1'],color = 'orange',hist = False, label = 'STAND_TO_SIT')
sns.distplot(df2['tBodyAccMag-Mean-1'],color = 'purple',hist = False,label = 'SIT_TO_STAND')
sns.distplot(df3['tBodyAccMag-Mean-1'],color = 'blue',hist = False, label = 'SIT_TO_LIE')
plt.legend(loc='center right')

plt.subplot(4,3,4)
plt.title('Transitional Activities')
sns.distplot(df1['tBodyAccMag-Mean-1'],color = 'orange',hist = False, label = 'LIE_TO_SIT')
sns.distplot(df2['tBodyAccMag-Mean-1'],color = 'purple',hist = False,label = 'STAND_TO_LIE')
sns.distplot(df3['tBodyAccMag-Mean-1'],color = 'blue',hist = False, label = 'LIE_TO_STAND')
plt.legend(loc='center right')



plt.tight_layout()
plt.show()

Activities = pd.read_csv(r"C:\Users\vinit\Desktop\EHU\Semester 2\Data Visualization\Coursework 2\HAPT Data Set\activity_labels.txt",
                         sep='\s+', names=('ActivityID', 'ActivityName'), header=None)
Activities


# Plotting various attributes of the dataset against activities 1-12
def plotgraph_FeaturesVsActivities(attributes, title, sharex=True, sharey=True):
    fig, axes = plt.subplots(nrows=4, ncols=3, sharey=sharey, sharex=sharex)
    fig.set_size_inches(40,40)
    activity_type, counts = np.unique(Activities.ActivityName.values, return_counts=True)
    # The activities (list for corresponding to graph)
    New_Activity = Activities.ActivityName
    
    # give the plot a title
    plt.suptitle(title, fontsize=42)

    # initialize locations. 
    xaxis = 0
    yaxis = 0

    for activity_number in range(1, len(counts)+1):
        # plot the current activity
        TrainData[TrainData['ActivityLabel'] == activity_number][variables].plot(
            use_index=False, ax=axes[yaxis, xaxis])
        
        # set the activity title for each chart
        axes[yaxis, xaxis].set_title(New_Activity[activity_number-1])
         # update the x & y locations for subplots 
        if xaxis < 2: 
            xaxis += 1
        else:
            xaxis = 0
            yaxis += 1
            
attributes = ['fBodyAcc-Mean-1', 'fBodyAcc-Mean-2', 'fBodyAcc-Mean-3']
title = "Plotting the frequency of 3 Axis Body Acceleration Mean Values against the Activity performed"

plotgraph_FeaturesVsActivities(attributes, title)
plt.savefig("LinePlot.png", format="png")


#Visualizing the Acceleration Magnitude along with the Activities performed
plt.figure(figsize=(7,5))
sns.boxplot(x='ActivityName', y='tBodyAccMag-Mean-1',data=TrainData, showfliers=False, saturation=1)
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='ActivityName', y='fBodyAccMag-Mean-1',data=TrainData, showfliers=False, saturation=1)
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='ActivityName', y='tXAxisAcc-AngleWRTGravity-1',data=TrainData, showfliers=False, saturation=1)
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=90)
plt.show()

#Implementing T-SNE on the data
def TSNEVisualization(Xinfo, yinfo, perplexities, n_iter=1000, img_name_prefix='T-SNE'):
        
    for idx,perplex in enumerate(perplexities):
        # perform t-sne
        print("\n Applying T-SNE with perplexity {} and with {} iterations at max".format(perplex, n_iter))
        reducedX = TSNE(verbose=2, perplexity=perplex).fit_transform(Xinfo)
        print('Completed!')
        
        # prepare the data for seaborn         
        print("Plotting the T-SNE visualization..")
        df = pd.DataFrame({'x':reducedX[:,0], 'y':reducedX[:,1] ,'label':yinfo})
        # draw the plot in appropriate place in the grid
        sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, height=8,\
                   palette="Set1",markers=['o', 'x', '^', '+', '*', '8', 's', 'P', 'D', 'v', '1','2'])
        plt.title("perplexity : {} and max_iter : {}".format(perplex, n_iter))
        img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplex, n_iter)
        plt.savefig(img_name)
        plt.show()
        print("Completed!")
        
        
X_pre_tsne = TrainData.drop(['Volunteer', 'ActivityLabel','ActivityName'], axis=1)
y_pre_tsne = TrainData["ActivityName"]
TSNEVisualization(Xinfo = X_pre_tsne, yinfo=y_pre_tsne, perplexities =[5,10,15])


# Drop the "Volunteer", "ActivityLabel" and ActivityName" columns from "TrainData" so that we have only floating point values
TrainData = TrainData.drop(['Volunteer', 'ActivityLabel','ActivityName'], axis=1)
TrainData


#Feature distribution visualization
Newdf = TrainData

sns.set(rc={'figure.figsize':(20,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in Newdf.columns[0:10]:
    index = index + 1
    fig = sns.kdeplot(Newdf[i] , shade=True, color=colours[index])
plt.xlabel("Features", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("Feature Distribution", fontsize=22)
plt.grid(True)
plt.annotate("Stationary Activities", xy=(-0.1,21), xytext=(-0.6, 26), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.show(fig)

index = -1
for i in Newdf.columns[10:20]:
    index = index + 1
    fig = sns.kdeplot(Newdf[i] , shade=True, color=colours[index])
    
plt.xlabel("Features", fontsize=14) 
plt.ylabel("Value", fontsize=14)
plt.title("Feature Distribution", fontsize=22)
plt.grid(True)
plt.show(fig)

index = -1
for i in Newdf.columns[20:30]:
    index = index + 1
    fig = sns.kdeplot(Newdf[i] , shade=True, color=colours[index])
plt.xlabel("Features", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("Feature Distribution", fontsize=22)
plt.grid(True)
plt.annotate("Moving Activities", xy=(0,2), xytext=(0.2, 3), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.show(fig)

#Applying PCA in 3 dimensions
pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(TrainData)
y = TrainY

labeldata = [('Walking', 1), ('Walking Upstairs', 2), ('Walking Downstairs', 3), 
          ('Sitting', 4), ('Standing', 5), ('Laying', 6),
          ('Stand_to_Sit', 7), ('Sit_to_Stand', 8), ('Sit_to_Lie', 9), 
          ('Lie_to_Sit', 10), ('Stand_to_Lie', 11),('Lie_to_Stand', 12) ]

fig = plt.figure(figsize=(10, 8))
ax = Axes3D(fig, elev=-150, azim=130)
sc= ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y, cmap='Set1', edgecolor='k', s=40)

ax.set_xlabel("PCA 1", fontsize=14) 
ax.set_ylabel("PCA 2", fontsize=14) 
ax.set_zlabel("PCA 3", fontsize=14) 
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# create the events marking the x data points
colorcode = [sc.cmap(sc.norm(i)) for i in [1,2,3,4,5,6,7,8,9,10,11,12]]
layoutplot = [plt.Line2D([],[], ls="", marker='.', 
                mec='k', mfc=c, mew=.1, ms=20) for c in colorcode]
ax.legend(layoutplot, [l[0] for l in labeldata], 
          loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_title('3 dimensional PCA visualization', fontsize=22) 
plt.show()



