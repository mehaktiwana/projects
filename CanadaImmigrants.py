import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import folium


df_can = pd.read_excel('Canada.xlsx')
# print('Data downloaded successfully')

# print(df_can.head())

# Get basic info of data
# print(df_can.info(verbose=False))

# Get the list of columns
# print(df_can.columns)

# Get rows
# print(df_can.index)

# print(type(df_can.columns))

# convert into list
df_can.columns.tolist()
df_can.index.to_list()

# print(type(df_can.columns.tolist()))
# print(type(df_can.index.tolist()))

# print(df_can.shape)

# Clean the data set
# Remove unnecessary columns
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis = 1, inplace=True)
# print(df_can.head())

# Rename the columns
df_can.rename(columns={'OdName': 'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
# print(df_can.head())


numeric_columns = df_can.select_dtypes(include='number')
df_can['Total'] = numeric_columns.sum(axis=1)
# print(df_can['Total'].head())

# Check how many null values are there in the dataset
# print(df_can.isnull().sum())
# Great! No null values

# Get a brief summary of each column of i=our datset
# print(df_can.describe())

# Filter out data of countries and years(columns)
# print(df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]])

# Select rows
# print(df_can.loc[[0, 2, 3]])

# Set country name as index
df_can.set_index('Country', inplace=True)
# print(df_can.head())

# print(df_can.loc['Japan'])
# print(df_can.iloc[87])
# print(df_can.loc['Japan', 2013])

# print(df_can.loc['Japan',[1980, 1981, 1982, 1983, 1984]])

# convert all column names to strints
df_can.columns = list(map(str, df_can.columns))
# print(df_can.columns)

# Create a variable that stores all the years
# It will be useful later on
years = list(map(str, range(1980, 2014)))
# print(years)

# Filtering based on criteria
# condition = df_can['Continent'] == 'Asia'
# print(df_can[condition])

# print(df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')])

# Now lets see what changes we have made so far
# print('Data dimensions: ', df_can.shape)

# Visualize the data using line plots
# Line plots are best suited for trend-visualization over a period of time
haiti = df_can.loc['Haiti', years]
# print(haiti.head())
# haiti.plot()
# plt.show()
# haiti.index = haiti.index.map(int)
# haiti.plot(kind='line')
# plt.title('Immigration from Haiti')
# plt.ylabel('Number of immigrants')
# plt.xlabel('Years')
# plt.show()

#  syntax: plt.text(x, y, label)
# plt.text(2000, 6000, '2010 Earthquake')
# plt.show()

# Let's compare the number of immigrants from India and China from 1980 to 2013.
df_CI = df_can.loc[['India', 'China'], years]
# print(df_CI)

# df_CI.plot(kind='line')
# plt.show()
df_CI = df_CI.transpose()
# print(df_CI.head())

# df_CI.plot(kind='line')
# plt.title('Immigrants from India and China')
# plt.xlabel('years')
# plt.ylabel('Number of immigrants')
# plt.show()


# Compare the trend of top 5 countries that contributed the most to immigration to Canada.
df_can.sort_values(by = 'Total', ascending=False, inplace=True)
# print(df_can.head())

df_top = df_can.iloc[0:5]
df_top = df_top[years]
# print(df_top)
df_top = df_top.transpose()
# df_top.plot(kind='line')
# plt.title('Immigrants from top 5 countries')
# plt.xlabel('years')
# plt.ylabel('Number of immigrants')
# plt.show()

#Area plots, histograms and bar charts

# To check whether all columns are of string type
# print(all(isinstance(column, str) for column in df_can.columns))

# Visualizing the data
mpl.style.use('ggplot')

# Area plots
# print(df_top)

df_top.index = df_top.index.map(int)
# df_top.plot(kind='area', stacked = False, figsize = (20,10))
# plt.title('Immigration trend of top 5 countries')
# plt.ylabel('Number of Immigrants')
# plt.xlabel('Years')
# plt.show()


# passing alpha as 0.25
# df_top.plot(kind='area', stacked = False, alpha=0.25, figsize = (20,10))
# plt.title('Immigration trend of top 5 countries')
# plt.ylabel('Number of Immigrants')
# plt.xlabel('Years')
# plt.show()

# Use the scripting layer to create a stacked area plot of the 5 countries that contributed the least to immigration to Canada from 1980 to 2013. Use a transparency value of 0.45.
df_least = df_can.tail(5)
df_least = df_least[years]
df_least = df_least.transpose()
# print(df_least)

# print(all(isinstance(column, str)for column in df_least.columns))
# Convert index to integer
df_least.index = df_least.index.map(int)

# create area plot
# df_least.plot(kind='area', alpha=0.45, stacked=False)
# plt.title('Immigration trend of least 5 countries')
# plt.ylabel('Number of Immigrants')
# plt.xlabel('Years')
# plt.show()

#  Use the artist layer to create an unstacked area plot of the 5 countries that contributed the least to immigration to Canada from 1980 to 2013. Use a transparency value of 0.55
# ax = df_least.plot(kind='area', alpha=0.55, stacked=False)
# ax.set_title('Immigration Trend of 5 Countries with Least Contribution to Immigration')
# ax.set_ylabel('Number of Immigrants')
# ax.set_xlabel('Years')
# plt.show()

# Histograms
# print(df_can['2013'].head())
count, bin_edges = np.histogram(df_can['2013'])
# print(count)
# frequesncy count
# print(bin_edges)
# bin ranges, by default = 10 bins

# df_can['2013'].plot(kind = 'hist', xticks=bin_edges)
# plt.title('Histogram of Immigration from 195 countries in 2013')
# plt.xlabel('Number of Immigrants')
# plt.ylabel('Countries')
# plt.show()

# What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?
# print(df_can.loc[['Denmark', 'Norway', 'Sweden'], years])

df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
count, bin_edges = np.histogram(df_t, 15)
# df_t.plot(kind='hist', figsize=(10,6), bins=15, xticks=bin_edges,color=['coral','darkslateblue','mediumseagreen'])
# plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
# plt.ylabel('Number of Years')
# plt.xlabel('Number of Immigrants')
# plt.show()


count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0]-10
xmax = bin_edges[-1]+10
#  stacked histogram
# df_t.plot(kind='hist', bins = 15, xticks=bin_edges, stacked=True, xlim=(xmin,xmax), color=['coral','darkslateblue','mediumseagreen'])
# plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
# plt.ylabel('Number of Years')
# plt.xlabel('Number of Immigrants') 

# plt.show()

# Bar charts
# analyze data of Iceland
df_iceland = df_can.loc['Iceland', years]
# print(df_iceland.head())
# df_iceland.plot(kind='bar')
# plt.xlabel('Year')
# plt.ylabel('Number of Immigrants')
# plt.title('Icelandic Immigrants to Canada from 1980 to 2013')
# plt.show()

# Lets annotate the rapid increase in the number of immigrants
# plt.annotate('', xy=(32,70), xytext=(28,20), xycoords='data', arrowprops = dict(arrowstyle='->', color='blue', lw=2))
# plt.show()

# Pie chart
df_continents = df_can.groupby('Continent', axis=0).sum(numeric_only=True)
# print(df_continents.head())
# df_continents['Total'].plot(kind='pie', autopct='%1.1f%%', startangle=90,shadow=True)
# plt.title('Immigration to Canada by continent [1980-2013]')
# plt.axis('equal')
# plt.show()

# Make it look more appealing
# colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
# explode_list = [0.1, 0, 0, 0, 0.1, 0.1, 0.1]
# df_continents['Total'].plot(kind='pie', autopct='%1.1f%%', startangle=90, shadow=True, labels=None, pctdistance=1.12,colors=colors_list,explode=explode_list)

# plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12) 
# plt.axis('equal') 
# plt.legend(labels=df_continents.index, loc='upper left') 
# plt.show()

# df_continents['2013'].plot(kind='pie', autopct='%1.1f%%', startangle=90, pctdistance=1.12, explode=explode_list, labels=None)
# plt.title('Immigration to Canada by Continent [2013]')
# plt.axis('equal')
# plt.legend(labels=df_continents.index, loc='upper left')
# plt.show()

# Box plots
df_japan = df_can.loc[['Japan'], years].transpose()
# print(df_japan.head())
# df_japan.plot(kind='box')
# plt.title('Box plot of Japanese Immigrants from 1980-2013')
# plt.ylabel('Number of Immigrants')
# plt.show()

# print(df_CI.describe())
# df_CI.plot(kind='box')
# plt.show()

# df_CI.plot(kind='box', color='blue', vert=False)
# plt.show()


# subplots
# fig = plt.figure()
# ax0 = fig.add_subplot(1,2,1)
# ax1 = fig.add_subplot(1,2,2)

# subplot1 : Box plot
# df_CI.plot(kind='box', color='blue', figsize=(20,6),vert=False, ax=ax0)
# ax0.set_title('Box plots of Immigrants from China and India (1980-2013)')
# ax0.set_xlabel('Number of Immigrants')
# ax0.set_ylabel('Countries')

# Subplot2 : Line plot
# df_CI.plot(kind='line', figsize=(20,6),ax=ax1)
# ax1.set_title('Line plots of Immigrants from China and India(1980-2013)')
# ax1.set_ylabel('Number of Immigrants')
# ax1.set_xlabel('Years')
# plt.show()

# Create a box plot to visualize the distribution of the top 15 countries (based on total immigration) grouped by the decades 1980s, 1990s, and 2000s.
# Create dataset for top 15 continents
# print(df_can.head())
df_top15 = df_can.head(15)

# Create a list of all years in decades 80's,90's and 00's
# Slice the original df_can to create a series for each decade and sum across all years of each country
# Merge the three series into df_new

years_80s = list(map(str,range(1980,1990)))
years_90s = list(map(str,range(1990,2000)))
years_00s = list(map(str,range(2000,2010)))

df_80s = df_top15.loc[:,years_80s].sum(axis=1)
df_90s = df_top15.loc[:,years_90s].sum(axis=1)
df_00s = df_top15.loc[:,years_00s].sum(axis=1)

new_df = pd.DataFrame({'1980s': df_80s, '1990s': df_90s, '2000s': df_00s})
# print(new_df.head())

# Plot the box plots
# new_df.plot(kind='box')
# plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')
# plt.show()

# Scatter plots
# visualize the trend of total immigrantion to Canada (all countries combined) for the years 1980 - 2013.

# convert years to int type
df_tot = pd.DataFrame(df_can[years].sum(axis=0))
df_tot.index = map(int, df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns = ['years', 'total']
# print(df_tot.head())

# plot the scatter plot
# df_tot.plot(kind='scatter', x='years',y='total',figsize=(10,6),color='darkblue')
# plt.title('Total Immigration to Canada from 1980-2013')
# plt.xlabel('Year')
# plt.ylabel('Number of Immigrants')

# plt.show()

x = df_tot['years']
y = df_tot['total']
fit = np.polyfit(x,y,deg=1)
# print(fit)
# df_tot.plot(kind='scatter', x='years',y='total', color='blue')
# plt.title('Total Immigration to Canada from 1980 - 2013')
# plt.xlabel('Year')
# plt.ylabel('Number of Immigrants')
# plt.plot(x,fit[0]*x+fit[1], color='red')
# plt.annotate('y = {0:.0f}x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000,15000))
# plt.show()

# Create a scatter plot of the total immigration from Denmark, Norway, and Sweden to Canada from 1980 to 2013?
df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
# print(df_countries.head())

df_total = pd.DataFrame(df_countries.sum(axis=1))
# print(df_total.head())
df_total.reset_index(inplace=True)
df_total.columns = ['year', 'total']
df_total['year'] = df_total['year'].astype(int)
# print(df_total.head())

# df_total.plot(kind='scatter', x = 'year', y = 'total', color = 'darkblue')
# plt.title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013')
# plt.xlabel('Year')
# plt.ylabel('Number of Immigrants')
# plt.show()

# Bubble plots
# Let's analyze the effect of this crisis, and compare Argentina's immigration to that of it's neighbour Brazil.

df_can_t = df_can[years].transpose()
# print(df_can_t.head())
df_can_t.index = map(int, df_can_t.index)
df_can_t.index.name = 'Year'
df_can_t.reset_index(inplace=True)
# print(df_can_t.head())
# print(df_can_t['Brazil'].head())

# Create normalized weights
norm_brazil = (df_can_t['Brazil'] - df_can_t['Brazil'].min()) / (df_can_t['Brazil'].max() - df_can_t['Brazil'].min())

# normalize Argentina data
norm_argentina = (df_can_t['Argentina'] - df_can_t['Argentina'].min()) / (df_can_t['Argentina'].max() - df_can_t['Argentina'].min())
# Plot the data
# line 276

# ax0 = df_can_t.plot(kind='scatter', 
#                     x='Year', 
#                     y='Brazil',
#                     figsize=(14,8),
#                     alpha=0.5,
#                     s=norm_brazil*2000+10, 
#                     xlim=(1975,2015))
# ax1 = df_can_t.plot(kind='scatter',
#                     x='Year',
#                     y='Argentina',
#                     alpha=0.5,
#                     color="blue",
#                     s=norm_argentina * 2000 + 10,
#                     ax=ax0
#                     )

# ax0.set_ylabel('Number of Immigrants')
# ax0.set_title('Immigration from Brazil and Argentina from 1980 to 2013')
# ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')
# plt.show()

# reviously in this lab, we created box plots to compare immigration from China and India to Canada. Create bubble plots of immigration from China and India to visualize any differences with time from 1980 to 2013.
norm_india = (df_can_t['India'] - df_can_t['India'].min())/(df_can_t['India'].max() - df_can_t['India'].min())
norm_china = (df_can_t['China'] - df_can_t['China'].min())/(df_can_t['China'].max() - df_can_t['China'].min())


# ax0 = df_can_t.plot(kind='scatter', x = 'Year', y = 'India', alpha = 0.7, color = 'blue', s = norm_india*2000+10, xlim=(1975,2015))
# ax1 = df_can_t.plot(kind='scatter', x='Year', y='China', alpha = 0.7, color = 'green', s = norm_china*2000+10, ax=ax0)
# ax0.set_ylabel('Number of immigrants')
# ax0.set_xlabel('Immigration from India and China from 1980 to 2013')
# ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='large')
# plt.show()

# Waffle chart
df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]
# # print(df_dsn.head())

# total_values = df_dsn['Total'].sum()
# category_proportions = df_dsn['Total']/total_values

# width = 40
# height = 10
# total_number_tiles = width*height
# # print(f'Total number of tiles is {total_number_tiles}')

# tiles_per_category = (category_proportions*total_number_tiles).round().astype(int)
# # print(pd.DataFrame({'Number of tiles' : tiles_per_category}))

# # initialize the waffle chart as an empty matrix
# waffle_chart = np.zeros((height, width), dtype=np.uint)
# # define indices to loop through waffle chart
# category_index = 0
# title_index = 0
# populate the waffle chart
# for col in range(width):
#     for row in range(height):
#         title_index+=1
#         # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
#         if title_index > sum(tiles_per_category[0:category_index]):
#             #...proceed to the next category
#             category_index+=1
#                 # set the class value to an integer, which increases with class
#         waffle_chart[row,col] = category_index
# print('Waffle chart populated!')

# print(waffle_chart)

# fig = plt.figure()
# colormap = plt.cm.coolwarm
# plt.matshow(waffle_chart, cmap=colormap)
# plt.colorbar()

# ax = plt.gca()
# ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
# ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
# ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

# plt.xticks([])
# plt.yticks([])

# plt.show()

# Regression plots

df_tot = pd.DataFrame(df_can[years].sum(axis=0))
# print(df_tot.head())
df_tot.index = map(float, df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns = ['year', 'total']


# print(df_tot.head())

# plt.figure()
# sns.set_style('whitegrid')
# ax=sns.regplot(x='year', y='total', data=df_tot, color='green', scatter_kws={'s':200})
# ax.set(xlabel='Year', ylabel='Total Immigration') # add x- and y-labels
# ax.set_title('Total Immigration to Canada from 1980 - 2013') # add title
# plt.show()

# Foilum
world_map = folium.Map()
# world_map.show_in_browser()

# lets create a map centered around canada
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)
world_map.save('map.html')

# Stamen toner maps
world_map1 = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Toner')
world_map1.save('Stamen Toner Map.html')
# print('Map saved successfully')

world_map2 = folium.Map(location = [56.130, -106.35], zoom_start=4, tiles = 'Stamen Terrain')
world_map2.save('Stamen Terrain Map.html')
 
