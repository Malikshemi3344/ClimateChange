import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# countries chosen for analysis
chosenCountries = ['China', 'Canada', 'India', 'Pakistan',
                   'United Kingdom', 'Australia', 'United States', 'Nigeria']

# font style for figures
font = dict(size=25)


def getDataframes(filename):
    '''
    Takes filename as argument.\n
    Returns two dataframes, one with years as columns and second with countries as columns.
    '''

    yearData = pd.read_csv(filename, skiprows=3, index_col='Country Name')

    # drop the unnecessary columns
    yearData.drop(['Country Code', 'Indicator Name',
                   'Indicator Code'], axis=1, inplace=True)

    # filter out the chosen countries
    yearData = yearData.loc[chosenCountries]

    # drop the columns (years) which have all null values
    yearData.dropna(axis=1, how='all', inplace=True)

    return yearData, yearData.transpose()


def getCommonYears(yearList1, yearList2):
    '''
    Takes two lists of years and returns a list of common years between them.
    '''
    return sorted(list(set(yearList1) & set(yearList2)))


def selectYears(yearsList, div=5):
    '''
    Takes a list of years and selects the years that are divisible by `div`.
    '''
    return [year for year in yearsList if not (int(year) % div)]


def getCountryData(country):
    '''
    Takes country name as argument.\n
    Returns data from all the metrics corresponding to that country.
    '''

    df = pd.DataFrame()
    for _file in os.listdir('./Data/'):
        temp = getDataframes(os.path.join('./Data/', _file))[0].loc[country]

        # rename the column to the metric name
        temp.name = _file.split('.')[0]

        # join the column with the dataframe
        df = pd.concat([df, temp], axis=1)

    return df


def getCorrMap(country):
    plt.figure(figsize=(20, 18))

    # correlation for given country
    corr = getCountryData(country).corr()

    # mask to get only the bottom triangle of the correlation map
    mask = np.triu(np.ones_like(corr, dtype=bool))

    corrMap = sns.heatmap(corr, mask=mask, cmap='rocket', annot=True,
                          linewidths=5, annot_kws={'size': 25}, cbar=False)
    corrMap.set_xticklabels(labels=corrMap.get_xticklabels(), fontdict={
                            'size': 20}, rotation=60)
    corrMap.set_yticklabels(labels=corrMap.get_yticklabels(), fontdict={
                            'size': 20}, rotation=0)
    plt.title(f'Correlation Map for {country}', fontdict=dict(size=30))


# read all the files
yearGDP, countryGDP = getDataframes('./Data/GDPPerCapita$.csv')
yearCO2, countryCO2 = getDataframes('./Data/CO2EmissionsTonPerCapita.csv')
yearMethane, countryMethane = getDataframes('./Data/MethaneEmissionTotal.csv')
yearPopu, countryPopu = getDataframes('./Data/PopulationTotal.csv')
yearPower, countryPower = getDataframes(
    './Data/PowerConsumptionkWhPerCapita.csv')
yearForest, countryForest = getDataframes('./Data/ForestArea.csv')
yearArable, countryArable = getDataframes('./Data/ArableLand%.csv')
yearAgri, countryAgri = getDataframes('./Data/AgriculturalLand%.csv')


# GDP per CO2 Emission

# find common years between the two dataframes and select the years divisble by 5
selectedYears = selectYears(getCommonYears(yearCO2.columns, yearGDP.columns))

# get GDP per CO2 Emission table
GDPPerCO2 = yearGDP[selectedYears] / yearCO2[selectedYears]

fig = plt.figure(figsize=(20, 20))

# plot line for each country
for country in chosenCountries:
    ax = plt.plot(GDPPerCO2.loc[country], label=country)

plt.legend(fontsize='xx-large')
plt.title('Energy Efficiency', fontdict=font)
plt.xticks(fontsize=20)  # increase font size of x-axis labels to 20
plt.xlabel('Year', fontdict=font)
plt.yticks(fontsize=20)  # increase font size of y-axis labels to 20
plt.ylabel('GDP ($) / CO2 Emission (Ton)', fontdict=font)
plt.savefig('EneryEfficiency', bbox_inches='tight')


# Methane per Capita

# find common years between the two dataframes and select the years divisble by 5
selectedYears = selectYears(getCommonYears(yearMethane, yearPopu))

# get Methane Emission per Person
MethanePerCapita = yearMethane[selectedYears] / yearPopu[selectedYears]

# melt the dataframe so that the years are under a single column
df = MethanePerCapita.reset_index().melt(id_vars='Country Name', var_name='Year',
                                         value_name='Methane per Person')

# plot a barplot with countries on the x-axis and Methane Emission per Capita on y-axis
plt.figure(figsize=(20, 20))
sns.barplot(x='Country Name', y='Methane per Person', hue='Year', data=df)
plt.legend(fontsize='xx-large')
plt.title('Methane Emissions per Capita', fontdict=font)
plt.xticks(fontsize=20)  # increase font size of x-axis labels to 20
plt.xlabel('Year', fontdict=font)
plt.yticks(fontsize=20)  # increase font size of y-axis labels to 20
plt.ylabel('Methane (Tons) / Person', fontdict=font)
plt.savefig('MethanePerCapita', bbox_inches='tight')


# Forest Area

# melt the Forest dataframe so that multiple columns of years become one
df = yearForest[selectedYears].reset_index().melt(
    id_vars='Country Name', var_name='Year', value_name='Forest Area')

# plot a barplot
plt.figure(figsize=(20, 20))
sns.barplot(x='Country Name', y='Forest Area', hue='Year', data=df)
plt.legend(fontsize='xx-large')
plt.title('Forest Area', fontdict=font)
plt.xticks(fontsize=20)
plt.xlabel('Country', fontdict=font)
plt.yticks(fontsize=20)
plt.ylabel('Area (sq. km)', fontdict=font)
plt.savefig('ForestArea', bbox_inches='tight')


# Power Consumption Vs GDP

selectedYears = selectYears(getCommonYears(yearPower, yearGDP))

# melt the dataframes so that multiple columns of years turn into a single column
df1 = yearPower.reset_index().melt(id_vars='Country Name',
                                   var_name='Year',  value_name='Power')
df2 = yearGDP.reset_index().melt(id_vars='Country Name',
                                 var_name='Year',  value_name='GDP')

# merge the two dataframes
df = df1.merge(df2, on=['Country Name', 'Year'])

plt.figure(figsize=(20, 20))
sns.scatterplot(data=df, x='Power', y='GDP', hue='Country Name')
plt.legend(fontsize='xx-large')
plt.title('Power Consumption Vs GDP', fontdict=font)
plt.xticks(fontsize=20)  # increase font size of x-axis labels to 20
plt.xlabel('Power Consumption (kWh per Capita)', fontdict=font)
plt.yticks(fontsize=20)  # increase font size of y-axis labels to 20
plt.ylabel('GDP ($ per Capita)', fontdict=font)
plt.savefig('PowerConsumptionVsGDP', bbox_inches='tight')


# Correlation Maps

# Pakistan
getCorrMap('Pakistan')
plt.savefig('PakistanCorrMap', bbox_inches='tight')

# United States
getCorrMap('United States')
plt.savefig('USCorrMap', bbox_inches='tight')

# Nigeria
getCorrMap('Nigeria')
plt.savefig('NigeriaCorrMap', bbox_inches='tight')
