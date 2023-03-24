import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
plt.xticks(fontsize=20)
plt.xlabel('Year', fontdict=font)
plt.yticks(fontsize=20)
plt.ylabel('GDP ($) / CO2 Emission (Ton)', fontdict=font)
plt.savefig('EneryEfficiency')


# Methane per Capita

# find common years between the two dataframes and select the years divisble by 5
selectedYears = selectYears(getCommonYears(yearMethane, yearPopu))

# get Methane Emission per Person
MethanePerCapita = yearMethane[selectedYears] / yearPopu[selectedYears]

# melt the dataframe so that the years are under a single column
df = MethanePerCapita.reset_index().melt(id_vars='Country Name', var_name='Year',
                                         value_name='Methane per Person')

# plot a barplot with countries on the x-axis and
plt.figure(figsize=(20, 20))
sns.barplot(x='Country Name', y='Methane per Person', hue='Year', data=df)
plt.legend(fontsize='xx-large')
plt.title('Methane Emissions per Capita', fontdict=font)
plt.xticks(fontsize=20)  # increase font size of x-axis labels to 20
plt.xlabel('Year', fontdict=font)
plt.yticks(fontsize=20)  # increase font size of y-axis labels to 20
plt.ylabel('Methane (Tons) / Person', fontdict=font)
plt.savefig('MethanePerCapita')
